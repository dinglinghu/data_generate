"""
RLHF数据质量验证器
实现数据验证、异常检测和质量控制功能
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class RLHFDataQualityValidator:
    """RLHF数据质量验证器"""
    
    def __init__(self, config_manager):
        """
        初始化数据质量验证器
        
        Args:
            config_manager: 配置管理器
        """
        self.config_manager = config_manager
        
        # 加载数据质量配置
        self.quality_config = config_manager.config.get('data_quality', {})
        
        # 验证规则配置
        self.validation_rules = self.quality_config.get('validation_rules', {})
        self.completeness_checks = self.quality_config.get('completeness_checks', {})
        self.anomaly_detection = self.quality_config.get('anomaly_detection', {})
        
        # 物理约束配置
        self.physical_constraints = {
            'position_bounds': {
                'min_altitude': 200.0,  # km
                'max_altitude': 50000.0,  # km
                'earth_radius': 6371.0  # km
            },
            'velocity_limits': {
                'max_orbital_velocity': 15.0,  # km/s
                'max_missile_velocity': 8.0   # km/s
            },
            'attitude_constraints': {
                'quaternion_norm_tolerance': 0.01
            },
            'temporal_constraints': {
                'max_time_gap': 3600.0,  # 秒
                'min_time_step': 1.0     # 秒
            }
        }
        
        # 统计信息
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'anomaly_detections': 0,
            'data_quality_score': 1.0
        }
        
        # 历史数据用于异常检测
        self.historical_data = []
        self.max_history_size = 1000
        
        logger.info("🔍 RLHF数据质量验证器初始化完成")
    
    def validate_rlhf_data_point(self, data_point) -> Dict[str, Any]:
        """
        验证RLHF数据点
        
        Args:
            data_point: RLHF数据点
            
        Returns:
            验证结果
        """
        try:
            validation_result = {
                'is_valid': True,
                'validation_score': 1.0,
                'errors': [],
                'warnings': [],
                'anomalies': [],
                'quality_metrics': {}
            }
            
            # 更新统计
            self.validation_stats['total_validations'] += 1
            
            # 1. 数据完整性检查
            completeness_result = self._check_data_completeness(data_point)
            validation_result['quality_metrics']['completeness'] = completeness_result
            
            if not completeness_result['is_complete']:
                validation_result['is_valid'] = False
                validation_result['errors'].extend(completeness_result['missing_fields'])
            
            # 2. 数据格式验证
            format_result = self._validate_data_format(data_point)
            validation_result['quality_metrics']['format'] = format_result
            
            if not format_result['is_valid']:
                validation_result['is_valid'] = False
                validation_result['errors'].extend(format_result['format_errors'])
            
            # 3. 物理约束检查
            physics_result = self._check_physical_constraints(data_point)
            validation_result['quality_metrics']['physics'] = physics_result
            
            if not physics_result['is_valid']:
                validation_result['warnings'].extend(physics_result['constraint_violations'])
            
            # 4. 时间一致性检查
            temporal_result = self._check_temporal_consistency(data_point)
            validation_result['quality_metrics']['temporal'] = temporal_result
            
            if not temporal_result['is_consistent']:
                validation_result['warnings'].extend(temporal_result['inconsistencies'])
            
            # 5. 异常检测
            if self.anomaly_detection.get('statistical_outliers', False):
                anomaly_result = self._detect_anomalies(data_point)
                validation_result['anomalies'] = anomaly_result['anomalies']
                validation_result['quality_metrics']['anomaly_score'] = anomaly_result['anomaly_score']
                
                if anomaly_result['anomalies']:
                    self.validation_stats['anomaly_detections'] += 1
            
            # 6. 计算总体质量分数
            validation_result['validation_score'] = self._calculate_quality_score(validation_result)
            
            # 更新统计
            if validation_result['is_valid']:
                self.validation_stats['passed_validations'] += 1
            else:
                self.validation_stats['failed_validations'] += 1
            
            # 更新历史数据
            self._update_historical_data(data_point, validation_result)
            
            logger.debug(f"数据验证完成: 有效={validation_result['is_valid']}, "
                        f"分数={validation_result['validation_score']:.3f}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {
                'is_valid': False,
                'validation_score': 0.0,
                'errors': [f'validation_error: {str(e)}'],
                'warnings': [],
                'anomalies': [],
                'quality_metrics': {}
            }
    
    def _check_data_completeness(self, data_point) -> Dict[str, Any]:
        """检查数据完整性"""
        try:
            required_fields = self.completeness_checks.get('required_fields', [])
            optional_fields = self.completeness_checks.get('optional_fields', [])
            missing_threshold = self.completeness_checks.get('missing_data_threshold', 0.05)
            
            missing_fields = []
            present_fields = []
            
            # 检查必需字段
            for field in required_fields:
                if self._check_field_exists(data_point, field):
                    present_fields.append(field)
                else:
                    missing_fields.append(field)
            
            # 检查可选字段
            optional_present = 0
            for field in optional_fields:
                if self._check_field_exists(data_point, field):
                    optional_present += 1
            
            # 计算完整性分数
            required_completeness = len(present_fields) / len(required_fields) if required_fields else 1.0
            optional_completeness = optional_present / len(optional_fields) if optional_fields else 1.0
            
            overall_completeness = 0.8 * required_completeness + 0.2 * optional_completeness
            
            is_complete = (len(missing_fields) == 0 and 
                          overall_completeness >= (1.0 - missing_threshold))
            
            return {
                'is_complete': is_complete,
                'completeness_score': overall_completeness,
                'missing_fields': missing_fields,
                'present_fields': present_fields,
                'required_completeness': required_completeness,
                'optional_completeness': optional_completeness
            }
            
        except Exception as e:
            logger.error(f"完整性检查失败: {e}")
            return {
                'is_complete': False,
                'completeness_score': 0.0,
                'missing_fields': ['completeness_check_error'],
                'present_fields': []
            }
    
    def _validate_data_format(self, data_point) -> Dict[str, Any]:
        """验证数据格式"""
        try:
            format_errors = []
            
            # 检查基本结构
            if not hasattr(data_point, 'state') or not hasattr(data_point, 'action'):
                format_errors.append('missing_basic_structure')
            
            # 检查状态格式
            if hasattr(data_point, 'state'):
                state_errors = self._validate_state_format(data_point.state)
                format_errors.extend(state_errors)
            
            # 检查动作格式
            if hasattr(data_point, 'action'):
                action_errors = self._validate_action_format(data_point.action)
                format_errors.extend(action_errors)
            
            # 检查奖励格式
            if hasattr(data_point, 'reward'):
                if not isinstance(data_point.reward, (int, float)):
                    format_errors.append('invalid_reward_type')
                elif not np.isfinite(data_point.reward):
                    format_errors.append('invalid_reward_value')
            
            # 检查时间戳格式
            if hasattr(data_point, 'timestamp'):
                if not isinstance(data_point.timestamp, datetime):
                    format_errors.append('invalid_timestamp_type')
            
            return {
                'is_valid': len(format_errors) == 0,
                'format_errors': format_errors
            }
            
        except Exception as e:
            logger.error(f"格式验证失败: {e}")
            return {
                'is_valid': False,
                'format_errors': [f'format_validation_error: {str(e)}']
            }
    
    def _validate_state_format(self, state: Dict[str, Any]) -> List[str]:
        """验证状态格式"""
        errors = []
        
        try:
            # 检查卫星位置格式
            if 'satellite_positions' in state:
                positions = state['satellite_positions']
                if not isinstance(positions, list):
                    errors.append('invalid_satellite_positions_type')
                else:
                    for i, pos in enumerate(positions):
                        if not isinstance(pos, list) or len(pos) != 3:
                            errors.append(f'invalid_satellite_position_{i}')
                        elif not all(isinstance(x, (int, float)) and np.isfinite(x) for x in pos):
                            errors.append(f'invalid_satellite_position_values_{i}')
            
            # 检查导弹位置格式
            if 'missile_positions' in state:
                positions = state['missile_positions']
                if not isinstance(positions, list):
                    errors.append('invalid_missile_positions_type')
                else:
                    for i, pos in enumerate(positions):
                        if not isinstance(pos, list) or len(pos) != 3:
                            errors.append(f'invalid_missile_position_{i}')
                        elif not all(isinstance(x, (int, float)) and np.isfinite(x) for x in pos):
                            errors.append(f'invalid_missile_position_values_{i}')
            
            # 检查可见性矩阵格式
            if 'visibility_matrix' in state:
                matrix = state['visibility_matrix']
                if not isinstance(matrix, list):
                    errors.append('invalid_visibility_matrix_type')
                else:
                    for i, row in enumerate(matrix):
                        if not isinstance(row, list):
                            errors.append(f'invalid_visibility_matrix_row_{i}')
                        elif not all(isinstance(x, (int, bool)) for x in row):
                            errors.append(f'invalid_visibility_matrix_values_{i}')
            
        except Exception as e:
            errors.append(f'state_format_error: {str(e)}')
        
        return errors
    
    def _validate_action_format(self, action: Dict[str, Any]) -> List[str]:
        """验证动作格式"""
        errors = []
        
        try:
            # 检查卫星动作格式
            if 'satellite_actions' in action:
                sat_actions = action['satellite_actions']
                if not isinstance(sat_actions, dict):
                    errors.append('invalid_satellite_actions_type')
                else:
                    for sat_id, sat_action in sat_actions.items():
                        if not isinstance(sat_action, dict):
                            errors.append(f'invalid_satellite_action_{sat_id}')
            
            # 检查任务动作格式
            if 'mission_actions' in action:
                mission_actions = action['mission_actions']
                if not isinstance(mission_actions, dict):
                    errors.append('invalid_mission_actions_type')
                else:
                    # 检查目标分配格式
                    if 'target_assignments' in mission_actions:
                        assignments = mission_actions['target_assignments']
                        if not isinstance(assignments, list):
                            errors.append('invalid_target_assignments_type')
                        else:
                            for i, assignment in enumerate(assignments):
                                if not isinstance(assignment, dict):
                                    errors.append(f'invalid_target_assignment_{i}')
                                elif not all(key in assignment for key in ['satellite_id', 'target_id']):
                                    errors.append(f'incomplete_target_assignment_{i}')
            
        except Exception as e:
            errors.append(f'action_format_error: {str(e)}')
        
        return errors
    
    def _check_physical_constraints(self, data_point) -> Dict[str, Any]:
        """检查物理约束"""
        try:
            constraint_violations = []
            
            if hasattr(data_point, 'state'):
                state = data_point.state
                
                # 检查位置约束
                if self.validation_rules.get('position_bounds', False):
                    pos_violations = self._check_position_bounds(state)
                    constraint_violations.extend(pos_violations)
                
                # 检查速度约束
                if self.validation_rules.get('velocity_limits', False):
                    vel_violations = self._check_velocity_limits(state)
                    constraint_violations.extend(vel_violations)
                
                # 检查姿态约束
                if self.validation_rules.get('attitude_constraints', False):
                    att_violations = self._check_attitude_constraints(state)
                    constraint_violations.extend(att_violations)
            
            return {
                'is_valid': len(constraint_violations) == 0,
                'constraint_violations': constraint_violations
            }
            
        except Exception as e:
            logger.error(f"物理约束检查失败: {e}")
            return {
                'is_valid': False,
                'constraint_violations': [f'physics_check_error: {str(e)}']
            }
    
    def _check_position_bounds(self, state: Dict[str, Any]) -> List[str]:
        """检查位置边界"""
        violations = []
        
        try:
            earth_radius = self.physical_constraints['position_bounds']['earth_radius']
            min_altitude = self.physical_constraints['position_bounds']['min_altitude']
            max_altitude = self.physical_constraints['position_bounds']['max_altitude']
            
            # 检查卫星位置
            if 'satellite_positions' in state:
                for i, pos in enumerate(state['satellite_positions']):
                    if len(pos) >= 3:
                        x, y, z = pos[0], pos[1], pos[2]
                        distance = np.sqrt(x*x + y*y + z*z)
                        altitude = distance - earth_radius
                        
                        if altitude < min_altitude:
                            violations.append(f'satellite_{i}_altitude_too_low: {altitude:.1f}km')
                        elif altitude > max_altitude:
                            violations.append(f'satellite_{i}_altitude_too_high: {altitude:.1f}km')
            
            # 检查导弹位置
            if 'missile_positions' in state:
                for i, pos in enumerate(state['missile_positions']):
                    if len(pos) >= 3:
                        x, y, z = pos[0], pos[1], pos[2]
                        distance = np.sqrt(x*x + y*y + z*z)
                        altitude = distance - earth_radius
                        
                        if altitude < -1.0:  # 允许轻微的地下位置（数值误差）
                            violations.append(f'missile_{i}_underground: {altitude:.1f}km')
                        elif altitude > max_altitude:
                            violations.append(f'missile_{i}_altitude_too_high: {altitude:.1f}km')
            
        except Exception as e:
            violations.append(f'position_bounds_error: {str(e)}')
        
        return violations
    
    def _check_velocity_limits(self, state: Dict[str, Any]) -> List[str]:
        """检查速度限制"""
        violations = []
        
        try:
            max_orbital_vel = self.physical_constraints['velocity_limits']['max_orbital_velocity']
            max_missile_vel = self.physical_constraints['velocity_limits']['max_missile_velocity']
            
            # 检查卫星速度
            if 'satellite_velocities' in state:
                for i, vel in enumerate(state['satellite_velocities']):
                    if len(vel) >= 3:
                        vx, vy, vz = vel[0], vel[1], vel[2]
                        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
                        
                        if speed > max_orbital_vel:
                            violations.append(f'satellite_{i}_speed_too_high: {speed:.2f}km/s')
            
            # 检查导弹速度
            if 'missile_velocities' in state:
                for i, vel in enumerate(state['missile_velocities']):
                    if len(vel) >= 3:
                        vx, vy, vz = vel[0], vel[1], vel[2]
                        speed = np.sqrt(vx*vx + vy*vy + vz*vz)
                        
                        if speed > max_missile_vel:
                            violations.append(f'missile_{i}_speed_too_high: {speed:.2f}km/s')
            
        except Exception as e:
            violations.append(f'velocity_limits_error: {str(e)}')
        
        return violations
    
    def _check_attitude_constraints(self, state: Dict[str, Any]) -> List[str]:
        """检查姿态约束"""
        violations = []
        
        try:
            tolerance = self.physical_constraints['attitude_constraints']['quaternion_norm_tolerance']
            
            # 检查卫星姿态
            if 'satellite_attitudes' in state:
                for i, att in enumerate(state['satellite_attitudes']):
                    if len(att) >= 4:
                        q0, q1, q2, q3 = att[0], att[1], att[2], att[3]
                        norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
                        
                        if abs(norm - 1.0) > tolerance:
                            violations.append(f'satellite_{i}_quaternion_norm_invalid: {norm:.3f}')
            
        except Exception as e:
            violations.append(f'attitude_constraints_error: {str(e)}')
        
        return violations
    
    def _check_temporal_consistency(self, data_point) -> Dict[str, Any]:
        """检查时间一致性"""
        try:
            inconsistencies = []
            
            if not hasattr(data_point, 'timestamp'):
                inconsistencies.append('missing_timestamp')
                return {
                    'is_consistent': False,
                    'inconsistencies': inconsistencies
                }
            
            current_time = data_point.timestamp
            
            # 检查与历史数据的时间间隔
            if self.historical_data:
                last_time = self.historical_data[-1]['timestamp']
                time_gap = (current_time - last_time).total_seconds()
                
                max_gap = self.physical_constraints['temporal_constraints']['max_time_gap']
                min_step = self.physical_constraints['temporal_constraints']['min_time_step']
                
                if time_gap > max_gap:
                    inconsistencies.append(f'time_gap_too_large: {time_gap:.1f}s')
                elif time_gap < min_step:
                    inconsistencies.append(f'time_step_too_small: {time_gap:.1f}s')
            
            return {
                'is_consistent': len(inconsistencies) == 0,
                'inconsistencies': inconsistencies
            }
            
        except Exception as e:
            logger.error(f"时间一致性检查失败: {e}")
            return {
                'is_consistent': False,
                'inconsistencies': [f'temporal_check_error: {str(e)}']
            }
    
    def _detect_anomalies(self, data_point) -> Dict[str, Any]:
        """检测异常"""
        try:
            anomalies = []
            anomaly_score = 0.0
            
            if len(self.historical_data) < 10:
                # 历史数据不足，无法进行异常检测
                return {
                    'anomalies': [],
                    'anomaly_score': 0.0
                }
            
            # 统计异常检测
            if self.anomaly_detection.get('statistical_outliers', False):
                stat_anomalies = self._detect_statistical_outliers(data_point)
                anomalies.extend(stat_anomalies)
            
            # 物理异常检测
            if self.anomaly_detection.get('physical_constraints', False):
                phys_anomalies = self._detect_physical_anomalies(data_point)
                anomalies.extend(phys_anomalies)
            
            # 时间异常检测
            if self.anomaly_detection.get('temporal_anomalies', False):
                temp_anomalies = self._detect_temporal_anomalies(data_point)
                anomalies.extend(temp_anomalies)
            
            # 计算异常分数
            anomaly_score = len(anomalies) / 10.0  # 简化的异常分数
            
            return {
                'anomalies': anomalies,
                'anomaly_score': min(1.0, anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return {
                'anomalies': [f'anomaly_detection_error: {str(e)}'],
                'anomaly_score': 1.0
            }
    
    def _detect_statistical_outliers(self, data_point) -> List[str]:
        """检测统计异常值"""
        outliers = []
        
        try:
            if not hasattr(data_point, 'reward'):
                return outliers
            
            # 收集历史奖励数据
            historical_rewards = [record['reward'] for record in self.historical_data 
                                if 'reward' in record]
            
            if len(historical_rewards) < 5:
                return outliers
            
            # 计算统计量
            mean_reward = np.mean(historical_rewards)
            std_reward = np.std(historical_rewards)
            
            # 3-sigma规则检测异常
            if abs(data_point.reward - mean_reward) > 3 * std_reward:
                outliers.append(f'reward_outlier: {data_point.reward:.3f} vs mean {mean_reward:.3f}')
            
        except Exception as e:
            outliers.append(f'statistical_outlier_error: {str(e)}')
        
        return outliers
    
    def _detect_physical_anomalies(self, data_point) -> List[str]:
        """检测物理异常"""
        anomalies = []
        
        try:
            if not hasattr(data_point, 'state'):
                return anomalies
            
            state = data_point.state
            
            # 检测位置跳跃
            if 'satellite_positions' in state and self.historical_data:
                last_state = self.historical_data[-1].get('state', {})
                last_positions = last_state.get('satellite_positions', [])
                
                current_positions = state['satellite_positions']
                
                for i, (curr_pos, last_pos) in enumerate(zip(current_positions, last_positions)):
                    if len(curr_pos) >= 3 and len(last_pos) >= 3:
                        distance = np.sqrt(sum((c - l)**2 for c, l in zip(curr_pos, last_pos)))
                        
                        # 检测异常大的位置变化（假设最大速度15km/s，时间间隔300s）
                        max_distance = 15.0 * 300.0  # km
                        if distance > max_distance:
                            anomalies.append(f'satellite_{i}_position_jump: {distance:.1f}km')
            
        except Exception as e:
            anomalies.append(f'physical_anomaly_error: {str(e)}')
        
        return anomalies
    
    def _detect_temporal_anomalies(self, data_point) -> List[str]:
        """检测时间异常"""
        anomalies = []
        
        try:
            if not hasattr(data_point, 'timestamp') or not self.historical_data:
                return anomalies
            
            current_time = data_point.timestamp
            last_time = self.historical_data[-1]['timestamp']
            
            # 检测时间倒退
            if current_time < last_time:
                anomalies.append(f'time_regression: {current_time} < {last_time}')
            
        except Exception as e:
            anomalies.append(f'temporal_anomaly_error: {str(e)}')
        
        return anomalies
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """计算质量分数"""
        try:
            score = 1.0
            
            # 基于错误数量降低分数
            error_count = len(validation_result.get('errors', []))
            warning_count = len(validation_result.get('warnings', []))
            anomaly_count = len(validation_result.get('anomalies', []))
            
            # 错误严重影响分数
            score -= error_count * 0.3
            
            # 警告轻微影响分数
            score -= warning_count * 0.1
            
            # 异常影响分数
            score -= anomaly_count * 0.05
            
            # 基于质量指标调整分数
            quality_metrics = validation_result.get('quality_metrics', {})
            
            if 'completeness' in quality_metrics:
                completeness_score = quality_metrics['completeness'].get('completeness_score', 1.0)
                score *= completeness_score
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"质量分数计算失败: {e}")
            return 0.0
    
    def _check_field_exists(self, data_point, field_path: str) -> bool:
        """检查字段是否存在"""
        try:
            parts = field_path.split('.')
            current = data_point
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                elif isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            
            return current is not None
            
        except Exception:
            return False
    
    def _update_historical_data(self, data_point, validation_result: Dict[str, Any]):
        """更新历史数据"""
        try:
            record = {
                'timestamp': data_point.timestamp if hasattr(data_point, 'timestamp') else datetime.now(),
                'reward': data_point.reward if hasattr(data_point, 'reward') else 0.0,
                'state': data_point.state if hasattr(data_point, 'state') else {},
                'validation_score': validation_result['validation_score']
            }
            
            self.historical_data.append(record)
            
            # 限制历史数据大小
            if len(self.historical_data) > self.max_history_size:
                self.historical_data = self.historical_data[-self.max_history_size:]
                
        except Exception as e:
            logger.error(f"历史数据更新失败: {e}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        try:
            total = self.validation_stats['total_validations']
            
            if total == 0:
                return {
                    'total_validations': 0,
                    'success_rate': 0.0,
                    'average_quality_score': 0.0
                }
            
            success_rate = self.validation_stats['passed_validations'] / total
            
            # 计算平均质量分数
            if self.historical_data:
                quality_scores = [record['validation_score'] for record in self.historical_data 
                                if 'validation_score' in record]
                avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            else:
                avg_quality = 0.0
            
            return {
                'total_validations': total,
                'passed_validations': self.validation_stats['passed_validations'],
                'failed_validations': self.validation_stats['failed_validations'],
                'success_rate': success_rate,
                'anomaly_detections': self.validation_stats['anomaly_detections'],
                'average_quality_score': avg_quality,
                'historical_data_size': len(self.historical_data)
            }
            
        except Exception as e:
            logger.error(f"验证统计计算失败: {e}")
            return {'total_validations': 0}
    
    def reset_statistics(self):
        """重置统计信息"""
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'anomaly_detections': 0,
            'data_quality_score': 1.0
        }
        self.historical_data = []
        logger.info("验证统计信息已重置")
