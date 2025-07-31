"""
RLHF奖励函数计算器
实现多目标奖励函数计算，包括跟踪性能、资源效率、任务完成度
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class RLHFRewardCalculator:
    """RLHF奖励函数计算器"""
    
    def __init__(self, config_manager):
        """
        初始化奖励计算器
        
        Args:
            config_manager: 配置管理器
        """
        self.config_manager = config_manager
        
        # 加载奖励配置
        self.reward_config = config_manager.config.get('rlhf_data_collection', {}).get('reward_components', {})
        
        # 奖励权重配置
        self.reward_weights = {
            'tracking_performance': 0.4,
            'resource_efficiency': 0.3,
            'mission_completion': 0.3
        }
        
        # 性能基准值
        self.performance_baselines = {
            'max_power_per_satellite': 100.0,  # 最大功率(W)
            'max_communication_bandwidth': 1000.0,  # 最大通信带宽(Mbps)
            'max_computational_load': 100.0,  # 最大计算负载(%)
            'target_response_time': 60.0,  # 目标响应时间(秒)
            'min_tracking_accuracy': 0.95,  # 最小跟踪精度
            'max_false_alarm_rate': 0.05  # 最大虚警率
        }
        
        logger.info("🎯 RLHF奖励计算器初始化完成")
    
    def calculate_total_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                              base_data: Dict[str, Any], next_state: Dict[str, Any] = None) -> float:
        """
        计算总奖励
        
        Args:
            state: 当前状态
            action: 执行的动作
            base_data: 基础数据
            next_state: 下一状态
            
        Returns:
            总奖励值
        """
        try:
            total_reward = 0.0
            reward_breakdown = {}
            
            # 1. 跟踪性能奖励
            tracking_reward = self._calculate_tracking_performance_reward(state, action, base_data)
            total_reward += self.reward_weights['tracking_performance'] * tracking_reward
            reward_breakdown['tracking_performance'] = tracking_reward
            
            # 2. 资源效率奖励
            efficiency_reward = self._calculate_resource_efficiency_reward(state, action, base_data)
            total_reward += self.reward_weights['resource_efficiency'] * efficiency_reward
            reward_breakdown['resource_efficiency'] = efficiency_reward
            
            # 3. 任务完成奖励
            completion_reward = self._calculate_mission_completion_reward(state, action, base_data)
            total_reward += self.reward_weights['mission_completion'] * completion_reward
            reward_breakdown['mission_completion'] = completion_reward
            
            # 4. 惩罚项
            penalty = self._calculate_penalty_terms(state, action, base_data)
            total_reward -= penalty
            reward_breakdown['penalty'] = penalty
            
            # 记录奖励分解
            logger.debug(f"奖励分解: {reward_breakdown}, 总奖励: {total_reward:.3f}")
            
            return total_reward
            
        except Exception as e:
            logger.error(f"奖励计算失败: {e}")
            return 0.0
    
    def _calculate_tracking_performance_reward(self, state: Dict[str, Any], 
                                             action: Dict[str, Any], 
                                             base_data: Dict[str, Any]) -> float:
        """计算跟踪性能奖励"""
        try:
            tracking_reward = 0.0
            
            # 覆盖时间奖励
            if self.reward_config.get('tracking_performance', {}).get('coverage_time', False):
                coverage_reward = self._calculate_coverage_reward(state, base_data)
                tracking_reward += 0.4 * coverage_reward
                
            # 跟踪精度奖励
            if self.reward_config.get('tracking_performance', {}).get('tracking_accuracy', False):
                accuracy_reward = self._calculate_accuracy_reward(state, action, base_data)
                tracking_reward += 0.3 * accuracy_reward
                
            # 目标检测奖励
            if self.reward_config.get('tracking_performance', {}).get('target_detection', False):
                detection_reward = self._calculate_detection_reward(state, base_data)
                tracking_reward += 0.3 * detection_reward
                
            return np.clip(tracking_reward, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"跟踪性能奖励计算失败: {e}")
            return 0.0
    
    def _calculate_resource_efficiency_reward(self, state: Dict[str, Any], 
                                            action: Dict[str, Any], 
                                            base_data: Dict[str, Any]) -> float:
        """计算资源效率奖励"""
        try:
            efficiency_reward = 0.0
            
            # 功率消耗效率
            if self.reward_config.get('resource_efficiency', {}).get('power_consumption', False):
                power_reward = self._calculate_power_efficiency_reward(state, action, base_data)
                efficiency_reward += 0.4 * power_reward
                
            # 通信带宽效率
            if self.reward_config.get('resource_efficiency', {}).get('communication_bandwidth', False):
                comm_reward = self._calculate_communication_efficiency_reward(state, action)
                efficiency_reward += 0.3 * comm_reward
                
            # 计算负载效率
            if self.reward_config.get('resource_efficiency', {}).get('computational_load', False):
                comp_reward = self._calculate_computational_efficiency_reward(state, action)
                efficiency_reward += 0.3 * comp_reward
                
            return np.clip(efficiency_reward, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"资源效率奖励计算失败: {e}")
            return 0.0
    
    def _calculate_mission_completion_reward(self, state: Dict[str, Any], 
                                           action: Dict[str, Any], 
                                           base_data: Dict[str, Any]) -> float:
        """计算任务完成奖励"""
        try:
            completion_reward = 0.0
            
            # 威胁中和奖励
            if self.reward_config.get('mission_completion', {}).get('threat_neutralization', False):
                neutralization_reward = self._calculate_threat_neutralization_reward(state, base_data)
                completion_reward += 0.5 * neutralization_reward
                
            # 响应时间奖励
            if self.reward_config.get('mission_completion', {}).get('response_time', False):
                response_reward = self._calculate_response_time_reward(state, action, base_data)
                completion_reward += 0.3 * response_reward
                
            # 覆盖完整性奖励
            if self.reward_config.get('mission_completion', {}).get('coverage_completeness', False):
                completeness_reward = self._calculate_coverage_completeness_reward(state, base_data)
                completion_reward += 0.2 * completeness_reward
                
            return np.clip(completion_reward, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"任务完成奖励计算失败: {e}")
            return 0.0
    
    def _calculate_coverage_reward(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> float:
        """计算覆盖奖励"""
        try:
            # 从状态中获取覆盖率
            coverage_ratio = state.get('coverage_ratio', 0.0)
            
            # 时间权重：任务进行时间越长，覆盖要求越高
            mission_progress = state.get('mission_progress', 0.0)
            time_weight = 0.5 + 0.5 * mission_progress
            
            # 覆盖奖励 = 覆盖率 * 时间权重
            coverage_reward = coverage_ratio * time_weight
            
            return coverage_reward
            
        except Exception as e:
            logger.error(f"覆盖奖励计算失败: {e}")
            return 0.0
    
    def _calculate_accuracy_reward(self, state: Dict[str, Any], 
                                 action: Dict[str, Any], 
                                 base_data: Dict[str, Any]) -> float:
        """计算跟踪精度奖励"""
        try:
            # 基于动作的指向精度评估
            satellite_actions = action.get('satellite_actions', {})
            
            if not satellite_actions:
                return 0.0
            
            accuracy_scores = []
            
            for sat_id, sat_action in satellite_actions.items():
                # 评估载荷指向精度
                payload_pointing = sat_action.get('payload_pointing', {})
                pointing_mode = payload_pointing.get('pointing_mode', 'fixed')
                
                if pointing_mode == 'tracking':
                    # 跟踪模式给予更高分数
                    accuracy_scores.append(0.9)
                elif pointing_mode == 'scanning':
                    # 扫描模式中等分数
                    accuracy_scores.append(0.6)
                else:
                    # 固定模式较低分数
                    accuracy_scores.append(0.3)
            
            # 计算平均精度分数
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            return avg_accuracy
            
        except Exception as e:
            logger.error(f"精度奖励计算失败: {e}")
            return 0.0
    
    def _calculate_detection_reward(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> float:
        """计算目标检测奖励"""
        try:
            active_missiles = state.get('active_missiles', 0)
            
            if active_missiles == 0:
                return 1.0  # 没有威胁时给满分
            
            # 计算被检测到的导弹比例
            visibility_data = base_data.get('visibility', [])
            detected_missiles = set()
            
            for vis in visibility_data:
                if vis.get('has_visibility', False):
                    missile_id = vis.get('missile_id', '')
                    if missile_id:
                        detected_missiles.add(missile_id)
            
            detection_ratio = len(detected_missiles) / active_missiles
            
            return detection_ratio
            
        except Exception as e:
            logger.error(f"检测奖励计算失败: {e}")
            return 0.0
    
    def _calculate_power_efficiency_reward(self, state: Dict[str, Any], 
                                         action: Dict[str, Any], 
                                         base_data: Dict[str, Any]) -> float:
        """计算功率效率奖励"""
        try:
            satellite_actions = action.get('satellite_actions', {})
            
            if not satellite_actions:
                return 1.0  # 没有动作时认为效率最高
            
            efficiency_scores = []
            
            for sat_id, sat_action in satellite_actions.items():
                power_mgmt = sat_action.get('power_management', {})
                power_allocation = power_mgmt.get('power_allocation', {})
                
                # 计算总功率分配
                total_allocation = sum(power_allocation.values())
                
                if total_allocation <= 1.0:
                    # 功率分配合理
                    efficiency_score = 1.0 - total_allocation * 0.5  # 使用越少效率越高
                else:
                    # 功率分配超限，给予惩罚
                    efficiency_score = max(0.0, 1.0 - (total_allocation - 1.0))
                
                efficiency_scores.append(efficiency_score)
            
            avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 1.0
            
            return avg_efficiency
            
        except Exception as e:
            logger.error(f"功率效率奖励计算失败: {e}")
            return 0.0
    
    def _calculate_communication_efficiency_reward(self, state: Dict[str, Any], 
                                                 action: Dict[str, Any]) -> float:
        """计算通信效率奖励"""
        try:
            # 简化的通信效率计算
            # 基于动作复杂度评估通信需求
            
            satellite_actions = action.get('satellite_actions', {})
            mission_actions = action.get('mission_actions', {})
            
            # 计算通信复杂度
            comm_complexity = 0.0
            
            # 卫星动作的通信需求
            comm_complexity += len(satellite_actions) * 0.1
            
            # 任务动作的通信需求
            target_assignments = mission_actions.get('target_assignments', [])
            comm_complexity += len(target_assignments) * 0.2
            
            # 协调指令的通信需求
            coordination = mission_actions.get('coordination_commands', {})
            comm_complexity += len(coordination) * 0.3
            
            # 效率 = 1 - 标准化的复杂度
            max_complexity = 2.0  # 假设的最大复杂度
            efficiency = max(0.0, 1.0 - comm_complexity / max_complexity)
            
            return efficiency
            
        except Exception as e:
            logger.error(f"通信效率奖励计算失败: {e}")
            return 0.0
    
    def _calculate_computational_efficiency_reward(self, state: Dict[str, Any], 
                                                 action: Dict[str, Any]) -> float:
        """计算计算效率奖励"""
        try:
            # 基于动作复杂度评估计算负载
            
            satellite_actions = action.get('satellite_actions', {})
            mission_actions = action.get('mission_actions', {})
            
            # 计算计算复杂度
            comp_complexity = 0.0
            
            # 姿态控制的计算需求
            for sat_action in satellite_actions.values():
                if 'attitude_control' in sat_action:
                    comp_complexity += 0.3
                if 'payload_pointing' in sat_action:
                    comp_complexity += 0.2
            
            # 任务规划的计算需求
            target_assignments = mission_actions.get('target_assignments', [])
            comp_complexity += len(target_assignments) * 0.1
            
            resource_allocation = mission_actions.get('resource_allocation', {})
            comp_complexity += len(resource_allocation) * 0.2
            
            # 效率 = 1 - 标准化的复杂度
            max_complexity = 3.0  # 假设的最大复杂度
            efficiency = max(0.0, 1.0 - comp_complexity / max_complexity)
            
            return efficiency
            
        except Exception as e:
            logger.error(f"计算效率奖励计算失败: {e}")
            return 0.0
    
    def _calculate_threat_neutralization_reward(self, state: Dict[str, Any], 
                                              base_data: Dict[str, Any]) -> float:
        """计算威胁中和奖励"""
        try:
            active_missiles = state.get('active_missiles', 0)
            
            if active_missiles == 0:
                return 1.0  # 没有威胁时给满分
            
            # 计算被跟踪的威胁比例
            visibility_data = base_data.get('visibility', [])
            tracked_missiles = set()
            
            for vis in visibility_data:
                if vis.get('has_visibility', False):
                    missile_id = vis.get('missile_id', '')
                    if missile_id:
                        tracked_missiles.add(missile_id)
            
            neutralization_ratio = len(tracked_missiles) / active_missiles
            
            # 考虑威胁等级权重
            missile_threat_levels = state.get('missile_threat_levels', [])
            if missile_threat_levels:
                # 高威胁等级的导弹被跟踪给予更高奖励
                weighted_score = 0.0
                total_weight = 0.0
                
                missiles = base_data.get('missiles', [])
                for i, missile in enumerate(missiles):
                    threat_level = missile_threat_levels[i] if i < len(missile_threat_levels) else 1
                    weight = threat_level  # 威胁等级作为权重
                    
                    missile_id = missile.get('missile_id', '')
                    is_tracked = missile_id in tracked_missiles
                    
                    weighted_score += weight * (1.0 if is_tracked else 0.0)
                    total_weight += weight
                
                if total_weight > 0:
                    neutralization_ratio = weighted_score / total_weight
            
            return neutralization_ratio
            
        except Exception as e:
            logger.error(f"威胁中和奖励计算失败: {e}")
            return 0.0
    
    def _calculate_response_time_reward(self, state: Dict[str, Any], 
                                      action: Dict[str, Any], 
                                      base_data: Dict[str, Any]) -> float:
        """计算响应时间奖励"""
        try:
            # 简化的响应时间评估
            # 基于动作的及时性评估
            
            mission_actions = action.get('mission_actions', {})
            target_assignments = mission_actions.get('target_assignments', [])
            
            if not target_assignments:
                return 0.5  # 没有分配任务时给中等分数
            
            # 评估分配的及时性
            # 这里简化为：分配的任务越多，响应越及时
            active_missiles = state.get('active_missiles', 0)
            
            if active_missiles == 0:
                return 1.0
            
            assignment_ratio = len(target_assignments) / active_missiles
            response_reward = min(1.0, assignment_ratio)
            
            return response_reward
            
        except Exception as e:
            logger.error(f"响应时间奖励计算失败: {e}")
            return 0.0
    
    def _calculate_coverage_completeness_reward(self, state: Dict[str, Any], 
                                              base_data: Dict[str, Any]) -> float:
        """计算覆盖完整性奖励"""
        try:
            # 评估覆盖的完整性和均匀性
            
            visibility_matrix = state.get('visibility_matrix', [])
            if not visibility_matrix:
                return 0.0
            
            # 计算覆盖均匀性
            satellite_coverage = state.get('satellite_coverage_counts', [])
            missile_coverage = state.get('missile_coverage_counts', [])
            
            completeness_score = 0.0
            
            # 卫星覆盖均匀性
            if satellite_coverage:
                sat_coverage_std = np.std(satellite_coverage)
                sat_coverage_mean = np.mean(satellite_coverage)
                if sat_coverage_mean > 0:
                    sat_uniformity = 1.0 - min(1.0, sat_coverage_std / sat_coverage_mean)
                    completeness_score += 0.5 * sat_uniformity
            
            # 导弹覆盖完整性
            if missile_coverage:
                uncovered_missiles = sum(1 for count in missile_coverage if count == 0)
                coverage_completeness = 1.0 - uncovered_missiles / len(missile_coverage)
                completeness_score += 0.5 * coverage_completeness
            
            return completeness_score
            
        except Exception as e:
            logger.error(f"覆盖完整性奖励计算失败: {e}")
            return 0.0
    
    def _calculate_penalty_terms(self, state: Dict[str, Any], 
                               action: Dict[str, Any], 
                               base_data: Dict[str, Any]) -> float:
        """计算惩罚项"""
        try:
            total_penalty = 0.0
            
            # 虚警惩罚
            false_alarm_penalty = self._calculate_false_alarm_penalty(state, base_data)
            total_penalty += false_alarm_penalty
            
            # 资源浪费惩罚
            resource_waste_penalty = self._calculate_resource_waste_penalty(state, action)
            total_penalty += resource_waste_penalty
            
            # 任务失败惩罚
            mission_failure_penalty = self._calculate_mission_failure_penalty(state, base_data)
            total_penalty += mission_failure_penalty
            
            return total_penalty
            
        except Exception as e:
            logger.error(f"惩罚项计算失败: {e}")
            return 0.0
    
    def _calculate_false_alarm_penalty(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> float:
        """计算虚警惩罚"""
        # 简化版本：假设没有虚警数据，返回0
        return 0.0
    
    def _calculate_resource_waste_penalty(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """计算资源浪费惩罚"""
        try:
            penalty = 0.0
            
            satellite_actions = action.get('satellite_actions', {})
            
            for sat_action in satellite_actions.values():
                power_mgmt = sat_action.get('power_management', {})
                power_allocation = power_mgmt.get('power_allocation', {})
                
                total_allocation = sum(power_allocation.values())
                
                # 功率分配超限惩罚
                if total_allocation > 1.0:
                    penalty += (total_allocation - 1.0) * 0.5
            
            return penalty
            
        except Exception as e:
            logger.error(f"资源浪费惩罚计算失败: {e}")
            return 0.0
    
    def _calculate_mission_failure_penalty(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> float:
        """计算任务失败惩罚"""
        try:
            penalty = 0.0
            
            # 覆盖空隙惩罚
            coverage_gap_ratio = state.get('coverage_gap_ratio', 0.0)
            penalty += coverage_gap_ratio * 0.3
            
            # 无响应惩罚
            active_missiles = state.get('active_missiles', 0)
            active_assignments = state.get('active_tracking_assignments', 0)
            
            if active_missiles > 0 and active_assignments == 0:
                penalty += 0.5  # 有威胁但无响应的严重惩罚
            
            return penalty
            
        except Exception as e:
            logger.error(f"任务失败惩罚计算失败: {e}")
            return 0.0
    
    def get_reward_breakdown(self, state: Dict[str, Any], action: Dict[str, Any], 
                           base_data: Dict[str, Any]) -> Dict[str, float]:
        """获取奖励分解详情"""
        try:
            breakdown = {}
            
            # 计算各组件奖励
            tracking_reward = self._calculate_tracking_performance_reward(state, action, base_data)
            efficiency_reward = self._calculate_resource_efficiency_reward(state, action, base_data)
            completion_reward = self._calculate_mission_completion_reward(state, action, base_data)
            penalty = self._calculate_penalty_terms(state, action, base_data)
            
            # 加权后的奖励
            breakdown['tracking_performance'] = self.reward_weights['tracking_performance'] * tracking_reward
            breakdown['resource_efficiency'] = self.reward_weights['resource_efficiency'] * efficiency_reward
            breakdown['mission_completion'] = self.reward_weights['mission_completion'] * completion_reward
            breakdown['penalty'] = penalty
            
            # 总奖励
            breakdown['total_reward'] = (breakdown['tracking_performance'] + 
                                       breakdown['resource_efficiency'] + 
                                       breakdown['mission_completion'] - 
                                       breakdown['penalty'])
            
            return breakdown
            
        except Exception as e:
            logger.error(f"奖励分解计算失败: {e}")
            return {'total_reward': 0.0}
