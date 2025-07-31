"""
RLHF强化学习数据采集器
专门用于收集强化学习训练所需的状态-动作-奖励数据
"""

import json
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import h5py

from .reward_calculator import RLHFRewardCalculator
from .action_executor import RLHFActionExecutor
from .data_quality_validator import RLHFDataQualityValidator

logger = logging.getLogger(__name__)

@dataclass
class RLHFDataPoint:
    """强化学习数据点"""
    timestamp: datetime
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Dict[str, Any]
    done: bool
    info: Dict[str, Any]

@dataclass
class Episode:
    """强化学习回合"""
    episode_id: str
    scenario_type: str
    start_time: datetime
    end_time: datetime
    data_points: List[RLHFDataPoint]
    total_reward: float
    success: bool
    metadata: Dict[str, Any]

class RLHFDataCollector:
    """RLHF强化学习数据采集器"""
    
    def __init__(self, base_data_collector, config_manager, time_manager, stk_manager=None):
        """
        初始化RLHF数据采集器

        Args:
            base_data_collector: 基础数据采集器
            config_manager: 配置管理器
            time_manager: 时间管理器
            stk_manager: STK管理器（可选）
        """
        self.base_collector = base_data_collector
        self.config_manager = config_manager
        self.time_manager = time_manager
        self.stk_manager = stk_manager

        # RLHF配置
        self.rlhf_config = config_manager.config.get('rlhf_data_collection', {})

        # 初始化核心组件
        self.reward_calculator = RLHFRewardCalculator(config_manager)
        self.data_quality_validator = RLHFDataQualityValidator(config_manager)

        # 如果有STK管理器，初始化动作执行器
        if stk_manager:
            self.action_executor = RLHFActionExecutor(stk_manager, config_manager)
        else:
            self.action_executor = None
            logger.warning("⚠️ 未提供STK管理器，动作执行功能将不可用")

        # 数据存储
        self.current_episode = None
        self.episodes = []
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.validation_history = []

        # 输出目录
        self.output_dir = Path("output/rlhf_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 状态空间定义
        self.state_space_config = self.rlhf_config.get('state_space', {})
        self.action_space_config = self.rlhf_config.get('action_space', {})
        self.reward_config = self.rlhf_config.get('reward_components', {})

        # 数据采集统计
        self.collection_stats = {
            'total_data_points': 0,
            'valid_data_points': 0,
            'invalid_data_points': 0,
            'average_reward': 0.0,
            'average_quality_score': 0.0
        }

        logger.info("🤖 RLHF数据采集器初始化完成")
    
    def start_episode(self, scenario_type: str, scenario_params: Dict[str, Any]) -> str:
        """
        开始新的强化学习回合
        
        Args:
            scenario_type: 场景类型
            scenario_params: 场景参数
            
        Returns:
            回合ID
        """
        episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scenario_type}"
        
        self.current_episode = Episode(
            episode_id=episode_id,
            scenario_type=scenario_type,
            start_time=self.time_manager.current_simulation_time,
            end_time=None,
            data_points=[],
            total_reward=0.0,
            success=False,
            metadata={
                'scenario_params': scenario_params,
                'constellation_info': self.base_collector.constellation_manager.get_constellation_info()
            }
        )
        
        logger.info(f"🎬 开始新回合: {episode_id}, 场景类型: {scenario_type}")
        return episode_id
    
    def collect_rlhf_data_point(self, action: Dict[str, Any], execute_action: bool = False) -> RLHFDataPoint:
        """
        采集单个RLHF数据点 - 增强版本

        Args:
            action: 执行的动作
            execute_action: 是否实际执行动作

        Returns:
            RLHF数据点
        """
        try:
            # 获取当前时间
            current_time = self.time_manager.current_simulation_time

            # 采集基础数据
            base_data = self.base_collector.collect_data_at_time(current_time)
            if not base_data:
                logger.error("基础数据采集失败")
                return None

            # 构建状态向量
            state = self._extract_state_vector(base_data)

            # 执行动作（如果需要）
            action_result = None
            if execute_action and self.action_executor:
                action_result = self.action_executor.execute_action(action)
                logger.info(f"动作执行结果: {action_result.get('success', False)}")

            # 计算奖励
            reward = self.reward_calculator.calculate_total_reward(state, action, base_data)

            # 预测下一状态（简化版本）
            next_state = self._predict_next_state(state, action)

            # 判断回合是否结束
            done = self._is_episode_done(state, base_data)

            # 创建数据点
            data_point = RLHFDataPoint(
                timestamp=current_time,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info={
                    'base_data': base_data,
                    'simulation_progress': self.time_manager.get_simulation_progress(),
                    'action_result': action_result
                }
            )

            # 数据质量验证
            validation_result = self.data_quality_validator.validate_rlhf_data_point(data_point)
            data_point.info['validation_result'] = validation_result

            # 更新统计
            self.collection_stats['total_data_points'] += 1
            if validation_result['is_valid']:
                self.collection_stats['valid_data_points'] += 1
            else:
                self.collection_stats['invalid_data_points'] += 1
                logger.warning(f"数据质量验证失败: {validation_result['errors']}")

            # 添加到当前回合
            if self.current_episode:
                self.current_episode.data_points.append(data_point)
                self.current_episode.total_reward += reward

            # 更新历史记录
            self.state_history.append(state)
            self.action_history.append(action)
            self.reward_history.append(reward)
            self.validation_history.append(validation_result)

            # 更新平均统计
            self._update_average_statistics()

            logger.info(f"📊 RLHF数据点采集完成: 奖励={reward:.3f}, 质量分数={validation_result['validation_score']:.3f}, 完成={done}")

            return data_point

        except Exception as e:
            logger.error(f"❌ RLHF数据点采集失败: {e}")
            self.collection_stats['invalid_data_points'] += 1
            return None
    
    def _extract_state_vector(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        从基础数据中提取状态向量 - 基于现有STK数据结构优化

        Args:
            base_data: 基础采集数据

        Returns:
            状态向量
        """
        state = {}

        try:
            # 获取卫星数据
            satellites = base_data.get('satellites', [])
            missiles = base_data.get('missiles', [])
            visibility_data = base_data.get('visibility', [])

            # 卫星状态提取
            state.update(self._extract_satellite_states(satellites))

            # 导弹状态提取
            state.update(self._extract_missile_states(missiles))

            # 可见性状态提取
            state.update(self._extract_visibility_states(visibility_data, len(satellites), len(missiles)))

            # 环境状态提取
            state.update(self._extract_environment_states(base_data))

            # 任务状态提取
            state.update(self._extract_mission_states(base_data, satellites, missiles))

            logger.debug(f"状态向量提取完成，包含 {len(state)} 个状态特征")

        except Exception as e:
            logger.error(f"状态向量提取失败: {e}")
            # 返回空状态以避免系统崩溃
            state = self._get_empty_state()

        return state

    def _extract_satellite_states(self, satellites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取卫星状态"""
        satellite_state = {}

        if not satellites:
            return self._get_empty_satellite_state()

        # 位置状态
        if self.state_space_config.get('satellite_states', {}).get('position', False):
            positions = []
            for sat in satellites:
                pos = sat.get('position', {})
                if isinstance(pos, dict):
                    positions.append([
                        pos.get('x', 0.0),
                        pos.get('y', 0.0),
                        pos.get('z', 0.0)
                    ])
                else:
                    positions.append([0.0, 0.0, 0.0])
            satellite_state['satellite_positions'] = positions

        # 速度状态
        if self.state_space_config.get('satellite_states', {}).get('velocity', False):
            velocities = []
            for sat in satellites:
                vel = sat.get('velocity', {})
                if isinstance(vel, dict):
                    velocities.append([
                        vel.get('vx', 0.0),
                        vel.get('vy', 0.0),
                        vel.get('vz', 0.0)
                    ])
                else:
                    velocities.append([0.0, 0.0, 0.0])
            satellite_state['satellite_velocities'] = velocities

        # 姿态状态
        if self.state_space_config.get('satellite_states', {}).get('attitude', False):
            attitudes = []
            for sat in satellites:
                att = sat.get('attitude', {})
                if isinstance(att, dict):
                    attitudes.append([
                        att.get('q0', 1.0),
                        att.get('q1', 0.0),
                        att.get('q2', 0.0),
                        att.get('q3', 0.0)
                    ])
                else:
                    attitudes.append([1.0, 0.0, 0.0, 0.0])
            satellite_state['satellite_attitudes'] = attitudes

        # 轨道参数状态
        if self.state_space_config.get('satellite_states', {}).get('orbital_elements', False):
            orbital_elements = []
            for sat in satellites:
                orb = sat.get('orbital_elements', {})
                if isinstance(orb, dict):
                    orbital_elements.append([
                        orb.get('semi_major_axis', 0.0),
                        orb.get('eccentricity', 0.0),
                        orb.get('inclination', 0.0),
                        orb.get('raan', 0.0),
                        orb.get('arg_perigee', 0.0),
                        orb.get('mean_anomaly', 0.0)
                    ])
                else:
                    orbital_elements.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            satellite_state['satellite_orbital_elements'] = orbital_elements

        # 功率状态
        if self.state_space_config.get('satellite_states', {}).get('power_status', False):
            power_states = []
            for sat in satellites:
                payload = sat.get('payload_status', {})
                power_states.append([
                    payload.get('power_consumption', 80.0),  # 默认功耗
                    1.0 if payload.get('operational', True) else 0.0,  # 运行状态
                    payload.get('temperature', 25.0)  # 温度
                ])
            satellite_state['satellite_power_states'] = power_states

        # 载荷状态
        if self.state_space_config.get('satellite_states', {}).get('payload_status', False):
            payload_states = []
            for sat in satellites:
                payload = sat.get('payload_status', {})
                payload_states.append([
                    1.0 if payload.get('operational', True) else 0.0,
                    payload.get('pointing_accuracy', 0.1),  # 指向精度
                    payload.get('detection_range', 5000.0)  # 探测距离
                ])
            satellite_state['satellite_payload_states'] = payload_states

        return satellite_state

    def _extract_missile_states(self, missiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取导弹状态"""
        missile_state = {}

        if not missiles:
            return self._get_empty_missile_state()

        # 导弹位置状态
        if self.state_space_config.get('missile_states', {}).get('position', False):
            positions = []
            for missile in missiles:
                pos = missile.get('position', {})
                if isinstance(pos, dict):
                    positions.append([
                        pos.get('x', 0.0),
                        pos.get('y', 0.0),
                        pos.get('z', 0.0)
                    ])
                else:
                    positions.append([0.0, 0.0, 0.0])
            missile_state['missile_positions'] = positions

        # 导弹速度状态
        if self.state_space_config.get('missile_states', {}).get('velocity', False):
            velocities = []
            for missile in missiles:
                vel = missile.get('velocity', {})
                if isinstance(vel, dict):
                    velocities.append([
                        vel.get('vx', 0.0),
                        vel.get('vy', 0.0),
                        vel.get('vz', 0.0)
                    ])
                else:
                    velocities.append([0.0, 0.0, 0.0])
            missile_state['missile_velocities'] = velocities

        # 轨迹预测状态
        if self.state_space_config.get('missile_states', {}).get('trajectory_prediction', False):
            trajectory_predictions = []
            for missile in missiles:
                traj = missile.get('trajectory', {})
                if isinstance(traj, dict):
                    # 提取关键轨迹信息
                    launch_pos = traj.get('launch_position', {})
                    impact_pos = traj.get('impact_position', {})
                    trajectory_predictions.append([
                        launch_pos.get('lat', 0.0),
                        launch_pos.get('lon', 0.0),
                        impact_pos.get('lat', 0.0),
                        impact_pos.get('lon', 0.0),
                        traj.get('flight_time', 0.0),
                        traj.get('max_altitude', 0.0)
                    ])
                else:
                    trajectory_predictions.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            missile_state['missile_trajectory_predictions'] = trajectory_predictions

        # 威胁等级状态
        if self.state_space_config.get('missile_states', {}).get('threat_level', False):
            threat_levels = []
            for missile in missiles:
                threat_level = missile.get('threat_level', 'unknown')
                threat_levels.append(self._encode_threat_level(threat_level))
            missile_state['missile_threat_levels'] = threat_levels

        # 飞行阶段状态
        if self.state_space_config.get('missile_states', {}).get('flight_phase', False):
            flight_phases = []
            for missile in missiles:
                phase = missile.get('flight_status', {}).get('status', 'unknown')
                flight_phases.append(self._encode_flight_phase(phase))
            missile_state['missile_flight_phases'] = flight_phases

        # 剩余飞行时间状态
        if self.state_space_config.get('missile_states', {}).get('remaining_time', False):
            remaining_times = []
            for missile in missiles:
                flight_status = missile.get('flight_status', {})
                flight_duration = flight_status.get('flight_duration', 0.0)
                total_flight_time = missile.get('trajectory', {}).get('flight_time', 1800.0)
                remaining_time = max(0.0, total_flight_time - flight_duration)
                remaining_times.append(remaining_time)
            missile_state['missile_remaining_times'] = remaining_times

        return missile_state

    def _extract_visibility_states(self, visibility_data: List[Dict[str, Any]],
                                 num_satellites: int, num_missiles: int) -> Dict[str, Any]:
        """提取可见性状态"""
        visibility_state = {}

        # 创建可见性矩阵
        visibility_matrix = self._create_visibility_matrix(visibility_data, num_satellites, num_missiles)
        visibility_state['visibility_matrix'] = visibility_matrix

        # 计算覆盖统计
        if visibility_matrix:
            total_pairs = num_satellites * num_missiles
            covered_pairs = sum(sum(row) for row in visibility_matrix)
            coverage_ratio = covered_pairs / total_pairs if total_pairs > 0 else 0.0
            visibility_state['coverage_ratio'] = coverage_ratio

            # 每个卫星的覆盖数
            satellite_coverage = [sum(row) for row in visibility_matrix]
            visibility_state['satellite_coverage_counts'] = satellite_coverage

            # 每个导弹被覆盖的卫星数
            if visibility_matrix and len(visibility_matrix[0]) > 0:
                missile_coverage = [sum(visibility_matrix[i][j] for i in range(len(visibility_matrix)))
                                  for j in range(len(visibility_matrix[0]))]
                visibility_state['missile_coverage_counts'] = missile_coverage
            else:
                visibility_state['missile_coverage_counts'] = []
        else:
            visibility_state['coverage_ratio'] = 0.0
            visibility_state['satellite_coverage_counts'] = []
            visibility_state['missile_coverage_counts'] = []

        return visibility_state

    def _extract_environment_states(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取环境状态"""
        environment_state = {}

        # 时间信息
        if self.state_space_config.get('environment_states', {}).get('time_of_day', False):
            collection_time = base_data.get('collection_time', '')
            environment_state['time_of_day'] = self._encode_time_of_day(collection_time)

        # 仿真进度
        environment_state['simulation_progress'] = base_data.get('simulation_progress', 0.0)

        # 太阳位置（简化计算）
        if self.state_space_config.get('environment_states', {}).get('sun_position', False):
            # 这里可以添加更复杂的太阳位置计算
            # 简化版本：基于时间的周期性变化
            time_of_day = environment_state.get('time_of_day', 0.0)
            sun_elevation = np.sin(2 * np.pi * time_of_day) * 90.0  # 简化的太阳高度角
            environment_state['sun_elevation'] = sun_elevation

        # 地影状态（简化）
        if self.state_space_config.get('environment_states', {}).get('earth_shadow', False):
            sun_elevation = environment_state.get('sun_elevation', 0.0)
            in_shadow = 1.0 if sun_elevation < 0 else 0.0
            environment_state['earth_shadow'] = in_shadow

        return environment_state

    def _extract_mission_states(self, base_data: Dict[str, Any],
                               satellites: List[Dict[str, Any]],
                               missiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """提取任务状态"""
        mission_state = {}

        # 基础任务统计
        mission_state['active_satellites'] = len(satellites)
        mission_state['active_missiles'] = len(missiles)
        mission_state['mission_progress'] = base_data.get('simulation_progress', 0.0)

        # 跟踪分配状态
        if self.state_space_config.get('mission_states', {}).get('tracking_assignments', False):
            visibility_data = base_data.get('visibility', [])
            active_assignments = len([v for v in visibility_data if v.get('has_visibility', False)])
            mission_state['active_tracking_assignments'] = active_assignments

        # 资源利用率
        if self.state_space_config.get('mission_states', {}).get('resource_utilization', False):
            total_satellites = len(satellites)
            if total_satellites > 0:
                # 计算平均功率利用率
                total_power = sum(sat.get('payload_status', {}).get('power_consumption', 80.0)
                                for sat in satellites)
                avg_power_utilization = total_power / (total_satellites * 100.0)  # 假设最大功率100W
                mission_state['power_utilization'] = min(1.0, avg_power_utilization)
            else:
                mission_state['power_utilization'] = 0.0

        # 覆盖空隙
        if self.state_space_config.get('mission_states', {}).get('coverage_gaps', False):
            uncovered_missiles = len([m for m in missiles
                                    if not any(v.get('missile_id') == m.get('missile_id') and v.get('has_visibility')
                                             for v in base_data.get('visibility', []))])
            coverage_gap_ratio = uncovered_missiles / len(missiles) if missiles else 0.0
            mission_state['coverage_gap_ratio'] = coverage_gap_ratio

        return mission_state

    def _get_empty_state(self) -> Dict[str, Any]:
        """获取空状态向量"""
        return {
            'satellite_positions': [],
            'satellite_velocities': [],
            'satellite_attitudes': [],
            'missile_positions': [],
            'missile_velocities': [],
            'visibility_matrix': [],
            'coverage_ratio': 0.0,
            'mission_progress': 0.0,
            'active_satellites': 0,
            'active_missiles': 0,
            'time_of_day': 0.0
        }

    def _get_empty_satellite_state(self) -> Dict[str, Any]:
        """获取空卫星状态"""
        return {
            'satellite_positions': [],
            'satellite_velocities': [],
            'satellite_attitudes': [],
            'satellite_orbital_elements': [],
            'satellite_power_states': [],
            'satellite_payload_states': []
        }

    def _get_empty_missile_state(self) -> Dict[str, Any]:
        """获取空导弹状态"""
        return {
            'missile_positions': [],
            'missile_velocities': [],
            'missile_trajectory_predictions': [],
            'missile_threat_levels': [],
            'missile_flight_phases': [],
            'missile_remaining_times': []
        }

    def _encode_flight_phase(self, phase: str) -> int:
        """编码飞行阶段"""
        phase_mapping = {
            'boost': 1,
            'midcourse': 2,
            'terminal': 3,
            'in_flight': 2,  # 默认为中段
            'impact': 4,
            'unknown': 0
        }
        return phase_mapping.get(phase.lower(), 0)

    def _create_visibility_matrix(self, visibility_data: List[Dict[str, Any]],
                                 num_satellites: int, num_missiles: int) -> List[List[int]]:
        """
        创建可见性矩阵 - 优化版本

        Args:
            visibility_data: 可见性数据
            num_satellites: 卫星数量
            num_missiles: 导弹数量

        Returns:
            可见性矩阵 [satellite_index][missile_index]
        """
        if num_satellites == 0 or num_missiles == 0:
            return []

        # 初始化矩阵
        matrix = [[0 for _ in range(num_missiles)] for _ in range(num_satellites)]

        # 创建ID到索引的映射
        satellite_ids = set()
        missile_ids = set()

        for vis in visibility_data:
            sat_id = vis.get('satellite_id', '')
            missile_id = vis.get('missile_id', '')
            if sat_id:
                satellite_ids.add(sat_id)
            if missile_id:
                missile_ids.add(missile_id)

        satellite_id_to_index = {sat_id: i for i, sat_id in enumerate(sorted(satellite_ids))}
        missile_id_to_index = {missile_id: i for i, missile_id in enumerate(sorted(missile_ids))}

        # 填充矩阵
        for vis in visibility_data:
            sat_id = vis.get('satellite_id', '')
            missile_id = vis.get('missile_id', '')
            has_visibility = vis.get('has_visibility', False)

            if sat_id in satellite_id_to_index and missile_id in missile_id_to_index:
                sat_idx = satellite_id_to_index[sat_id]
                missile_idx = missile_id_to_index[missile_id]

                if sat_idx < num_satellites and missile_idx < num_missiles:
                    matrix[sat_idx][missile_idx] = 1 if has_visibility else 0

        return matrix
    
    def _update_average_statistics(self):
        """更新平均统计信息"""
        try:
            if self.reward_history:
                self.collection_stats['average_reward'] = np.mean(self.reward_history)

            if self.validation_history:
                quality_scores = [v['validation_score'] for v in self.validation_history
                                if 'validation_score' in v]
                if quality_scores:
                    self.collection_stats['average_quality_score'] = np.mean(quality_scores)

        except Exception as e:
            logger.error(f"统计信息更新失败: {e}")
    
    def get_reward_breakdown(self, state: Dict[str, Any], action: Dict[str, Any],
                           base_data: Dict[str, Any]) -> Dict[str, float]:
        """获取奖励分解详情"""
        return self.reward_calculator.get_reward_breakdown(state, action, base_data)
    
    def _predict_next_state(self, current_state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """预测下一状态（简化版本）"""
        # 这里应该基于动力学模型预测下一状态
        # 简化版本：假设状态基本不变
        next_state = current_state.copy()
        
        # 可以添加一些基于动作的状态变化预测
        if 'mission_progress' in next_state:
            next_state['mission_progress'] += 0.01  # 假设进度增加
            
        return next_state
    
    def _is_episode_done(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> bool:
        """判断回合是否结束"""
        # 检查仿真是否结束
        if self.time_manager.is_simulation_finished():
            return True
            
        # 检查是否达到目标采集次数
        if self.time_manager.is_collection_finished():
            return True
            
        # 检查是否所有威胁都被处理
        missiles = base_data.get('missiles', [])
        if not missiles:
            return True
            
        return False
    
    def end_episode(self, success: bool = False) -> Episode:
        """
        结束当前回合
        
        Args:
            success: 是否成功完成任务
            
        Returns:
            完成的回合
        """
        if not self.current_episode:
            logger.warning("⚠️ 没有活跃的回合可以结束")
            return None
            
        self.current_episode.end_time = self.time_manager.current_simulation_time
        self.current_episode.success = success
        
        # 添加到回合列表
        self.episodes.append(self.current_episode)
        
        logger.info(f"🏁 回合结束: {self.current_episode.episode_id}")
        logger.info(f"   总奖励: {self.current_episode.total_reward:.3f}")
        logger.info(f"   数据点数: {len(self.current_episode.data_points)}")
        logger.info(f"   任务成功: {success}")
        
        completed_episode = self.current_episode
        self.current_episode = None
        
        return completed_episode
    
    def save_rlhf_data(self, format_type: str = "json") -> str:
        """
        保存RLHF数据
        
        Args:
            format_type: 保存格式 ("json", "hdf5", "numpy")
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "json":
            return self._save_json_format(timestamp)
        elif format_type == "hdf5":
            return self._save_hdf5_format(timestamp)
        elif format_type == "numpy":
            return self._save_numpy_format(timestamp)
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
    
    def _save_json_format(self, timestamp: str) -> str:
        """保存为JSON格式"""
        filename = f"rlhf_data_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # 准备数据
        data = {
            "metadata": {
                "collection_time": datetime.now().isoformat(),
                "total_episodes": len(self.episodes),
                "total_data_points": sum(len(ep.data_points) for ep in self.episodes),
                "config": self.rlhf_config
            },
            "episodes": []
        }
        
        # 转换回合数据
        for episode in self.episodes:
            episode_data = {
                "episode_id": episode.episode_id,
                "scenario_type": episode.scenario_type,
                "start_time": episode.start_time.isoformat(),
                "end_time": episode.end_time.isoformat() if episode.end_time else None,
                "total_reward": episode.total_reward,
                "success": episode.success,
                "metadata": episode.metadata,
                "data_points": []
            }
            
            # 转换数据点
            for dp in episode.data_points:
                point_data = {
                    "timestamp": dp.timestamp.isoformat(),
                    "state": dp.state,
                    "action": dp.action,
                    "reward": dp.reward,
                    "next_state": dp.next_state,
                    "done": dp.done,
                    "info": dp.info
                }
                episode_data["data_points"].append(point_data)
            
            data["episodes"].append(episode_data)
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 RLHF数据已保存为JSON格式: {filepath}")
        return str(filepath)
    
    def _save_hdf5_format(self, timestamp: str) -> str:
        """保存为HDF5格式"""
        filename = f"rlhf_data_{timestamp}.h5"
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # 保存元数据
            metadata_group = f.create_group("metadata")
            metadata_group.attrs["collection_time"] = datetime.now().isoformat()
            metadata_group.attrs["total_episodes"] = len(self.episodes)
            
            # 保存回合数据
            episodes_group = f.create_group("episodes")
            
            for i, episode in enumerate(self.episodes):
                ep_group = episodes_group.create_group(f"episode_{i}")
                ep_group.attrs["episode_id"] = episode.episode_id
                ep_group.attrs["scenario_type"] = episode.scenario_type
                ep_group.attrs["total_reward"] = episode.total_reward
                ep_group.attrs["success"] = episode.success
                
                # 保存状态、动作、奖励数组
                if episode.data_points:
                    states = [dp.state for dp in episode.data_points]
                    actions = [dp.action for dp in episode.data_points]
                    rewards = [dp.reward for dp in episode.data_points]
                    
                    # 这里需要将字典转换为数组格式
                    # 简化版本，实际需要更复杂的序列化
                    ep_group.create_dataset("rewards", data=rewards)
        
        logger.info(f"💾 RLHF数据已保存为HDF5格式: {filepath}")
        return str(filepath)
    
    def _save_numpy_format(self, timestamp: str) -> str:
        """保存为NumPy格式"""
        filename = f"rlhf_data_{timestamp}.npz"
        filepath = self.output_dir / filename
        
        # 准备数组数据
        all_rewards = []
        all_states = []
        all_actions = []
        
        for episode in self.episodes:
            for dp in episode.data_points:
                all_rewards.append(dp.reward)
                # 这里需要将状态和动作转换为数组格式
                # 简化版本
                all_states.append(list(dp.state.values()))
                all_actions.append(list(dp.action.values()))
        
        # 保存数组
        np.savez(filepath,
                rewards=np.array(all_rewards),
                states=np.array(all_states),
                actions=np.array(all_actions))
        
        logger.info(f"💾 RLHF数据已保存为NumPy格式: {filepath}")
        return str(filepath)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据采集统计信息 - 增强版本"""
        try:
            # 基础统计
            basic_stats = {
                "total_episodes": len(self.episodes),
                "total_data_points": self.collection_stats['total_data_points'],
                "valid_data_points": self.collection_stats['valid_data_points'],
                "invalid_data_points": self.collection_stats['invalid_data_points'],
                "data_quality_rate": (self.collection_stats['valid_data_points'] /
                                    max(1, self.collection_stats['total_data_points'])),
                "average_reward": self.collection_stats['average_reward'],
                "average_quality_score": self.collection_stats['average_quality_score']
            }

            # 回合统计
            if self.episodes:
                episode_data_points = sum(len(ep.data_points) for ep in self.episodes)
                successful_episodes = sum(1 for ep in self.episodes if ep.success)
                episode_rewards = [ep.total_reward for ep in self.episodes]

                basic_stats.update({
                    "successful_episodes": successful_episodes,
                    "episode_success_rate": successful_episodes / len(self.episodes),
                    "episode_data_points": episode_data_points,
                    "average_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
                    "average_episode_length": episode_data_points / len(self.episodes)
                })
            else:
                basic_stats.update({
                    "successful_episodes": 0,
                    "episode_success_rate": 0.0,
                    "episode_data_points": 0,
                    "average_episode_reward": 0.0,
                    "average_episode_length": 0.0
                })

            # 组件统计
            component_stats = {}

            # 奖励计算器统计
            if hasattr(self.reward_calculator, 'get_statistics'):
                component_stats['reward_calculator'] = self.reward_calculator.get_statistics()

            # 数据质量验证器统计
            component_stats['data_quality_validator'] = self.data_quality_validator.get_validation_statistics()

            # 动作执行器统计
            if self.action_executor:
                component_stats['action_executor'] = self.action_executor.get_execution_statistics()

            return {
                "basic_statistics": basic_stats,
                "component_statistics": component_stats,
                "collection_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"统计信息获取失败: {e}")
            return {
                "total_episodes": len(self.episodes),
                "total_data_points": self.collection_stats.get('total_data_points', 0),
                "error": str(e)
            }
    
    # 辅助方法
    def _encode_threat_level(self, threat_level: str) -> int:
        """编码威胁等级"""
        mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4, "unknown": 0}
        return mapping.get(threat_level.lower(), 0)
    
    def _encode_time_of_day(self, timestamp_str: str) -> float:
        """编码时间信息"""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            # 将时间编码为0-1之间的值
            return (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
        except:
            return 0.0
    

