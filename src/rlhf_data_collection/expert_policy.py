"""
RLHF专家策略生成器
实现启发式专家策略，用于生成高质量的训练数据
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class RLHFExpertPolicy:
    """RLHF专家策略生成器"""
    
    def __init__(self, config_manager):
        """
        初始化专家策略
        
        Args:
            config_manager: 配置管理器
        """
        self.config_manager = config_manager
        
        # 策略配置
        self.policy_config = config_manager.config.get('expert_demonstrations', {})
        
        # 策略类型
        self.strategy_types = {
            'optimal_tracking': self._optimal_tracking_strategy,
            'resource_efficient': self._resource_efficient_strategy,
            'robust_defense': self._robust_defense_strategy,
            'balanced': self._balanced_strategy
        }
        
        # 当前策略类型
        self.current_strategy = 'balanced'
        
        # 策略参数
        self.strategy_params = {
            'tracking_priority_threshold': 0.7,
            'resource_utilization_limit': 0.8,
            'threat_level_weights': {1: 0.2, 2: 0.5, 3: 0.8, 4: 1.0},
            'coordination_threshold': 3,  # 导弹数量超过此值时启用协调
            'power_conservation_mode': False
        }
        
        logger.info("🧠 RLHF专家策略生成器初始化完成")
    
    def get_expert_action(self, state: Dict[str, Any], base_data: Dict[str, Any], 
                         strategy_type: str = None) -> Dict[str, Any]:
        """
        获取专家动作
        
        Args:
            state: 当前状态
            base_data: 基础数据
            strategy_type: 策略类型
            
        Returns:
            专家动作
        """
        try:
            # 选择策略
            strategy = strategy_type or self.current_strategy
            if strategy not in self.strategy_types:
                strategy = 'balanced'
                logger.warning(f"未知策略类型 {strategy_type}，使用默认策略")
            
            # 执行策略
            action = self.strategy_types[strategy](state, base_data)
            
            # 添加策略元信息
            action['strategy_info'] = {
                'strategy_type': strategy,
                'generation_time': datetime.now().isoformat(),
                'confidence': self._calculate_action_confidence(action, state, base_data)
            }
            
            logger.debug(f"专家动作生成完成: 策略={strategy}, 置信度={action['strategy_info']['confidence']:.3f}")
            
            return action
            
        except Exception as e:
            logger.error(f"专家动作生成失败: {e}")
            return self._get_default_action(state, base_data)
    
    def _optimal_tracking_strategy(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> Dict[str, Any]:
        """最优跟踪策略 - 专注于最大化跟踪性能"""
        try:
            action = {
                'satellite_actions': {},
                'mission_actions': {
                    'target_assignments': [],
                    'resource_allocation': {},
                    'coordination_commands': {}
                }
            }
            
            satellites = base_data.get('satellites', [])
            missiles = base_data.get('missiles', [])
            visibility_data = base_data.get('visibility', [])
            
            # 构建可见性映射
            visibility_map = self._build_visibility_map(visibility_data)
            
            # 威胁优先级排序
            sorted_missiles = self._sort_missiles_by_threat(missiles, state)
            
            # 为每个卫星分配最高优先级的可见目标
            assigned_missiles = set()
            
            for sat in satellites:
                sat_id = sat.get('satellite_id', '')
                
                # 找到该卫星可见的导弹
                visible_missiles = [
                    missile for missile in sorted_missiles
                    if missile.get('missile_id', '') in visibility_map.get(sat_id, [])
                    and missile.get('missile_id', '') not in assigned_missiles
                ]
                
                if visible_missiles:
                    # 选择最高优先级的导弹
                    target_missile = visible_missiles[0]
                    missile_id = target_missile.get('missile_id', '')
                    
                    # 创建任务分配
                    action['mission_actions']['target_assignments'].append({
                        'satellite_id': sat_id,
                        'target_id': missile_id,
                        'priority': self._get_missile_priority(target_missile, state),
                        'assignment_duration': 300.0
                    })
                    
                    # 配置卫星动作 - 最大性能模式
                    action['satellite_actions'][sat_id] = {
                        'payload_pointing': {
                            'target_coordinates': target_missile.get('position', {}).values(),
                            'pointing_mode': 'tracking',
                            'scan_pattern': 'none'
                        },
                        'power_management': {
                            'power_allocation': {
                                'payload': 0.8,  # 最大载荷功率
                                'communication': 0.15,
                                'attitude_control': 0.05
                            }
                        }
                    }
                    
                    assigned_missiles.add(missile_id)
            
            return action
            
        except Exception as e:
            logger.error(f"最优跟踪策略失败: {e}")
            return self._get_default_action(state, base_data)
    
    def _resource_efficient_strategy(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> Dict[str, Any]:
        """资源高效策略 - 专注于最小化资源消耗"""
        try:
            action = {
                'satellite_actions': {},
                'mission_actions': {
                    'target_assignments': [],
                    'resource_allocation': {},
                    'coordination_commands': {}
                }
            }
            
            satellites = base_data.get('satellites', [])
            missiles = base_data.get('missiles', [])
            visibility_data = base_data.get('visibility', [])
            
            # 构建可见性映射
            visibility_map = self._build_visibility_map(visibility_data)
            
            # 选择最少的卫星来覆盖所有威胁
            optimal_assignments = self._find_minimal_coverage(satellites, missiles, visibility_map)
            
            for sat_id, missile_id in optimal_assignments.items():
                # 创建任务分配
                action['mission_actions']['target_assignments'].append({
                    'satellite_id': sat_id,
                    'target_id': missile_id,
                    'priority': 1,
                    'assignment_duration': 600.0  # 更长的分配时间以减少切换
                })
                
                # 配置卫星动作 - 节能模式
                action['satellite_actions'][sat_id] = {
                    'payload_pointing': {
                        'target_coordinates': [0, 0, 0],  # 简化
                        'pointing_mode': 'tracking',
                        'scan_pattern': 'none'
                    },
                    'power_management': {
                        'power_allocation': {
                            'payload': 0.5,  # 中等载荷功率
                            'communication': 0.3,
                            'attitude_control': 0.2
                        }
                    }
                }
            
            return action
            
        except Exception as e:
            logger.error(f"资源高效策略失败: {e}")
            return self._get_default_action(state, base_data)
    
    def _robust_defense_strategy(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> Dict[str, Any]:
        """鲁棒防御策略 - 专注于应对复杂威胁"""
        try:
            action = {
                'satellite_actions': {},
                'mission_actions': {
                    'target_assignments': [],
                    'resource_allocation': {},
                    'coordination_commands': {}
                }
            }
            
            satellites = base_data.get('satellites', [])
            missiles = base_data.get('missiles', [])
            visibility_data = base_data.get('visibility', [])
            
            # 威胁评估
            high_threat_missiles = [
                m for m in missiles 
                if self._get_missile_priority(m, state) >= 3
            ]
            
            # 如果有高威胁目标，启用协调模式
            if len(high_threat_missiles) >= self.strategy_params['coordination_threshold']:
                action['mission_actions']['coordination_commands'] = {
                    'collaborative_tracking': True,
                    'handover_instructions': [],
                    'formation_adjustment': {'mode': 'defensive'}
                }
            
            # 多卫星协同跟踪高威胁目标
            for missile in high_threat_missiles:
                missile_id = missile.get('missile_id', '')
                
                # 找到所有可以跟踪此导弹的卫星
                capable_satellites = [
                    sat for sat in satellites
                    if missile_id in self._build_visibility_map(visibility_data).get(
                        sat.get('satellite_id', ''), []
                    )
                ]
                
                # 分配多个卫星跟踪同一高威胁目标
                for i, sat in enumerate(capable_satellites[:2]):  # 最多2个卫星
                    sat_id = sat.get('satellite_id', '')
                    
                    action['mission_actions']['target_assignments'].append({
                        'satellite_id': sat_id,
                        'target_id': missile_id,
                        'priority': 4,  # 最高优先级
                        'assignment_duration': 180.0  # 较短时间以便快速调整
                    })
                    
                    # 配置卫星动作 - 高性能模式
                    action['satellite_actions'][sat_id] = {
                        'payload_pointing': {
                            'target_coordinates': [0, 0, 0],
                            'pointing_mode': 'tracking',
                            'scan_pattern': 'adaptive'
                        },
                        'power_management': {
                            'power_allocation': {
                                'payload': 0.7,
                                'communication': 0.2,
                                'attitude_control': 0.1
                            }
                        }
                    }
            
            return action
            
        except Exception as e:
            logger.error(f"鲁棒防御策略失败: {e}")
            return self._get_default_action(state, base_data)
    
    def _balanced_strategy(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> Dict[str, Any]:
        """平衡策略 - 在性能和效率之间平衡"""
        try:
            action = {
                'satellite_actions': {},
                'mission_actions': {
                    'target_assignments': [],
                    'resource_allocation': {},
                    'coordination_commands': {}
                }
            }
            
            satellites = base_data.get('satellites', [])
            missiles = base_data.get('missiles', [])
            visibility_data = base_data.get('visibility', [])
            
            # 构建可见性映射
            visibility_map = self._build_visibility_map(visibility_data)
            
            # 威胁优先级排序
            sorted_missiles = self._sort_missiles_by_threat(missiles, state)
            
            # 平衡分配策略
            assigned_missiles = set()
            
            for sat in satellites:
                sat_id = sat.get('satellite_id', '')
                
                # 找到可见的未分配导弹
                visible_missiles = [
                    missile for missile in sorted_missiles
                    if missile.get('missile_id', '') in visibility_map.get(sat_id, [])
                    and missile.get('missile_id', '') not in assigned_missiles
                ]
                
                if visible_missiles:
                    target_missile = visible_missiles[0]
                    missile_id = target_missile.get('missile_id', '')
                    priority = self._get_missile_priority(target_missile, state)
                    
                    # 创建任务分配
                    action['mission_actions']['target_assignments'].append({
                        'satellite_id': sat_id,
                        'target_id': missile_id,
                        'priority': priority,
                        'assignment_duration': 300.0
                    })
                    
                    # 根据威胁等级调整功率分配
                    if priority >= 3:
                        # 高威胁 - 高性能模式
                        power_allocation = {'payload': 0.7, 'communication': 0.2, 'attitude_control': 0.1}
                    else:
                        # 低威胁 - 节能模式
                        power_allocation = {'payload': 0.5, 'communication': 0.3, 'attitude_control': 0.2}
                    
                    action['satellite_actions'][sat_id] = {
                        'payload_pointing': {
                            'target_coordinates': [0, 0, 0],
                            'pointing_mode': 'tracking',
                            'scan_pattern': 'none'
                        },
                        'power_management': {
                            'power_allocation': power_allocation
                        }
                    }
                    
                    assigned_missiles.add(missile_id)
            
            return action
            
        except Exception as e:
            logger.error(f"平衡策略失败: {e}")
            return self._get_default_action(state, base_data)
    
    def _build_visibility_map(self, visibility_data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建可见性映射"""
        visibility_map = {}
        
        for vis in visibility_data:
            if vis.get('has_visibility', False):
                sat_id = vis.get('satellite_id', '')
                missile_id = vis.get('missile_id', '')
                
                if sat_id not in visibility_map:
                    visibility_map[sat_id] = []
                visibility_map[sat_id].append(missile_id)
        
        return visibility_map
    
    def _sort_missiles_by_threat(self, missiles: List[Dict[str, Any]], 
                                state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """按威胁等级排序导弹"""
        def threat_score(missile):
            priority = self._get_missile_priority(missile, state)
            # 可以添加更多因素，如剩余时间、距离等
            return priority
        
        return sorted(missiles, key=threat_score, reverse=True)
    
    def _get_missile_priority(self, missile: Dict[str, Any], state: Dict[str, Any]) -> int:
        """获取导弹优先级"""
        # 从状态中获取威胁等级
        missile_threat_levels = state.get('missile_threat_levels', [])
        missiles = state.get('missile_positions', [])
        
        # 简化：基于导弹索引获取威胁等级
        missile_id = missile.get('missile_id', '')
        
        # 这里应该有更复杂的逻辑来确定优先级
        # 简化版本：返回默认优先级
        return 2
    
    def _find_minimal_coverage(self, satellites: List[Dict[str, Any]], 
                              missiles: List[Dict[str, Any]], 
                              visibility_map: Dict[str, List[str]]) -> Dict[str, str]:
        """找到最小覆盖分配"""
        assignments = {}
        covered_missiles = set()
        
        # 贪心算法：优先分配能覆盖最多未覆盖导弹的卫星
        for sat in satellites:
            sat_id = sat.get('satellite_id', '')
            visible_missiles = visibility_map.get(sat_id, [])
            
            # 找到未覆盖的可见导弹
            uncovered_visible = [m for m in visible_missiles if m not in covered_missiles]
            
            if uncovered_visible:
                # 选择第一个未覆盖的导弹
                target_missile = uncovered_visible[0]
                assignments[sat_id] = target_missile
                covered_missiles.add(target_missile)
        
        return assignments
    
    def _calculate_action_confidence(self, action: Dict[str, Any], 
                                   state: Dict[str, Any], 
                                   base_data: Dict[str, Any]) -> float:
        """计算动作置信度"""
        try:
            confidence = 1.0
            
            # 基于分配覆盖率的置信度
            assignments = action.get('mission_actions', {}).get('target_assignments', [])
            total_missiles = len(base_data.get('missiles', []))
            
            if total_missiles > 0:
                coverage_ratio = len(assignments) / total_missiles
                confidence *= min(1.0, coverage_ratio + 0.5)
            
            # 基于资源利用的置信度
            satellite_actions = action.get('satellite_actions', {})
            if satellite_actions:
                power_allocations = []
                for sat_action in satellite_actions.values():
                    power_mgmt = sat_action.get('power_management', {})
                    power_allocation = power_mgmt.get('power_allocation', {})
                    total_power = sum(power_allocation.values())
                    power_allocations.append(total_power)
                
                if power_allocations:
                    avg_power = np.mean(power_allocations)
                    # 功率利用率在0.5-1.0之间时置信度最高
                    if 0.5 <= avg_power <= 1.0:
                        confidence *= 1.0
                    else:
                        confidence *= 0.8
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5
    
    def _get_default_action(self, state: Dict[str, Any], base_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取默认动作"""
        return {
            'satellite_actions': {},
            'mission_actions': {
                'target_assignments': [],
                'resource_allocation': {},
                'coordination_commands': {}
            },
            'strategy_info': {
                'strategy_type': 'default',
                'generation_time': datetime.now().isoformat(),
                'confidence': 0.1
            }
        }
    
    def set_strategy_type(self, strategy_type: str):
        """设置策略类型"""
        if strategy_type in self.strategy_types:
            self.current_strategy = strategy_type
            logger.info(f"策略类型已设置为: {strategy_type}")
        else:
            logger.warning(f"未知策略类型: {strategy_type}")
    
    def get_available_strategies(self) -> List[str]:
        """获取可用策略列表"""
        return list(self.strategy_types.keys())
