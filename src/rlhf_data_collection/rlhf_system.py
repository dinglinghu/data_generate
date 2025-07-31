"""
RLHF数据采集系统主类
整合所有RLHF相关组件，提供统一的接口
"""

import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .rlhf_data_collector import RLHFDataCollector, Episode
from .scenario_generator import RLHFScenarioGenerator, ScenarioConfig
from ..utils.config_manager import get_config_manager
from ..utils.time_manager import get_time_manager

logger = logging.getLogger(__name__)

class RLHFDataCollectionSystem:
    """RLHF数据采集系统主类"""
    
    def __init__(self, base_system, config_path: str = None):
        """
        初始化RLHF数据采集系统
        
        Args:
            base_system: 基础STK数据采集系统
            config_path: 配置文件路径
        """
        self.base_system = base_system
        
        # 配置管理
        self.config_manager = get_config_manager(config_path)
        self.time_manager = get_time_manager(self.config_manager)
        
        # 加载RLHF配置
        self._load_rlhf_config()
        
        # 初始化RLHF组件
        self.rlhf_collector = RLHFDataCollector(
            base_system.data_collector,
            self.config_manager,
            self.time_manager
        )
        
        self.scenario_generator = RLHFScenarioGenerator(
            self.config_manager,
            self.time_manager
        )
        
        # 数据存储
        self.collected_episodes = []
        self.current_scenario = None
        
        # 输出目录
        self.output_dir = Path("output/rlhf_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🤖 RLHF数据采集系统初始化完成")
    
    def _load_rlhf_config(self):
        """加载RLHF配置"""
        try:
            # 尝试加载RLHF场景配置
            rlhf_config_path = Path("config/rlhf_scenarios.yaml")
            if rlhf_config_path.exists():
                import yaml
                with open(rlhf_config_path, 'r', encoding='utf-8') as f:
                    rlhf_config = yaml.safe_load(f)
                
                # 合并到主配置中
                self.config_manager.config.update(rlhf_config)
                logger.info("✅ RLHF场景配置加载成功")
            else:
                logger.warning("⚠️ RLHF场景配置文件不存在，使用默认配置")
                
        except Exception as e:
            logger.error(f"❌ 加载RLHF配置失败: {e}")
    
    async def generate_training_dataset(self, num_scenarios: int = 100,
                                      difficulty_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """
        生成训练数据集
        
        Args:
            num_scenarios: 场景数量
            difficulty_distribution: 难度分布
            
        Returns:
            数据集统计信息
        """
        logger.info(f"🎯 开始生成训练数据集，目标场景数: {num_scenarios}")
        
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
        
        # 生成场景配置
        scenarios = self.scenario_generator.generate_training_scenarios(
            num_scenarios, difficulty_distribution
        )
        
        # 采集数据
        successful_episodes = 0
        failed_episodes = 0
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"📊 处理场景 {i+1}/{num_scenarios}: {scenario.scenario_id}")
            
            try:
                # 执行单个场景的数据采集
                episode = await self._collect_scenario_data(scenario)
                
                if episode and episode.success:
                    successful_episodes += 1
                    self.collected_episodes.append(episode)
                else:
                    failed_episodes += 1
                    
            except Exception as e:
                logger.error(f"❌ 场景 {scenario.scenario_id} 数据采集失败: {e}")
                failed_episodes += 1
        
        # 保存数据集
        dataset_file = self._save_training_dataset()
        
        # 统计信息
        stats = {
            "total_scenarios": num_scenarios,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": successful_episodes / num_scenarios if num_scenarios > 0 else 0,
            "dataset_file": dataset_file,
            "total_data_points": sum(len(ep.data_points) for ep in self.collected_episodes)
        }
        
        logger.info("🎉 训练数据集生成完成")
        logger.info(f"📊 统计信息: {stats}")
        
        return stats
    
    async def generate_evaluation_dataset(self, num_scenarios: int = 50) -> Dict[str, Any]:
        """
        生成评估数据集
        
        Args:
            num_scenarios: 评估场景数量
            
        Returns:
            评估数据集统计信息
        """
        logger.info(f"📋 开始生成评估数据集，目标场景数: {num_scenarios}")
        
        # 生成评估场景
        eval_scenarios = self.scenario_generator.generate_evaluation_scenarios(num_scenarios)
        
        # 采集评估数据
        eval_episodes = []
        
        for i, scenario in enumerate(eval_scenarios):
            logger.info(f"📊 处理评估场景 {i+1}/{num_scenarios}: {scenario.scenario_id}")
            
            try:
                episode = await self._collect_scenario_data(scenario, is_evaluation=True)
                if episode:
                    eval_episodes.append(episode)
                    
            except Exception as e:
                logger.error(f"❌ 评估场景 {scenario.scenario_id} 数据采集失败: {e}")
        
        # 保存评估数据集
        eval_file = self._save_evaluation_dataset(eval_episodes)
        
        # 计算评估指标
        eval_metrics = self._calculate_evaluation_metrics(eval_episodes)
        
        stats = {
            "total_scenarios": num_scenarios,
            "collected_episodes": len(eval_episodes),
            "evaluation_file": eval_file,
            "evaluation_metrics": eval_metrics
        }
        
        logger.info("📋 评估数据集生成完成")
        logger.info(f"📊 评估统计: {stats}")
        
        return stats
    
    async def _collect_scenario_data(self, scenario: ScenarioConfig, 
                                   is_evaluation: bool = False) -> Optional[Episode]:
        """
        采集单个场景的数据
        
        Args:
            scenario: 场景配置
            is_evaluation: 是否为评估模式
            
        Returns:
            采集的回合数据
        """
        try:
            # 设置当前场景
            self.current_scenario = scenario
            
            # 配置STK环境
            await self._setup_stk_scenario(scenario)
            
            # 开始RLHF回合
            episode_id = self.rlhf_collector.start_episode(
                scenario_type=scenario.scenario_type,
                scenario_params=scenario.__dict__
            )
            
            # 数据采集循环
            episode_done = False
            step_count = 0
            max_steps = scenario.time_constraints.get('scenario_duration', 3600) // 30  # 假设30秒一步
            
            while not episode_done and step_count < max_steps:
                # 获取当前状态
                current_time = self.time_manager.current_simulation_time
                base_data = self.base_system.data_collector.collect_data_at_time(current_time)
                
                if not base_data:
                    logger.warning("⚠️ 基础数据采集失败，跳过此步")
                    break
                
                # 生成动作 (这里使用模拟的专家动作)
                action = self._generate_expert_action(base_data, scenario)
                
                # 采集RLHF数据点
                data_point = self.rlhf_collector.collect_rlhf_data_point(action)
                
                if data_point:
                    episode_done = data_point.done
                    step_count += 1
                    
                    # 推进仿真时间
                    next_time = current_time + timedelta(seconds=30)
                    self.time_manager.advance_simulation_time(next_time)
                else:
                    logger.warning("⚠️ RLHF数据点采集失败")
                    break
            
            # 结束回合
            mission_success = self._evaluate_mission_success(scenario)
            episode = self.rlhf_collector.end_episode(success=mission_success)
            
            logger.info(f"✅ 场景 {scenario.scenario_id} 数据采集完成")
            logger.info(f"   步数: {step_count}, 成功: {mission_success}")
            
            return episode
            
        except Exception as e:
            logger.error(f"❌ 场景 {scenario.scenario_id} 数据采集异常: {e}")
            return None
    
    async def _setup_stk_scenario(self, scenario: ScenarioConfig):
        """设置STK场景"""
        try:
            # 更新星座配置
            constellation_config = scenario.constellation_config
            self.config_manager.config['constellation'].update(constellation_config)
            
            # 重新创建星座 (如果需要)
            if hasattr(self.base_system, 'constellation_manager'):
                success = self.base_system.constellation_manager.create_walker_constellation()
                if not success:
                    raise Exception("星座创建失败")
            
            # 创建导弹目标
            await self._create_scenario_missiles(scenario)
            
            logger.info(f"✅ STK场景设置完成: {scenario.scenario_id}")
            
        except Exception as e:
            logger.error(f"❌ STK场景设置失败: {e}")
            raise
    
    async def _create_scenario_missiles(self, scenario: ScenarioConfig):
        """创建场景导弹"""
        try:
            missile_count = scenario.missile_count
            missile_config = self.config_manager.get_missile_config()
            
            for i in range(missile_count):
                missile_id = f"{scenario.scenario_id}_missile_{i+1:02d}"
                
                # 生成随机位置
                launch_position = self._generate_random_position(
                    missile_config["global_launch_positions"]
                )
                target_position = self._generate_random_position(
                    missile_config["global_target_positions"]
                )
                
                # 生成轨迹参数
                trajectory_params = self._generate_trajectory_params(
                    missile_config["trajectory_params"]
                )
                
                # 创建导弹场景配置
                missile_scenario = {
                    "missile_id": missile_id,
                    "launch_position": launch_position,
                    "target_position": target_position,
                    "trajectory_params": trajectory_params,
                    "launch_time": self.time_manager.current_simulation_time
                }
                
                # 创建导弹
                if hasattr(self.base_system, 'missile_manager'):
                    result = self.base_system.missile_manager.create_single_missile_target(missile_scenario)
                    if result:
                        logger.info(f"✅ 导弹创建成功: {missile_id}")
                    else:
                        logger.warning(f"⚠️ 导弹创建失败: {missile_id}")
                        
        except Exception as e:
            logger.error(f"❌ 场景导弹创建失败: {e}")
            raise
    
    def _generate_expert_action(self, base_data: Dict[str, Any], 
                              scenario: ScenarioConfig) -> Dict[str, Any]:
        """
        生成专家动作 (模拟)
        
        Args:
            base_data: 基础数据
            scenario: 场景配置
            
        Returns:
            专家动作
        """
        # 这里实现简化的专家策略
        # 实际应用中可以集成真实的专家系统或最优控制算法
        
        satellites = base_data.get('satellites', [])
        missiles = base_data.get('missiles', [])
        visibility = base_data.get('visibility', [])
        
        action = {
            "satellite_actions": {},
            "mission_actions": {
                "target_assignments": [],
                "resource_allocation": {},
                "coordination_commands": {}
            }
        }
        
        # 简单的目标分配策略：每个卫星跟踪最近的导弹
        for sat in satellites:
            sat_id = sat.get('satellite_id', '')
            
            # 找到可见的导弹
            visible_missiles = [
                vis for vis in visibility 
                if vis.get('satellite_id') == sat_id and vis.get('has_visibility', False)
            ]
            
            if visible_missiles:
                # 选择第一个可见的导弹
                target_missile = visible_missiles[0].get('missile_id', '')
                
                action["mission_actions"]["target_assignments"].append({
                    "satellite_id": sat_id,
                    "target_id": target_missile,
                    "priority": 1,
                    "assignment_duration": 300.0
                })
                
                # 设置载荷指向
                action["satellite_actions"][sat_id] = {
                    "payload_pointing": {
                        "target_coordinates": [0, 0, 0],  # 简化
                        "pointing_mode": "tracking"
                    },
                    "power_management": {
                        "power_allocation": {
                            "payload": 0.6,
                            "communication": 0.2,
                            "attitude_control": 0.2
                        }
                    }
                }
        
        return action
    
    def _evaluate_mission_success(self, scenario: ScenarioConfig) -> bool:
        """评估任务成功"""
        # 简化的成功评估：如果有可见性记录就认为成功
        # 实际应用中应该有更复杂的成功标准
        
        try:
            if self.rlhf_collector.current_episode:
                data_points = self.rlhf_collector.current_episode.data_points
                
                # 检查是否有有效的跟踪记录
                for dp in data_points:
                    visibility_matrix = dp.state.get('visibility_matrix', [])
                    if any(any(row) for row in visibility_matrix):
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"❌ 任务成功评估失败: {e}")
            return False
    
    def _save_training_dataset(self) -> str:
        """保存训练数据集"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_dataset_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # 准备数据
        dataset = {
            "metadata": {
                "dataset_type": "training",
                "generation_time": datetime.now().isoformat(),
                "total_episodes": len(self.collected_episodes),
                "total_data_points": sum(len(ep.data_points) for ep in self.collected_episodes),
                "scenario_distribution": self._get_scenario_distribution()
            },
            "episodes": []
        }
        
        # 转换回合数据
        for episode in self.collected_episodes:
            episode_data = self._convert_episode_to_dict(episode)
            dataset["episodes"].append(episode_data)
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 训练数据集已保存: {filepath}")
        return str(filepath)
    
    def _save_evaluation_dataset(self, eval_episodes: List[Episode]) -> str:
        """保存评估数据集"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_dataset_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # 准备数据
        dataset = {
            "metadata": {
                "dataset_type": "evaluation",
                "generation_time": datetime.now().isoformat(),
                "total_episodes": len(eval_episodes),
                "total_data_points": sum(len(ep.data_points) for ep in eval_episodes)
            },
            "episodes": []
        }
        
        # 转换回合数据
        for episode in eval_episodes:
            episode_data = self._convert_episode_to_dict(episode)
            dataset["episodes"].append(episode_data)
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 评估数据集已保存: {filepath}")
        return str(filepath)
    
    def _convert_episode_to_dict(self, episode: Episode) -> Dict[str, Any]:
        """将回合转换为字典格式"""
        return {
            "episode_id": episode.episode_id,
            "scenario_type": episode.scenario_type,
            "start_time": episode.start_time.isoformat(),
            "end_time": episode.end_time.isoformat() if episode.end_time else None,
            "total_reward": episode.total_reward,
            "success": episode.success,
            "metadata": episode.metadata,
            "data_points": [
                {
                    "timestamp": dp.timestamp.isoformat(),
                    "state": dp.state,
                    "action": dp.action,
                    "reward": dp.reward,
                    "next_state": dp.next_state,
                    "done": dp.done,
                    "info": dp.info
                }
                for dp in episode.data_points
            ]
        }
    
    def _get_scenario_distribution(self) -> Dict[str, int]:
        """获取场景分布统计"""
        distribution = {}
        for episode in self.collected_episodes:
            scenario_type = episode.scenario_type
            distribution[scenario_type] = distribution.get(scenario_type, 0) + 1
        return distribution
    
    def _calculate_evaluation_metrics(self, eval_episodes: List[Episode]) -> Dict[str, float]:
        """计算评估指标"""
        if not eval_episodes:
            return {}
        
        # 基础统计
        total_episodes = len(eval_episodes)
        successful_episodes = sum(1 for ep in eval_episodes if ep.success)
        total_rewards = [ep.total_reward for ep in eval_episodes]
        episode_lengths = [len(ep.data_points) for ep in eval_episodes]
        
        metrics = {
            "success_rate": successful_episodes / total_episodes,
            "average_reward": sum(total_rewards) / len(total_rewards),
            "average_episode_length": sum(episode_lengths) / len(episode_lengths),
            "reward_std": np.std(total_rewards) if len(total_rewards) > 1 else 0.0
        }
        
        return metrics
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        rlhf_stats = self.rlhf_collector.get_statistics()
        scenario_stats = self.scenario_generator.get_scenario_statistics()
        
        return {
            "rlhf_collector": rlhf_stats,
            "scenario_generator": scenario_stats,
            "total_collected_episodes": len(self.collected_episodes),
            "current_scenario": self.current_scenario.scenario_id if self.current_scenario else None
        }
    
    # 辅助方法
    def _generate_random_position(self, position_config: dict) -> dict:
        """生成随机位置"""
        import random
        
        lat_range = position_config.get("lat_range", [-60, 60])
        lon_range = position_config.get("lon_range", [-180, 180])
        alt_range = position_config.get("alt_range", [0, 100])
        
        return {
            "lat": random.uniform(lat_range[0], lat_range[1]),
            "lon": random.uniform(lon_range[0], lon_range[1]),
            "alt": random.uniform(alt_range[0], alt_range[1])
        }
    
    def _generate_trajectory_params(self, trajectory_config: dict) -> dict:
        """生成轨迹参数"""
        import random
        
        max_alt_range = trajectory_config.get("max_altitude_range", [300, 1500])
        flight_time_range = trajectory_config.get("flight_time_range", [600, 1800])
        
        return {
            "max_altitude": random.uniform(max_alt_range[0], max_alt_range[1]),
            "flight_time": random.uniform(flight_time_range[0], flight_time_range[1])
        }
