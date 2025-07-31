"""
RLHF场景生成器
用于生成多样化的强化学习训练场景
"""

import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ScenarioConfig:
    """场景配置"""
    scenario_id: str
    scenario_type: str
    difficulty_level: str
    missile_count: int
    constellation_config: Dict[str, Any]
    environment_conditions: Dict[str, Any]
    mission_objectives: List[str]
    time_constraints: Dict[str, Any]

class RLHFScenarioGenerator:
    """RLHF场景生成器"""
    
    def __init__(self, config_manager, time_manager):
        """
        初始化场景生成器
        
        Args:
            config_manager: 配置管理器
            time_manager: 时间管理器
        """
        self.config_manager = config_manager
        self.time_manager = time_manager
        
        # 加载场景配置
        self.scenario_config = config_manager.config.get('scenario_diversity', {})
        self.rl_config = config_manager.config.get('rlhf_data_collection', {})
        
        # 场景模板
        self.scenario_templates = self._load_scenario_templates()
        
        # 生成历史
        self.generated_scenarios = []
        
        logger.info("🎭 RLHF场景生成器初始化完成")
    
    def generate_training_scenarios(self, num_scenarios: int, 
                                  difficulty_distribution: Dict[str, float] = None) -> List[ScenarioConfig]:
        """
        生成训练场景集合
        
        Args:
            num_scenarios: 场景数量
            difficulty_distribution: 难度分布 {"easy": 0.3, "medium": 0.4, "hard": 0.3}
            
        Returns:
            场景配置列表
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
        
        scenarios = []
        
        for i in range(num_scenarios):
            # 选择难度等级
            difficulty = self._sample_difficulty(difficulty_distribution)
            
            # 选择场景类型
            scenario_type = self._select_scenario_type(difficulty)
            
            # 生成场景
            scenario = self._generate_single_scenario(
                scenario_id=f"training_scenario_{i+1:04d}",
                scenario_type=scenario_type,
                difficulty_level=difficulty
            )
            
            scenarios.append(scenario)
            self.generated_scenarios.append(scenario)
        
        logger.info(f"🎯 生成了 {num_scenarios} 个训练场景")
        self._log_scenario_distribution(scenarios)
        
        return scenarios
    
    def generate_evaluation_scenarios(self, num_scenarios: int) -> List[ScenarioConfig]:
        """
        生成评估场景集合
        
        Args:
            num_scenarios: 场景数量
            
        Returns:
            评估场景配置列表
        """
        scenarios = []
        
        # 确保评估场景覆盖所有难度等级和场景类型
        difficulty_levels = ["easy", "medium", "hard", "extreme"]
        scenario_types = ["single_threat", "multiple_threats", "saturation_attack", "evasive_targets"]
        
        scenarios_per_combination = max(1, num_scenarios // (len(difficulty_levels) * len(scenario_types)))
        
        scenario_id = 1
        for difficulty in difficulty_levels:
            for scenario_type in scenario_types:
                for _ in range(scenarios_per_combination):
                    if len(scenarios) >= num_scenarios:
                        break
                        
                    scenario = self._generate_single_scenario(
                        scenario_id=f"eval_scenario_{scenario_id:04d}",
                        scenario_type=scenario_type,
                        difficulty_level=difficulty
                    )
                    
                    scenarios.append(scenario)
                    scenario_id += 1
                
                if len(scenarios) >= num_scenarios:
                    break
            if len(scenarios) >= num_scenarios:
                break
        
        logger.info(f"📊 生成了 {len(scenarios)} 个评估场景")
        return scenarios
    
    def _generate_single_scenario(self, scenario_id: str, scenario_type: str, 
                                 difficulty_level: str) -> ScenarioConfig:
        """
        生成单个场景
        
        Args:
            scenario_id: 场景ID
            scenario_type: 场景类型
            difficulty_level: 难度等级
            
        Returns:
            场景配置
        """
        # 获取场景模板
        template = self.scenario_templates.get(scenario_type, {})
        
        # 生成导弹配置
        missile_count = self._generate_missile_count(scenario_type, difficulty_level)
        
        # 生成星座配置
        constellation_config = self._generate_constellation_config(difficulty_level)
        
        # 生成环境条件
        environment_conditions = self._generate_environment_conditions()
        
        # 生成任务目标
        mission_objectives = self._generate_mission_objectives(scenario_type, difficulty_level)
        
        # 生成时间约束
        time_constraints = self._generate_time_constraints(scenario_type, difficulty_level)
        
        scenario = ScenarioConfig(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            difficulty_level=difficulty_level,
            missile_count=missile_count,
            constellation_config=constellation_config,
            environment_conditions=environment_conditions,
            mission_objectives=mission_objectives,
            time_constraints=time_constraints
        )
        
        logger.info(f"🎬 生成场景: {scenario_id} ({scenario_type}, {difficulty_level})")
        
        return scenario
    
    def _generate_missile_count(self, scenario_type: str, difficulty_level: str) -> int:
        """生成导弹数量"""
        threat_config = self.scenario_config.get('threat_scenarios', {})
        
        if scenario_type == "single_threat":
            base_count = 1
        elif scenario_type == "multiple_threats":
            base_range = threat_config.get('multiple_threats', {}).get('missile_count', [2, 8])
            base_count = random.randint(*base_range)
        elif scenario_type == "saturation_attack":
            base_range = threat_config.get('saturation_attack', {}).get('missile_count', [10, 20])
            base_count = random.randint(*base_range)
        else:
            base_count = random.randint(1, 5)
        
        # 根据难度调整
        difficulty_multipliers = {
            "easy": 0.7,
            "medium": 1.0,
            "hard": 1.3,
            "extreme": 1.6
        }
        
        multiplier = difficulty_multipliers.get(difficulty_level, 1.0)
        adjusted_count = int(base_count * multiplier)
        
        return max(1, adjusted_count)
    
    def _generate_constellation_config(self, difficulty_level: str) -> Dict[str, Any]:
        """生成星座配置"""
        constellation_variations = self.scenario_config.get('constellation_variations', {})
        sizes = constellation_variations.get('constellation_sizes', {})
        
        # 根据难度选择星座规模
        if difficulty_level == "easy":
            size_config = sizes.get('large', {'planes': 4, 'satellites_per_plane': 4})
        elif difficulty_level == "medium":
            size_config = sizes.get('medium', {'planes': 3, 'satellites_per_plane': 3})
        elif difficulty_level == "hard":
            size_config = sizes.get('small', {'planes': 2, 'satellites_per_plane': 2})
        else:  # extreme
            size_config = {'planes': 2, 'satellites_per_plane': 1}  # 最小配置
        
        # 添加轨道参数变化
        orbital_variations = constellation_variations.get('orbital_variations', {})
        
        config = {
            'planes': size_config['planes'],
            'satellites_per_plane': size_config['satellites_per_plane'],
            'total_satellites': size_config['planes'] * size_config['satellites_per_plane'],
            'reference_satellite': {
                'altitude': random.uniform(*orbital_variations.get('altitude_range', [800, 2000])),
                'inclination': random.uniform(*orbital_variations.get('inclination_range', [45, 98])),
                'eccentricity': random.uniform(*orbital_variations.get('eccentricity_range', [0.0, 0.1])),
                'arg_of_perigee': random.uniform(0, 360),
                'raan_offset': random.uniform(0, 360),
                'mean_anomaly_offset': random.uniform(0, 360)
            }
        }
        
        return config
    
    def _generate_environment_conditions(self) -> Dict[str, Any]:
        """生成环境条件"""
        env_variations = self.scenario_config.get('environment_variations', {})
        
        # 时间条件
        temporal = env_variations.get('temporal_conditions', {})
        time_of_day = random.choice(temporal.get('time_of_day', ['day']))
        season = random.choice(temporal.get('season', ['summer']))
        
        # 空间环境
        space_env = env_variations.get('space_environment', {})
        solar_activity = random.choice(space_env.get('solar_activity', ['medium']))
        atm_density_range = space_env.get('atmospheric_density', [0.8, 1.2])
        atmospheric_density = random.uniform(*atm_density_range)
        
        return {
            'time_of_day': time_of_day,
            'season': season,
            'solar_activity': solar_activity,
            'atmospheric_density': atmospheric_density,
            'weather_conditions': random.choice(['clear', 'cloudy', 'stormy']),
            'electromagnetic_interference': random.choice(['none', 'low', 'medium', 'high'])
        }
    
    def _generate_mission_objectives(self, scenario_type: str, difficulty_level: str) -> List[str]:
        """生成任务目标"""
        base_objectives = [
            "track_all_threats",
            "maintain_coverage",
            "minimize_resource_usage"
        ]
        
        # 根据场景类型添加特定目标
        if scenario_type == "single_threat":
            base_objectives.append("achieve_continuous_tracking")
        elif scenario_type == "multiple_threats":
            base_objectives.extend(["prioritize_threats", "coordinate_satellites"])
        elif scenario_type == "saturation_attack":
            base_objectives.extend(["rapid_threat_assessment", "emergency_response"])
        
        # 根据难度添加额外约束
        if difficulty_level in ["hard", "extreme"]:
            base_objectives.extend([
                "minimize_false_alarms",
                "maintain_communication_links",
                "handle_satellite_failures"
            ])
        
        return base_objectives
    
    def _generate_time_constraints(self, scenario_type: str, difficulty_level: str) -> Dict[str, Any]:
        """生成时间约束"""
        base_duration = 3600  # 1小时基础时长
        
        # 根据场景类型调整
        if scenario_type == "single_threat":
            duration_multiplier = 0.5
        elif scenario_type == "multiple_threats":
            duration_multiplier = 1.0
        elif scenario_type == "saturation_attack":
            duration_multiplier = 0.3
        else:
            duration_multiplier = 1.0
        
        # 根据难度调整
        difficulty_multipliers = {
            "easy": 1.5,
            "medium": 1.0,
            "hard": 0.7,
            "extreme": 0.5
        }
        
        total_multiplier = duration_multiplier * difficulty_multipliers.get(difficulty_level, 1.0)
        scenario_duration = int(base_duration * total_multiplier)
        
        return {
            'scenario_duration': scenario_duration,
            'max_response_time': scenario_duration * 0.1,  # 10%的时间作为最大响应时间
            'decision_interval': random.randint(30, 120),  # 决策间隔
            'evaluation_interval': random.randint(300, 600)  # 评估间隔
        }
    
    def _load_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载场景模板"""
        templates = {
            "single_threat": {
                "description": "单一威胁场景",
                "complexity": "low",
                "focus": "基础跟踪能力"
            },
            "multiple_threats": {
                "description": "多威胁场景",
                "complexity": "medium",
                "focus": "资源分配和协调"
            },
            "saturation_attack": {
                "description": "饱和攻击场景",
                "complexity": "high",
                "focus": "高压力下的决策"
            },
            "evasive_targets": {
                "description": "规避目标场景",
                "complexity": "high",
                "focus": "适应性跟踪"
            },
            "communication_failure": {
                "description": "通信故障场景",
                "complexity": "medium",
                "focus": "故障恢复能力"
            },
            "sensor_degradation": {
                "description": "传感器退化场景",
                "complexity": "medium",
                "focus": "性能退化适应"
            }
        }
        
        return templates
    
    def _sample_difficulty(self, distribution: Dict[str, float]) -> str:
        """根据分布采样难度等级"""
        difficulties = list(distribution.keys())
        probabilities = list(distribution.values())
        
        return np.random.choice(difficulties, p=probabilities)
    
    def _select_scenario_type(self, difficulty_level: str) -> str:
        """根据难度选择场景类型"""
        if difficulty_level == "easy":
            types = ["single_threat", "multiple_threats"]
            weights = [0.7, 0.3]
        elif difficulty_level == "medium":
            types = ["single_threat", "multiple_threats", "evasive_targets"]
            weights = [0.3, 0.5, 0.2]
        elif difficulty_level == "hard":
            types = ["multiple_threats", "saturation_attack", "evasive_targets", "communication_failure"]
            weights = [0.3, 0.3, 0.2, 0.2]
        else:  # extreme
            types = ["saturation_attack", "evasive_targets", "communication_failure", "sensor_degradation"]
            weights = [0.4, 0.3, 0.15, 0.15]
        
        return np.random.choice(types, p=weights)
    
    def _log_scenario_distribution(self, scenarios: List[ScenarioConfig]):
        """记录场景分布统计"""
        type_counts = {}
        difficulty_counts = {}
        
        for scenario in scenarios:
            # 统计场景类型
            type_counts[scenario.scenario_type] = type_counts.get(scenario.scenario_type, 0) + 1
            
            # 统计难度分布
            difficulty_counts[scenario.difficulty_level] = difficulty_counts.get(scenario.difficulty_level, 0) + 1
        
        logger.info("📊 场景分布统计:")
        logger.info(f"   场景类型: {type_counts}")
        logger.info(f"   难度分布: {difficulty_counts}")
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """获取场景生成统计信息"""
        if not self.generated_scenarios:
            return {"total_scenarios": 0}
        
        type_counts = {}
        difficulty_counts = {}
        missile_counts = []
        
        for scenario in self.generated_scenarios:
            type_counts[scenario.scenario_type] = type_counts.get(scenario.scenario_type, 0) + 1
            difficulty_counts[scenario.difficulty_level] = difficulty_counts.get(scenario.difficulty_level, 0) + 1
            missile_counts.append(scenario.missile_count)
        
        return {
            "total_scenarios": len(self.generated_scenarios),
            "scenario_types": type_counts,
            "difficulty_distribution": difficulty_counts,
            "average_missile_count": np.mean(missile_counts),
            "missile_count_range": [min(missile_counts), max(missile_counts)]
        }
    
    def export_scenarios(self, scenarios: List[ScenarioConfig], filepath: str):
        """导出场景配置"""
        import json
        
        scenario_data = []
        for scenario in scenarios:
            data = {
                "scenario_id": scenario.scenario_id,
                "scenario_type": scenario.scenario_type,
                "difficulty_level": scenario.difficulty_level,
                "missile_count": scenario.missile_count,
                "constellation_config": scenario.constellation_config,
                "environment_conditions": scenario.environment_conditions,
                "mission_objectives": scenario.mission_objectives,
                "time_constraints": scenario.time_constraints
            }
            scenario_data.append(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "generation_time": datetime.now().isoformat(),
                    "total_scenarios": len(scenarios),
                    "generator_version": "1.0"
                },
                "scenarios": scenario_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📁 场景配置已导出到: {filepath}")
