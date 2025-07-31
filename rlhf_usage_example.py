#!/usr/bin/env python3
"""
RLHF数据采集系统使用示例
展示如何使用完整的RLHF数据采集系统进行强化学习数据生成
"""

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rlhf_usage_example.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 导入RLHF模块
from src.rlhf_data_collection.rlhf_data_collector import RLHFDataCollector
from src.rlhf_data_collection.expert_policy import RLHFExpertPolicy
from src.utils.config_manager import get_config_manager
from src.utils.time_manager import get_time_manager

class RLHFUsageExample:
    """RLHF数据采集系统使用示例"""
    
    def __init__(self):
        """初始化示例系统"""
        logger.info("🚀 RLHF数据采集系统使用示例启动...")
        
        # 初始化配置和时间管理器
        self.config_manager = get_config_manager()
        self.time_manager = get_time_manager(self.config_manager)
        
        # 初始化RLHF组件
        self.mock_base_collector = MockDataCollector()
        self.rlhf_collector = RLHFDataCollector(
            self.mock_base_collector,
            self.config_manager,
            self.time_manager
        )
        
        self.expert_policy = RLHFExpertPolicy(self.config_manager)
        
        logger.info("✅ RLHF系统初始化完成")
    
    async def demonstrate_basic_usage(self):
        """演示基本使用方法"""
        logger.info("=" * 80)
        logger.info("📚 基本使用方法演示")
        logger.info("=" * 80)
        
        try:
            # 1. 开始新的训练回合
            episode_id = self.rlhf_collector.start_episode(
                scenario_type="demonstration",
                scenario_params={
                    "difficulty": "medium",
                    "missile_count": 3,
                    "duration": 600
                }
            )
            
            logger.info(f"🎬 开始训练回合: {episode_id}")
            
            # 2. 模拟数据采集循环
            for step in range(10):
                logger.info(f"📊 采集步骤 {step + 1}/10")
                
                # 获取当前状态（通过基础数据采集）
                current_time = self.time_manager.current_simulation_time
                base_data = self.mock_base_collector.collect_data_at_time(current_time)
                
                # 从基础数据提取RLHF状态
                state = self.rlhf_collector._extract_state_vector(base_data)
                
                # 使用专家策略生成动作
                expert_action = self.expert_policy.get_expert_action(state, base_data, "balanced")
                
                # 采集RLHF数据点
                data_point = self.rlhf_collector.collect_rlhf_data_point(expert_action)
                
                if data_point:
                    logger.info(f"   ✅ 数据点采集成功: 奖励={data_point.reward:.3f}")
                    
                    # 显示奖励分解
                    reward_breakdown = self.rlhf_collector.get_reward_breakdown(state, expert_action, base_data)
                    logger.info(f"   📊 奖励分解: {reward_breakdown}")
                else:
                    logger.warning(f"   ⚠️ 数据点采集失败")
                
                # 推进仿真时间
                next_time = current_time + timedelta(seconds=60)
                self.time_manager.advance_simulation_time(next_time)
            
            # 3. 结束回合
            episode = self.rlhf_collector.end_episode(success=True)
            
            logger.info(f"🏁 回合结束: 总奖励={episode.total_reward:.3f}, 数据点数={len(episode.data_points)}")
            
            # 4. 保存数据
            saved_file = self.rlhf_collector.save_rlhf_data("json")
            logger.info(f"💾 数据已保存: {saved_file}")
            
            # 5. 显示统计信息
            stats = self.rlhf_collector.get_statistics()
            logger.info("📈 采集统计:")
            basic_stats = stats.get('basic_statistics', {})
            for key, value in basic_stats.items():
                logger.info(f"   {key}: {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 基本使用演示失败: {e}")
            return False
    
    async def demonstrate_advanced_features(self):
        """演示高级功能"""
        logger.info("=" * 80)
        logger.info("🔬 高级功能演示")
        logger.info("=" * 80)
        
        try:
            # 1. 演示不同专家策略
            await self._demonstrate_expert_strategies()
            
            # 2. 演示数据质量验证
            await self._demonstrate_data_quality()
            
            # 3. 演示批量数据生成
            await self._demonstrate_batch_generation()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 高级功能演示失败: {e}")
            return False
    
    async def _demonstrate_expert_strategies(self):
        """演示不同的专家策略"""
        logger.info("🧠 演示专家策略...")
        
        # 创建测试状态
        test_state = {
            'satellite_positions': [[7000, 0, 0], [0, 7000, 0]],
            'missile_positions': [[6500, 0, 0], [0, 6500, 0]],
            'missile_threat_levels': [2, 4],  # 中等威胁和高威胁
            'visibility_matrix': [[1, 0], [0, 1]],
            'active_satellites': 2,
            'active_missiles': 2
        }
        
        test_base_data = {
            'satellites': [{'satellite_id': 'Sat01'}, {'satellite_id': 'Sat02'}],
            'missiles': [{'missile_id': 'Missile01'}, {'missile_id': 'Missile02'}],
            'visibility': [
                {'satellite_id': 'Sat01', 'missile_id': 'Missile01', 'has_visibility': True},
                {'satellite_id': 'Sat02', 'missile_id': 'Missile02', 'has_visibility': True}
            ]
        }
        
        # 测试不同策略
        strategies = self.expert_policy.get_available_strategies()
        
        for strategy in strategies:
            logger.info(f"   🎯 测试策略: {strategy}")
            
            action = self.expert_policy.get_expert_action(test_state, test_base_data, strategy)
            confidence = action['strategy_info']['confidence']
            
            # 计算该策略的奖励
            reward = self.rlhf_collector.reward_calculator.calculate_total_reward(
                test_state, action, test_base_data
            )
            
            logger.info(f"      置信度: {confidence:.3f}, 奖励: {reward:.3f}")
            
            # 显示策略特点
            sat_actions = len(action.get('satellite_actions', {}))
            mission_actions = len(action.get('mission_actions', {}).get('target_assignments', []))
            logger.info(f"      卫星动作: {sat_actions}, 任务分配: {mission_actions}")
    
    async def _demonstrate_data_quality(self):
        """演示数据质量验证"""
        logger.info("🔍 演示数据质量验证...")
        
        validator = self.rlhf_collector.data_quality_validator
        
        # 创建不同质量的测试数据
        test_cases = [
            ("正常数据", self._create_normal_data()),
            ("缺失数据", self._create_missing_data()),
            ("异常数据", self._create_anomaly_data())
        ]
        
        for case_name, data_point in test_cases:
            logger.info(f"   📊 测试 {case_name}:")
            
            validation_result = validator.validate_rlhf_data_point(data_point)
            
            logger.info(f"      有效性: {validation_result['is_valid']}")
            logger.info(f"      质量分数: {validation_result['validation_score']:.3f}")
            
            if validation_result['errors']:
                logger.info(f"      错误: {validation_result['errors']}")
            if validation_result['warnings']:
                logger.info(f"      警告: {validation_result['warnings']}")
            if validation_result['anomalies']:
                logger.info(f"      异常: {validation_result['anomalies']}")
    
    async def _demonstrate_batch_generation(self):
        """演示批量数据生成"""
        logger.info("📦 演示批量数据生成...")
        
        # 生成多个回合的数据
        episodes = []
        
        for i in range(3):
            episode_id = self.rlhf_collector.start_episode(
                scenario_type=f"batch_test_{i}",
                scenario_params={"batch_index": i}
            )
            
            # 每个回合生成5个数据点
            for step in range(5):
                current_time = self.time_manager.current_simulation_time
                base_data = self.mock_base_collector.collect_data_at_time(current_time)
                state = self.rlhf_collector._extract_state_vector(base_data)
                
                # 随机选择策略
                strategies = ["optimal_tracking", "resource_efficient", "balanced"]
                strategy = strategies[step % len(strategies)]
                
                action = self.expert_policy.get_expert_action(state, base_data, strategy)
                data_point = self.rlhf_collector.collect_rlhf_data_point(action)
                
                # 推进时间
                next_time = current_time + timedelta(seconds=30)
                self.time_manager.advance_simulation_time(next_time)
            
            episode = self.rlhf_collector.end_episode(success=True)
            episodes.append(episode)
            
            logger.info(f"   ✅ 回合 {i+1} 完成: {len(episode.data_points)} 数据点")
        
        # 保存批量数据
        batch_file = self.rlhf_collector.save_rlhf_data("json")
        logger.info(f"   💾 批量数据已保存: {batch_file}")
        
        # 显示批量统计
        total_data_points = sum(len(ep.data_points) for ep in episodes)
        avg_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
        
        logger.info(f"   📊 批量统计: {len(episodes)} 回合, {total_data_points} 数据点, 平均奖励 {avg_reward:.3f}")
    
    def _create_normal_data(self):
        """创建正常数据"""
        from src.rlhf_data_collection.rlhf_data_collector import RLHFDataPoint
        
        return RLHFDataPoint(
            timestamp=datetime.now(),
            state={
                'satellite_positions': [[7000, 0, 0]],
                'missile_positions': [[6500, 0, 0]],
                'visibility_matrix': [[1]],
                'coverage_ratio': 0.8
            },
            action={
                'satellite_actions': {'Sat01': {'payload_pointing': {'pointing_mode': 'tracking'}}},
                'mission_actions': {'target_assignments': []}
            },
            reward=0.75,
            next_state={},
            done=False,
            info={}
        )
    
    def _create_missing_data(self):
        """创建缺失数据"""
        from src.rlhf_data_collection.rlhf_data_collector import RLHFDataPoint
        
        return RLHFDataPoint(
            timestamp=datetime.now(),
            state={},  # 缺失状态数据
            action={'satellite_actions': {}},
            reward=0.0,
            next_state={},
            done=False,
            info={}
        )
    
    def _create_anomaly_data(self):
        """创建异常数据"""
        from src.rlhf_data_collection.rlhf_data_collector import RLHFDataPoint
        
        return RLHFDataPoint(
            timestamp=datetime.now(),
            state={
                'satellite_positions': [[999999, 0, 0]],  # 异常位置
                'missile_positions': [[0, 0, 0]],
                'visibility_matrix': [[1]],
                'coverage_ratio': 2.0  # 异常覆盖率
            },
            action={'satellite_actions': {}},
            reward=float('inf'),  # 异常奖励
            next_state={},
            done=False,
            info={}
        )

class MockDataCollector:
    """模拟数据采集器"""
    
    def __init__(self):
        self.constellation_manager = MockConstellationManager()
        self.step_count = 0
    
    def collect_data_at_time(self, current_time):
        """模拟数据采集"""
        self.step_count += 1
        
        # 模拟动态数据
        satellite_x = 7000 + self.step_count * 100
        missile_x = 6500 - self.step_count * 50
        
        return {
            'collection_time': current_time.isoformat(),
            'satellites': [
                {
                    'satellite_id': 'Satellite01',
                    'position': {'x': satellite_x, 'y': 0, 'z': 0},
                    'velocity': {'vx': 0, 'vy': 7.5, 'vz': 0},
                    'payload_status': {'operational': True, 'power_consumption': 80}
                }
            ],
            'missiles': [
                {
                    'missile_id': 'Missile01',
                    'position': {'x': missile_x, 'y': 0, 'z': 0},
                    'velocity': {'vx': 2, 'vy': 0, 'vz': 0},
                    'threat_level': 'medium'
                }
            ],
            'visibility': [
                {
                    'satellite_id': 'Satellite01',
                    'missile_id': 'Missile01',
                    'has_visibility': True
                }
            ],
            'simulation_progress': min(1.0, self.step_count * 0.1)
        }

class MockConstellationManager:
    """模拟星座管理器"""
    
    def get_constellation_info(self):
        return {
            'type': 'Walker',
            'total_satellites': 1,
            'satellite_list': ['Satellite01']
        }

async def main():
    """主函数"""
    try:
        logger.info("🚀 RLHF数据采集系统使用示例启动")
        
        # 创建示例系统
        example = RLHFUsageExample()
        
        # 演示基本使用
        basic_success = await example.demonstrate_basic_usage()
        
        if basic_success:
            # 演示高级功能
            advanced_success = await example.demonstrate_advanced_features()
            
            if advanced_success:
                logger.info("✅ 所有演示完成")
                
                # 显示使用总结
                logger.info("=" * 80)
                logger.info("📖 使用总结")
                logger.info("=" * 80)
                logger.info("1. 基本流程: start_episode -> collect_data_point -> end_episode")
                logger.info("2. 专家策略: 使用不同策略生成高质量动作")
                logger.info("3. 数据质量: 自动验证和质量控制")
                logger.info("4. 多种格式: 支持JSON、HDF5、NumPy格式输出")
                logger.info("5. 统计监控: 实时统计和性能监控")
                logger.info("=" * 80)
            else:
                logger.error("❌ 高级功能演示失败")
        else:
            logger.error("❌ 基本使用演示失败")
            
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断演示")
    except Exception as e:
        logger.error(f"❌ 演示异常: {e}")
    finally:
        logger.info("🏁 RLHF数据采集系统使用示例结束")

if __name__ == "__main__":
    asyncio.run(main())
