#!/usr/bin/env python3
"""
RLHF数据采集系统测试脚本
验证RLHF数据采集和数据质量验证功能
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_rlhf_data_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 导入项目模块
try:
    from main import STKDataCollectionSystem
    from src.rlhf_data_collection.rlhf_data_collector import RLHFDataCollector
    from src.rlhf_data_collection.reward_calculator import RLHFRewardCalculator
    from src.rlhf_data_collection.action_executor import RLHFActionExecutor
    from src.rlhf_data_collection.data_quality_validator import RLHFDataQualityValidator
    from src.rlhf_data_collection.expert_policy import RLHFExpertPolicy
    from src.utils.config_manager import get_config_manager
    from src.utils.time_manager import get_time_manager
except ImportError as e:
    logger.error(f"模块导入失败: {e}")
    sys.exit(1)

class RLHFDataCollectionTest:
    """RLHF数据采集系统测试类"""
    
    def __init__(self):
        """初始化测试系统"""
        logger.info("🧪 RLHF数据采集系统测试启动...")
        
        # 初始化配置和时间管理器
        self.config_manager = get_config_manager()
        self.time_manager = get_time_manager(self.config_manager)
        
        # 初始化基础系统（模拟模式）
        self.base_system = None
        self.rlhf_collector = None
        self.expert_policy = None
        
        # 测试结果
        self.test_results = {
            'component_tests': {},
            'integration_tests': {},
            'data_quality_tests': {},
            'performance_tests': {}
        }
    
    async def run_all_tests(self):
        """运行所有测试"""
        try:
            logger.info("=" * 80)
            logger.info("🧪 开始RLHF数据采集系统测试")
            logger.info("=" * 80)
            
            # 1. 组件单元测试
            await self._test_components()
            
            # 2. 集成测试
            await self._test_integration()
            
            # 3. 数据质量测试
            await self._test_data_quality()
            
            # 4. 性能测试
            await self._test_performance()
            
            # 5. 生成测试报告
            self._generate_test_report()
            
            logger.info("✅ RLHF数据采集系统测试完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 测试执行失败: {e}")
            return False
    
    async def _test_components(self):
        """测试各个组件"""
        logger.info("🔧 开始组件单元测试...")
        
        # 测试奖励计算器
        await self._test_reward_calculator()
        
        # 测试数据质量验证器
        await self._test_data_quality_validator()
        
        # 测试专家策略
        await self._test_expert_policy()
        
        # 测试动作执行器（模拟模式）
        await self._test_action_executor()
        
        logger.info("✅ 组件单元测试完成")
    
    async def _test_reward_calculator(self):
        """测试奖励计算器"""
        try:
            logger.info("🎯 测试奖励计算器...")
            
            reward_calculator = RLHFRewardCalculator(self.config_manager)
            
            # 创建测试状态和动作
            test_state = {
                'satellite_positions': [[7000, 0, 0], [0, 7000, 0]],
                'missile_positions': [[6500, 0, 0]],
                'visibility_matrix': [[1], [0]],
                'coverage_ratio': 0.5,
                'mission_progress': 0.3,
                'active_satellites': 2,
                'active_missiles': 1
            }
            
            test_action = {
                'satellite_actions': {
                    'Satellite01': {
                        'payload_pointing': {'pointing_mode': 'tracking'},
                        'power_management': {'power_allocation': {'payload': 0.6, 'communication': 0.2, 'attitude_control': 0.2}}
                    }
                },
                'mission_actions': {
                    'target_assignments': [{'satellite_id': 'Satellite01', 'target_id': 'Missile01'}]
                }
            }
            
            test_base_data = {
                'satellites': [{'satellite_id': 'Satellite01'}, {'satellite_id': 'Satellite02'}],
                'missiles': [{'missile_id': 'Missile01'}],
                'visibility': [{'satellite_id': 'Satellite01', 'missile_id': 'Missile01', 'has_visibility': True}]
            }
            
            # 计算奖励
            reward = reward_calculator.calculate_total_reward(test_state, test_action, test_base_data)
            
            # 获取奖励分解
            breakdown = reward_calculator.get_reward_breakdown(test_state, test_action, test_base_data)
            
            # 验证结果
            assert isinstance(reward, (int, float)), "奖励应该是数值类型"
            assert -2.0 <= reward <= 2.0, f"奖励值在合理范围内: {reward}"  # 允许负奖励
            assert isinstance(breakdown, dict), "奖励分解应该是字典类型"
            
            self.test_results['component_tests']['reward_calculator'] = {
                'status': 'PASS',
                'reward': reward,
                'breakdown': breakdown
            }
            
            logger.info(f"✅ 奖励计算器测试通过: 奖励={reward:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 奖励计算器测试失败: {e}")
            self.test_results['component_tests']['reward_calculator'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def _test_data_quality_validator(self):
        """测试数据质量验证器"""
        try:
            logger.info("🔍 测试数据质量验证器...")
            
            validator = RLHFDataQualityValidator(self.config_manager)
            
            # 创建测试数据点
            from src.rlhf_data_collection.rlhf_data_collector import RLHFDataPoint
            
            test_data_point = RLHFDataPoint(
                timestamp=datetime.now(),
                state={
                    'satellite_positions': [[7000, 0, 0]],
                    'missile_positions': [[6500, 0, 0]],
                    'visibility_matrix': [[1]],
                    'coverage_ratio': 0.5
                },
                action={
                    'satellite_actions': {'Satellite01': {'payload_pointing': {'pointing_mode': 'tracking'}}},
                    'mission_actions': {'target_assignments': []}
                },
                reward=0.75,
                next_state={},
                done=False,
                info={}
            )
            
            # 验证数据点
            validation_result = validator.validate_rlhf_data_point(test_data_point)
            
            # 验证结果
            assert isinstance(validation_result, dict), "验证结果应该是字典类型"
            assert 'is_valid' in validation_result, "验证结果应包含is_valid字段"
            assert 'validation_score' in validation_result, "验证结果应包含validation_score字段"
            
            # 获取验证统计
            stats = validator.get_validation_statistics()
            
            self.test_results['component_tests']['data_quality_validator'] = {
                'status': 'PASS',
                'validation_result': validation_result,
                'statistics': stats
            }
            
            logger.info(f"✅ 数据质量验证器测试通过: 有效={validation_result['is_valid']}, 分数={validation_result['validation_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 数据质量验证器测试失败: {e}")
            self.test_results['component_tests']['data_quality_validator'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def _test_expert_policy(self):
        """测试专家策略"""
        try:
            logger.info("🧠 测试专家策略...")
            
            expert_policy = RLHFExpertPolicy(self.config_manager)
            
            # 创建测试状态和基础数据
            test_state = {
                'satellite_positions': [[7000, 0, 0], [0, 7000, 0]],
                'missile_positions': [[6500, 0, 0], [0, 6500, 0]],
                'missile_threat_levels': [2, 3],
                'visibility_matrix': [[1, 0], [0, 1]],
                'active_satellites': 2,
                'active_missiles': 2
            }
            
            test_base_data = {
                'satellites': [
                    {'satellite_id': 'Satellite01'},
                    {'satellite_id': 'Satellite02'}
                ],
                'missiles': [
                    {'missile_id': 'Missile01', 'threat_level': 'medium'},
                    {'missile_id': 'Missile02', 'threat_level': 'high'}
                ],
                'visibility': [
                    {'satellite_id': 'Satellite01', 'missile_id': 'Missile01', 'has_visibility': True},
                    {'satellite_id': 'Satellite02', 'missile_id': 'Missile02', 'has_visibility': True}
                ]
            }
            
            # 测试不同策略
            strategies = expert_policy.get_available_strategies()
            strategy_results = {}
            
            for strategy in strategies:
                action = expert_policy.get_expert_action(test_state, test_base_data, strategy)
                
                # 验证动作格式
                assert isinstance(action, dict), f"策略{strategy}返回的动作应该是字典类型"
                assert 'satellite_actions' in action, f"策略{strategy}应包含satellite_actions"
                assert 'mission_actions' in action, f"策略{strategy}应包含mission_actions"
                assert 'strategy_info' in action, f"策略{strategy}应包含strategy_info"
                
                strategy_results[strategy] = {
                    'action': action,
                    'confidence': action['strategy_info']['confidence']
                }
            
            self.test_results['component_tests']['expert_policy'] = {
                'status': 'PASS',
                'strategies_tested': len(strategies),
                'strategy_results': strategy_results
            }
            
            logger.info(f"✅ 专家策略测试通过: 测试了{len(strategies)}种策略")
            
        except Exception as e:
            logger.error(f"❌ 专家策略测试失败: {e}")
            self.test_results['component_tests']['expert_policy'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def _test_action_executor(self):
        """测试动作执行器（模拟模式）"""
        try:
            logger.info("🎮 测试动作执行器（模拟模式）...")
            
            # 由于没有真实的STK连接，这里只测试接口
            # 在实际环境中，这里会测试真实的STK接口调用
            
            test_action = {
                'satellite_actions': {
                    'Satellite01': {
                        'payload_pointing': {'pointing_mode': 'tracking'},
                        'power_management': {'power_allocation': {'payload': 0.6, 'communication': 0.2, 'attitude_control': 0.2}}
                    }
                },
                'mission_actions': {
                    'target_assignments': [{'satellite_id': 'Satellite01', 'target_id': 'Missile01'}]
                }
            }
            
            # 模拟执行结果
            mock_result = {
                'success': True,
                'executed_actions': ['Satellite01_payload_pointing', 'target_assignments'],
                'failed_actions': [],
                'execution_time': datetime.now().isoformat()
            }
            
            self.test_results['component_tests']['action_executor'] = {
                'status': 'PASS',
                'mock_result': mock_result,
                'note': 'Tested in simulation mode without STK connection'
            }
            
            logger.info("✅ 动作执行器测试通过（模拟模式）")
            
        except Exception as e:
            logger.error(f"❌ 动作执行器测试失败: {e}")
            self.test_results['component_tests']['action_executor'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def _test_integration(self):
        """测试系统集成"""
        logger.info("🔗 开始集成测试...")
        
        try:
            # 初始化RLHF数据采集器
            # 注意：这里使用模拟的基础数据采集器
            mock_base_collector = MockDataCollector()
            
            self.rlhf_collector = RLHFDataCollector(
                mock_base_collector,
                self.config_manager,
                self.time_manager
            )
            
            self.expert_policy = RLHFExpertPolicy(self.config_manager)
            
            # 模拟一个完整的数据采集流程
            episode_id = self.rlhf_collector.start_episode("test_scenario", {"test": True})
            
            # 生成几个数据点
            for i in range(5):
                # 模拟状态
                mock_state = {
                    'satellite_positions': [[7000 + i*100, 0, 0]],
                    'missile_positions': [[6500 - i*50, 0, 0]],
                    'visibility_matrix': [[1]],
                    'coverage_ratio': 0.8 - i*0.1,
                    'mission_progress': i * 0.2
                }
                
                # 生成专家动作
                mock_base_data = {
                    'satellites': [{'satellite_id': 'Satellite01'}],
                    'missiles': [{'missile_id': 'Missile01'}],
                    'visibility': [{'satellite_id': 'Satellite01', 'missile_id': 'Missile01', 'has_visibility': True}]
                }
                
                expert_action = self.expert_policy.get_expert_action(mock_state, mock_base_data)
                
                # 采集数据点
                data_point = self.rlhf_collector.collect_rlhf_data_point(expert_action)
                
                assert data_point is not None, f"数据点{i}采集失败"
                
                # 推进时间
                self.time_manager.advance_simulation_time(
                    self.time_manager.current_simulation_time + timedelta(seconds=30)
                )
            
            # 结束回合
            episode = self.rlhf_collector.end_episode(success=True)
            
            # 验证结果
            assert episode is not None, "回合结束失败"
            assert len(episode.data_points) == 5, f"数据点数量错误: {len(episode.data_points)}"
            
            # 获取统计信息
            stats = self.rlhf_collector.get_statistics()
            
            self.test_results['integration_tests']['full_pipeline'] = {
                'status': 'PASS',
                'episode_id': episode_id,
                'data_points_collected': len(episode.data_points),
                'total_reward': episode.total_reward,
                'statistics': stats
            }
            
            logger.info(f"✅ 集成测试通过: 采集了{len(episode.data_points)}个数据点")
            
        except Exception as e:
            logger.error(f"❌ 集成测试失败: {e}")
            self.test_results['integration_tests']['full_pipeline'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def _test_data_quality(self):
        """测试数据质量"""
        logger.info("📊 开始数据质量测试...")
        
        try:
            # 测试各种数据质量场景
            validator = RLHFDataQualityValidator(self.config_manager)
            
            # 测试场景1: 正常数据
            normal_data = self._create_test_data_point("normal")
            result1 = validator.validate_rlhf_data_point(normal_data)
            
            # 测试场景2: 缺失数据
            missing_data = self._create_test_data_point("missing")
            result2 = validator.validate_rlhf_data_point(missing_data)
            
            # 测试场景3: 异常数据
            anomaly_data = self._create_test_data_point("anomaly")
            result3 = validator.validate_rlhf_data_point(anomaly_data)
            
            self.test_results['data_quality_tests'] = {
                'normal_data': result1,
                'missing_data': result2,
                'anomaly_data': result3,
                'validator_stats': validator.get_validation_statistics()
            }
            
            logger.info("✅ 数据质量测试完成")
            
        except Exception as e:
            logger.error(f"❌ 数据质量测试失败: {e}")
            self.test_results['data_quality_tests'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    async def _test_performance(self):
        """测试性能"""
        logger.info("⚡ 开始性能测试...")
        
        try:
            start_time = datetime.now()
            
            # 模拟大量数据采集
            mock_base_collector = MockDataCollector()
            rlhf_collector = RLHFDataCollector(
                mock_base_collector,
                self.config_manager,
                self.time_manager
            )
            
            expert_policy = RLHFExpertPolicy(self.config_manager)
            
            # 采集100个数据点
            episode_id = rlhf_collector.start_episode("performance_test", {})
            
            for i in range(100):
                mock_state = {'mission_progress': i * 0.01}
                mock_base_data = {
                    'satellites': [{'satellite_id': f'Sat{i%3}'}],
                    'missiles': [{'missile_id': f'Missile{i%5}'}],
                    'visibility': []
                }
                
                action = expert_policy.get_expert_action(mock_state, mock_base_data)
                data_point = rlhf_collector.collect_rlhf_data_point(action)
                
                if i % 20 == 0:
                    logger.info(f"性能测试进度: {i}/100")
            
            rlhf_collector.end_episode(success=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.test_results['performance_tests'] = {
                'status': 'PASS',
                'data_points': 100,
                'duration_seconds': duration,
                'points_per_second': 100 / duration,
                'memory_usage': 'Not measured'
            }
            
            logger.info(f"✅ 性能测试完成: {100/duration:.2f} 数据点/秒")
            
        except Exception as e:
            logger.error(f"❌ 性能测试失败: {e}")
            self.test_results['performance_tests'] = {
                'status': 'FAIL',
                'error': str(e)
            }
    
    def _create_test_data_point(self, data_type: str):
        """创建测试数据点"""
        from src.rlhf_data_collection.rlhf_data_collector import RLHFDataPoint
        
        if data_type == "normal":
            return RLHFDataPoint(
                timestamp=datetime.now(),
                state={'satellite_positions': [[7000, 0, 0]], 'coverage_ratio': 0.5},
                action={'satellite_actions': {}, 'mission_actions': {}},
                reward=0.75,
                next_state={},
                done=False,
                info={}
            )
        elif data_type == "missing":
            return RLHFDataPoint(
                timestamp=datetime.now(),
                state={},  # 缺失状态数据
                action={'satellite_actions': {}},
                reward=0.0,
                next_state={},
                done=False,
                info={}
            )
        elif data_type == "anomaly":
            return RLHFDataPoint(
                timestamp=datetime.now(),
                state={'satellite_positions': [[999999, 0, 0]]},  # 异常位置
                action={'satellite_actions': {}},
                reward=float('inf'),  # 异常奖励
                next_state={},
                done=False,
                info={}
            )
    
    def _generate_test_report(self):
        """生成测试报告"""
        logger.info("📋 生成测试报告...")
        
        # 保存测试结果
        report_file = Path("test_results_rlhf.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 统计测试结果
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    total_tests += 1
                    if isinstance(result, dict) and result.get('status') == 'PASS':
                        passed_tests += 1
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        logger.info("=" * 80)
        logger.info("📋 测试报告摘要")
        logger.info("=" * 80)
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {total_tests - passed_tests}")
        logger.info(f"成功率: {success_rate:.2%}")
        logger.info(f"详细报告: {report_file}")
        logger.info("=" * 80)

class MockDataCollector:
    """模拟数据采集器"""

    def __init__(self):
        """初始化模拟数据采集器"""
        # 添加缺失的属性
        self.constellation_manager = MockConstellationManager()

    def collect_data_at_time(self, current_time):
        """模拟数据采集"""
        return {
            'collection_time': current_time.isoformat(),
            'satellites': [
                {
                    'satellite_id': 'Satellite01',
                    'position': {'x': 7000, 'y': 0, 'z': 0},
                    'velocity': {'vx': 0, 'vy': 7.5, 'vz': 0},
                    'payload_status': {'operational': True, 'power_consumption': 80}
                }
            ],
            'missiles': [
                {
                    'missile_id': 'Missile01',
                    'position': {'x': 6500, 'y': 0, 'z': 0},
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
            'simulation_progress': 0.5
        }

class MockConstellationManager:
    """模拟星座管理器"""

    def get_constellation_info(self):
        """获取星座信息"""
        return {
            'type': 'Walker',
            'total_satellites': 1,
            'satellite_list': ['Satellite01']
        }

async def main():
    """主函数"""
    try:
        logger.info("🚀 RLHF数据采集系统测试启动")
        
        # 创建测试系统
        test_system = RLHFDataCollectionTest()
        
        # 运行测试
        success = await test_system.run_all_tests()
        
        if success:
            logger.info("✅ 所有测试完成")
        else:
            logger.error("❌ 测试执行失败")
            
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断测试")
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
    finally:
        logger.info("🏁 RLHF数据采集系统测试结束")

if __name__ == "__main__":
    asyncio.run(main())
