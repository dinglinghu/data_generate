#!/usr/bin/env python3
"""
RLHF数据采集系统使用示例
演示如何使用RLHF数据采集系统生成强化学习训练数据
"""

import asyncio
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rlhf_data_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 导入项目模块
from main import STKDataCollectionSystem
from src.rlhf_data_collection.rlhf_system import RLHFDataCollectionSystem

class RLHFDataCollectionExample:
    """RLHF数据采集示例类"""
    
    def __init__(self):
        """初始化示例系统"""
        logger.info("🚀 RLHF数据采集示例系统启动...")
        
        # 初始化基础STK系统
        self.base_system = STKDataCollectionSystem()
        
        # 初始化RLHF系统
        self.rlhf_system = None
        
    async def run_example(self):
        """运行完整的RLHF数据采集示例"""
        try:
            logger.info("=" * 80)
            logger.info("🤖 RLHF数据采集系统示例开始")
            logger.info("=" * 80)
            
            # 1. 初始化基础系统
            if not await self._initialize_base_system():
                logger.error("❌ 基础系统初始化失败")
                return False
            
            # 2. 初始化RLHF系统
            if not self._initialize_rlhf_system():
                logger.error("❌ RLHF系统初始化失败")
                return False
            
            # 3. 生成小规模训练数据集
            await self._generate_small_training_dataset()
            
            # 4. 生成评估数据集
            await self._generate_evaluation_dataset()
            
            # 5. 展示数据采集统计
            self._show_statistics()
            
            # 6. 演示数据格式
            self._demonstrate_data_formats()
            
            logger.info("✅ RLHF数据采集示例完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ RLHF数据采集示例异常: {e}")
            return False
        finally:
            self._cleanup()
    
    async def _initialize_base_system(self) -> bool:
        """初始化基础STK系统"""
        try:
            logger.info("🔧 初始化基础STK系统...")
            
            # 连接STK
            if not await self.base_system._connect_stk():
                logger.error("❌ STK连接失败")
                return False
            
            # 初始化组件
            if not self.base_system._initialize_components():
                logger.error("❌ 组件初始化失败")
                return False
            
            # 设置星座
            if not await self.base_system._setup_constellation():
                logger.error("❌ 星座设置失败")
                return False
            
            logger.info("✅ 基础STK系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 基础系统初始化异常: {e}")
            return False
    
    def _initialize_rlhf_system(self) -> bool:
        """初始化RLHF系统"""
        try:
            logger.info("🤖 初始化RLHF系统...")
            
            self.rlhf_system = RLHFDataCollectionSystem(self.base_system)
            
            logger.info("✅ RLHF系统初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ RLHF系统初始化异常: {e}")
            return False
    
    async def _generate_small_training_dataset(self):
        """生成小规模训练数据集"""
        try:
            logger.info("📊 开始生成小规模训练数据集...")
            
            # 配置小规模数据集参数
            num_scenarios = 10  # 小规模示例
            difficulty_distribution = {
                "easy": 0.4,
                "medium": 0.4,
                "hard": 0.2
            }
            
            # 生成训练数据集
            training_stats = await self.rlhf_system.generate_training_dataset(
                num_scenarios=num_scenarios,
                difficulty_distribution=difficulty_distribution
            )
            
            logger.info("📊 训练数据集生成完成")
            logger.info(f"   总场景数: {training_stats['total_scenarios']}")
            logger.info(f"   成功回合: {training_stats['successful_episodes']}")
            logger.info(f"   失败回合: {training_stats['failed_episodes']}")
            logger.info(f"   成功率: {training_stats['success_rate']:.2%}")
            logger.info(f"   总数据点: {training_stats['total_data_points']}")
            logger.info(f"   数据文件: {training_stats['dataset_file']}")
            
        except Exception as e:
            logger.error(f"❌ 训练数据集生成失败: {e}")
    
    async def _generate_evaluation_dataset(self):
        """生成评估数据集"""
        try:
            logger.info("📋 开始生成评估数据集...")
            
            # 生成评估数据集
            eval_stats = await self.rlhf_system.generate_evaluation_dataset(
                num_scenarios=5  # 小规模示例
            )
            
            logger.info("📋 评估数据集生成完成")
            logger.info(f"   总场景数: {eval_stats['total_scenarios']}")
            logger.info(f"   采集回合: {eval_stats['collected_episodes']}")
            logger.info(f"   评估文件: {eval_stats['evaluation_file']}")
            
            # 显示评估指标
            metrics = eval_stats.get('evaluation_metrics', {})
            if metrics:
                logger.info("📊 评估指标:")
                logger.info(f"   成功率: {metrics.get('success_rate', 0):.2%}")
                logger.info(f"   平均奖励: {metrics.get('average_reward', 0):.3f}")
                logger.info(f"   平均回合长度: {metrics.get('average_episode_length', 0):.1f}")
                logger.info(f"   奖励标准差: {metrics.get('reward_std', 0):.3f}")
            
        except Exception as e:
            logger.error(f"❌ 评估数据集生成失败: {e}")
    
    def _show_statistics(self):
        """显示数据采集统计"""
        try:
            logger.info("📈 数据采集统计信息:")
            
            # 获取系统统计
            stats = self.rlhf_system.get_system_statistics()
            
            # RLHF采集器统计
            rlhf_stats = stats.get('rlhf_collector', {})
            logger.info("🤖 RLHF采集器统计:")
            logger.info(f"   总回合数: {rlhf_stats.get('total_episodes', 0)}")
            logger.info(f"   成功回合: {rlhf_stats.get('successful_episodes', 0)}")
            logger.info(f"   成功率: {rlhf_stats.get('success_rate', 0):.2%}")
            logger.info(f"   总数据点: {rlhf_stats.get('total_data_points', 0)}")
            logger.info(f"   平均奖励: {rlhf_stats.get('average_reward', 0):.3f}")
            
            # 场景生成器统计
            scenario_stats = stats.get('scenario_generator', {})
            logger.info("🎭 场景生成器统计:")
            logger.info(f"   总场景数: {scenario_stats.get('total_scenarios', 0)}")
            
            scenario_types = scenario_stats.get('scenario_types', {})
            if scenario_types:
                logger.info("   场景类型分布:")
                for scenario_type, count in scenario_types.items():
                    logger.info(f"     {scenario_type}: {count}")
            
            difficulty_dist = scenario_stats.get('difficulty_distribution', {})
            if difficulty_dist:
                logger.info("   难度分布:")
                for difficulty, count in difficulty_dist.items():
                    logger.info(f"     {difficulty}: {count}")
            
        except Exception as e:
            logger.error(f"❌ 统计信息显示失败: {e}")
    
    def _demonstrate_data_formats(self):
        """演示数据格式"""
        try:
            logger.info("📄 数据格式演示:")
            
            # 检查输出目录
            output_dir = Path("output/rlhf_data")
            if output_dir.exists():
                files = list(output_dir.glob("*.json"))
                
                if files:
                    # 读取第一个JSON文件作为示例
                    sample_file = files[0]
                    logger.info(f"📁 示例数据文件: {sample_file}")
                    
                    try:
                        import json
                        with open(sample_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 显示数据结构
                        logger.info("📊 数据结构:")
                        logger.info(f"   元数据键: {list(data.get('metadata', {}).keys())}")
                        
                        episodes = data.get('episodes', [])
                        if episodes:
                            sample_episode = episodes[0]
                            logger.info(f"   回合键: {list(sample_episode.keys())}")
                            
                            data_points = sample_episode.get('data_points', [])
                            if data_points:
                                sample_point = data_points[0]
                                logger.info(f"   数据点键: {list(sample_point.keys())}")
                                
                                # 显示状态空间结构
                                state = sample_point.get('state', {})
                                logger.info(f"   状态空间键: {list(state.keys())}")
                                
                                # 显示动作空间结构
                                action = sample_point.get('action', {})
                                logger.info(f"   动作空间键: {list(action.keys())}")
                        
                        logger.info(f"✅ 数据格式演示完成")
                        
                    except Exception as e:
                        logger.error(f"❌ 读取示例文件失败: {e}")
                else:
                    logger.warning("⚠️ 没有找到数据文件")
            else:
                logger.warning("⚠️ 输出目录不存在")
                
        except Exception as e:
            logger.error(f"❌ 数据格式演示失败: {e}")
    
    def _cleanup(self):
        """清理资源"""
        try:
            logger.info("🧹 清理系统资源...")
            
            if self.base_system:
                self.base_system._cleanup()
            
            logger.info("✅ 资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理失败: {e}")

async def main():
    """主函数"""
    try:
        logger.info("=" * 80)
        logger.info("🚀 RLHF数据采集系统示例启动")
        logger.info("=" * 80)
        
        # 创建示例系统
        example_system = RLHFDataCollectionExample()
        
        # 运行示例
        success = await example_system.run_example()
        
        if success:
            logger.info("✅ RLHF数据采集示例运行完成")
            
            # 显示使用说明
            logger.info("=" * 80)
            logger.info("📖 使用说明:")
            logger.info("1. 生成的训练数据可用于强化学习模型训练")
            logger.info("2. 数据格式支持JSON、HDF5、NumPy等多种格式")
            logger.info("3. 可以通过配置文件调整场景参数和数据采集策略")
            logger.info("4. 支持自定义奖励函数和状态空间编码")
            logger.info("5. 数据文件保存在 output/rlhf_data/ 目录下")
            logger.info("=" * 80)
            
        else:
            logger.error("❌ RLHF数据采集示例运行失败")
            
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断程序")
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
    finally:
        logger.info("=" * 80)
        logger.info("🏁 RLHF数据采集系统示例结束")
        logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
