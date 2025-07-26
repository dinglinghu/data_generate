#!/usr/bin/env python3
"""
测试修改后的系统 - 验证系统只进行一次仿真场景时间设置
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config_manager import ConfigManager
from src.stk_interface.stk_manager import STKManager
from src.data_collection.data_collector import DataCollector
from src.constellation.constellation_manager import ConstellationManager
from src.stk_interface.missile_manager import MissileManager
from src.stk_interface.visibility_calculator import VisibilityCalculator
from src.utils.time_manager import UnifiedTimeManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_fixed_scenario_time():
    """测试固定场景时间设置"""
    logger.info("=== 测试固定场景时间设置 ===")
    
    try:
        # 初始化管理器
        config_manager = ConfigManager()
        time_manager = UnifiedTimeManager(config_manager)
        
        # STKManager需要STK配置
        stk_config = config_manager.get_stk_config()
        stk_manager = STKManager(stk_config)
        
        # 连接STK
        logger.info("连接STK...")
        if not stk_manager.connect():
            logger.error("STK连接失败")
            return False
        
        # 记录初始场景时间设置
        scenario = stk_manager.scenario
        initial_start_time = scenario.StartTime
        initial_stop_time = scenario.StopTime
        
        logger.info(f"✅ 初始场景时间设置:")
        logger.info(f"   开始时间: {initial_start_time}")
        logger.info(f"   结束时间: {initial_stop_time}")
        
        # 创建其他组件
        class MockOutputManager:
            def __init__(self, config_manager):
                self.config_manager = config_manager
        
        output_manager = MockOutputManager(config_manager)

        # 创建完整配置字典
        full_config = {
            "constellation": config_manager.get_constellation_config(),
            "payload": config_manager.get_payload_config(),
            "simulation": config_manager.get_simulation_config(),
            "stk": config_manager.get_stk_config(),
            "missile": config_manager.get_missile_config()
        }

        missile_manager = MissileManager(stk_manager, full_config, output_manager)
        constellation_manager = ConstellationManager(stk_manager, config_manager)
        visibility_calculator = VisibilityCalculator(stk_manager)
        data_collector = DataCollector(
            stk_manager, 
            constellation_manager, 
            missile_manager, 
            visibility_calculator, 
            time_manager, 
            config_manager
        )
        
        # 创建简单的星座
        logger.info("创建测试星座...")
        constellation_manager.create_walker_constellation()
        
        # 创建测试导弹
        logger.info("创建测试导弹...")
        test_missile_config = {
            "missile_id": "FixedTimeTest_01",
            "launch_position": {"lat": 39.9042, "lon": 116.4074, "alt": 0.0},
            "target_position": {"lat": 31.2304, "lon": 121.4737, "alt": 0.0},
            "apogee_alt": 500.0,
            "launch_sequence": 1,
            "launch_time": datetime.strptime("2025-07-25 04:30:00", "%Y-%m-%d %H:%M:%S")
        }
        
        missile_manager.create_single_missile_target(test_missile_config)
        
        # 模拟多次数据采集，验证场景时间不会被重复设置
        logger.info(f"\n=== 开始模拟数据采集循环 ===")
        
        collection_times = [
            datetime.strptime("2025-07-25 04:15:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2025-07-25 04:30:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2025-07-25 04:45:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2025-07-25 05:00:00", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2025-07-25 05:15:00", "%Y-%m-%d %H:%M:%S")
        ]
        
        for i, collection_time in enumerate(collection_times, 1):
            logger.info(f"\n🔄 第{i}次数据采集: {collection_time}")
            
            # 记录采集前的场景时间
            before_start = scenario.StartTime
            before_stop = scenario.StopTime
            
            # 执行数据采集
            data_snapshot = data_collector.collect_data_at_time(collection_time)
            
            # 记录采集后的场景时间
            after_start = scenario.StartTime
            after_stop = scenario.StopTime
            
            # 验证场景时间是否保持不变
            time_unchanged = (before_start == after_start and before_stop == after_stop)
            
            if time_unchanged:
                logger.info(f"   ✅ 场景时间保持不变: {before_start} - {before_stop}")
            else:
                logger.warning(f"   ⚠️ 场景时间发生变化!")
                logger.warning(f"      变化前: {before_start} - {before_stop}")
                logger.warning(f"      变化后: {after_start} - {after_stop}")
            
            # 检查数据采集结果
            if data_snapshot:
                metadata = data_snapshot.get("metadata", {})
                scenario_time_fixed = metadata.get("scenario_time_fixed", False)
                
                logger.info(f"   📊 数据采集成功:")
                logger.info(f"      卫星数量: {len(data_snapshot.get('satellites', []))}")
                logger.info(f"      导弹数量: {len(data_snapshot.get('missiles', []))}")
                logger.info(f"      可见性记录: {len(data_snapshot.get('visibility', []))}")
                logger.info(f"      固定场景时间标记: {scenario_time_fixed}")
                
                if not scenario_time_fixed:
                    logger.warning(f"   ⚠️ 固定场景时间标记为False")
            else:
                logger.error(f"   ❌ 数据采集失败")
        
        # 最终验证
        final_start = scenario.StartTime
        final_stop = scenario.StopTime
        
        logger.info(f"\n=== 最终验证结果 ===")
        logger.info(f"初始场景时间: {initial_start_time} - {initial_stop_time}")
        logger.info(f"最终场景时间: {final_start} - {final_stop}")
        
        if initial_start_time == final_start and initial_stop_time == final_stop:
            logger.info(f"🎉 验证成功: 场景时间在整个数据采集过程中保持不变")
            return True
        else:
            logger.error(f"❌ 验证失败: 场景时间发生了变化")
            return False
        
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
        return False
    
    finally:
        # 清理资源
        try:
            if 'stk_manager' in locals():
                stk_manager.disconnect()
        except:
            pass

def main():
    """主函数"""
    logger.info("🔧 开始测试固定场景时间设置")
    
    success = test_fixed_scenario_time()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("🎉 固定场景时间设置测试成功！")
        logger.info("✅ 系统已正确修改为只进行一次场景时间设置")
    else:
        logger.error("❌ 固定场景时间设置测试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
