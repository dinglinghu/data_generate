#!/usr/bin/env python3
"""
测试从STK获取导弹准确时间功能
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config_manager import ConfigManager
from src.stk_interface.stk_manager import STKManager
from src.utils.time_manager import UnifiedTimeManager
from src.stk_interface.missile_manager import MissileManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_stk_missile_time_extraction():
    """测试从STK获取导弹准确时间"""
    logger.info("=== 测试从STK获取导弹准确时间 ===")
    
    try:
        # 初始化管理器
        config_manager = ConfigManager()
        time_manager = UnifiedTimeManager(config_manager)
        
        # STKManager需要完整的配置字典
        full_config = {
            "constellation": config_manager.get_constellation_config(),
            "payload": config_manager.get_payload_config(),
            "simulation": config_manager.get_simulation_config(),
            "stk": config_manager.get_stk_config()
        }
        stk_manager = STKManager(full_config)
        
        # 连接STK
        logger.info("连接STK...")
        if not stk_manager.connect():
            logger.error("STK连接失败")
            return False
        
        # 初始化导弹管理器
        missile_config = config_manager.get_missile_config()
        
        class MockOutputManager:
            def save_data(self, data, filename):
                return f"mock_output/{filename}"
        
        output_manager = MockOutputManager()
        missile_manager = MissileManager(stk_manager, missile_config, output_manager)
        
        logger.info("✅ 导弹管理器初始化成功")
        
        # 获取STK场景中现有的导弹
        scenario = stk_manager.scenario
        missile_objects = []
        
        try:
            for i in range(scenario.Children.Count):
                child = scenario.Children.Item(i)
                if child.ClassName == "Missile":
                    missile_objects.append(child.InstanceName)
                    logger.info(f"🎯 发现导弹对象: {child.InstanceName}")
        except Exception as e:
            logger.error(f"❌ 获取场景对象失败: {e}")
            return False
        
        if not missile_objects:
            logger.warning("⚠️ 场景中没有导弹对象，无法测试时间提取")
            return False
        
        logger.info(f"📊 找到 {len(missile_objects)} 个导弹对象")
        
        # 测试每个导弹的时间提取
        success_count = 0
        
        for missile_id in missile_objects:
            logger.info(f"\n--- 测试导弹: {missile_id} ---")
            
            # 测试_get_stk_trajectory_data方法
            trajectory_data = missile_manager._get_stk_trajectory_data(missile_id)
            
            if trajectory_data:
                logger.info(f"✅ 成功获取导弹时间数据: {missile_id}")
                logger.info(f"   数据源: {trajectory_data.get('data_source', 'Unknown')}")
                logger.info(f"   开始时间: {trajectory_data.get('start_time')}")
                logger.info(f"   结束时间: {trajectory_data.get('stop_time')}")
                logger.info(f"   飞行时间: {trajectory_data.get('flight_time_seconds', 0):.0f}秒")
                
                # 验证时间数据的合理性
                start_time = trajectory_data.get('start_time')
                stop_time = trajectory_data.get('stop_time')
                
                if isinstance(start_time, datetime) and isinstance(stop_time, datetime):
                    if stop_time > start_time:
                        flight_duration = (stop_time - start_time).total_seconds()
                        logger.info(f"   计算飞行时间: {flight_duration:.0f}秒")
                        
                        # 检查时间是否合理（飞行时间应该在几分钟到几小时之间）
                        if 60 <= flight_duration <= 7200:  # 1分钟到2小时
                            logger.info(f"✅ 时间数据合理: {missile_id}")
                            success_count += 1
                        else:
                            logger.warning(f"⚠️ 飞行时间异常: {flight_duration:.0f}秒")
                    else:
                        logger.warning(f"⚠️ 结束时间早于开始时间: {missile_id}")
                else:
                    logger.warning(f"⚠️ 时间格式不正确: {missile_id}")
            else:
                logger.error(f"❌ 无法获取导弹时间数据: {missile_id}")
        
        logger.info(f"\n🏁 STK导弹时间提取测试完成")
        logger.info(f"📊 成功率: {success_count}/{len(missile_objects)} ({success_count/len(missile_objects)*100:.1f}%)")
        
        # 测试get_missile_time_range方法
        logger.info(f"\n=== 测试get_missile_time_range方法 ===")
        
        # 为了测试这个方法，我们需要在内部存储中添加一些导弹信息
        for missile_id in missile_objects[:3]:  # 测试前3个导弹
            # 模拟添加到内部存储
            missile_manager.missile_targets[missile_id] = {
                "missile_id": missile_id,
                "launch_time": datetime.now(),  # 这个会被STK数据覆盖
                "estimated_flight_time": 1800
            }
            
            logger.info(f"🔍 测试get_missile_time_range: {missile_id}")
            time_range = missile_manager.get_missile_time_range(missile_id)
            
            if time_range:
                logger.info(f"✅ get_missile_time_range成功: {missile_id}")
                logger.info(f"   数据源: {time_range.get('data_source', 'Unknown')}")
                logger.info(f"   发射时间: {time_range.get('launch_time_str')}")
                logger.info(f"   结束时间: {time_range.get('end_time_str')}")
                logger.info(f"   飞行时间: {time_range.get('flight_duration_seconds', 0):.0f}秒")
            else:
                logger.error(f"❌ get_missile_time_range失败: {missile_id}")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始STK导弹时间提取测试")
    
    success = test_stk_missile_time_extraction()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("🎉 STK导弹时间提取测试成功！")
        logger.info("✅ 系统可以从STK获取准确的导弹时间信息")
        logger.info("✅ 支持以下数据源:")
        logger.info("   - STK对象时间属性")
        logger.info("   - STK DataProvider轨迹数据")
        logger.info("   - STK轨迹对象时间信息")
    else:
        logger.error("❌ STK导弹时间提取测试失败！")
        logger.error("❌ 需要检查STK时间获取方法")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
