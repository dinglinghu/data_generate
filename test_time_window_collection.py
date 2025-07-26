#!/usr/bin/env python3
"""
测试1秒时间窗口数据采集策略
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
from src.data_collection.data_collector import DataCollector

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_time_window_data_collection():
    """测试1秒时间窗口数据采集"""
    logger.info("=== 测试1秒时间窗口数据采集策略 ===")
    
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
        
        # 初始化数据采集器
        data_collector = DataCollector(
            stk_manager=stk_manager,
            time_manager=time_manager,
            constellation_manager=None,  # 简化测试
            missile_manager=None,
            visibility_calculator=None
        )
        
        logger.info("✅ 数据采集器初始化成功")
        
        # 测试时间窗口设置
        logger.info("开始测试STK时间窗口设置...")
        
        # 获取当前仿真时间
        current_time = time_manager.current_simulation_time
        logger.info(f"当前仿真时间: {current_time}")
        
        # 测试多个1秒时间窗口
        test_windows = []
        for i in range(5):
            window_start = current_time + timedelta(seconds=i*60)  # 每分钟一个窗口
            window_end = window_start + timedelta(seconds=1)
            test_windows.append((window_start, window_end))
        
        logger.info(f"计划测试 {len(test_windows)} 个时间窗口")
        
        success_count = 0
        for i, (start_time, end_time) in enumerate(test_windows, 1):
            logger.info(f"\n--- 测试时间窗口 {i}/{len(test_windows)} ---")
            logger.info(f"时间窗口: {start_time} -> {end_time}")
            
            # 测试时间窗口设置
            success = data_collector._set_stk_scenario_time_window(start_time, end_time)
            
            if success:
                logger.info(f"✅ 时间窗口 {i} 设置成功")
                success_count += 1
                
                # 验证时间设置是否生效
                try:
                    scenario = stk_manager.scenario
                    current_start = scenario.StartTime
                    current_stop = scenario.StopTime
                    logger.info(f"📊 STK场景当前时间: {current_start} - {current_stop}")
                except Exception as verify_error:
                    logger.warning(f"⚠️ 时间验证失败: {verify_error}")
                
            else:
                logger.warning(f"❌ 时间窗口 {i} 设置失败")
            
            # 短暂延迟
            time.sleep(0.5)
        
        logger.info(f"\n🏁 时间窗口设置测试完成")
        logger.info(f"📊 成功率: {success_count}/{len(test_windows)} ({success_count/len(test_windows)*100:.1f}%)")
        
        # 如果有成功的时间窗口设置，测试数据采集
        if success_count > 0:
            logger.info("\n=== 测试数据采集功能 ===")
            
            # 选择第一个时间窗口进行数据采集测试
            test_time = test_windows[0][0]
            logger.info(f"使用时间点进行数据采集测试: {test_time}")
            
            # 执行数据采集
            data_snapshot = data_collector.collect_data_at_time(test_time)
            
            if data_snapshot:
                logger.info("✅ 数据采集测试成功")
                logger.info(f"📊 采集数据摘要:")
                logger.info(f"   采集时间: {data_snapshot.get('collection_time')}")
                logger.info(f"   时间窗口: {data_snapshot.get('time_window', {})}")
                logger.info(f"   仿真进度: {data_snapshot.get('simulation_progress', 0):.1f}%")
                logger.info(f"   STK时间窗口设置: {data_snapshot.get('metadata', {}).get('stk_time_window_set', False)}")
                
                return True
            else:
                logger.error("❌ 数据采集测试失败")
                return False
        else:
            logger.error("❌ 所有时间窗口设置都失败，无法进行数据采集测试")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始1秒时间窗口数据采集策略测试")
    
    success = test_time_window_data_collection()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("🎉 1秒时间窗口数据采集策略测试成功！")
        logger.info("✅ 系统已准备好使用新的数据采集策略")
    else:
        logger.error("❌ 1秒时间窗口数据采集策略测试失败！")
        logger.error("❌ 需要检查STK时间设置方法")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
