#!/usr/bin/env python3
"""
测试导弹管理功能
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

def test_missile_management():
    """测试导弹管理功能"""
    logger.info("=== 测试导弹管理功能 ===")
    
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

        # 创建一个简单的输出管理器模拟
        class MockOutputManager:
            def save_data(self, data, filename):
                return f"mock_output/{filename}"

        output_manager = MockOutputManager()
        missile_manager = MissileManager(stk_manager, missile_config, output_manager)
        
        logger.info("✅ 导弹管理器初始化成功")
        
        # 获取仿真时间范围
        simulation_start = time_manager.start_time
        simulation_end = time_manager.end_time
        
        logger.info(f"📅 仿真时间范围: {simulation_start} - {simulation_end}")
        
        # 测试1: 创建一些测试导弹（包括有效和无效的）
        logger.info("\n=== 测试1: 创建测试导弹 ===")
        
        test_missiles = []
        
        # 创建有效导弹（在仿真时间范围内）
        for i in range(3):
            missile_scenario = {
                "missile_id": f"ValidMissile{i+1:02d}",
                "missile_type": "ICBM",
                "threat_level": "高",
                "description": f"有效测试导弹 {i+1}",
                "launch_position": {"lat": 40.0 + i*5, "lon": 116.0 + i*5, "alt": 0},
                "target_position": {"lat": -40.0 - i*5, "lon": -116.0 - i*5, "alt": 0},
                "trajectory_params": {"max_altitude": 800, "flight_time": 1200},
                "launch_time": simulation_start + timedelta(minutes=i*30),
                "launch_sequence": i+1
            }
            
            result = missile_manager.create_single_missile_target(missile_scenario)
            if result:
                test_missiles.append(missile_scenario["missile_id"])
                logger.info(f"✅ 创建有效导弹: {missile_scenario['missile_id']}")
        
        # 创建无效导弹（超出仿真时间范围）
        for i in range(2):
            missile_scenario = {
                "missile_id": f"InvalidMissile{i+1:02d}",
                "missile_type": "IRBM",
                "threat_level": "中",
                "description": f"无效测试导弹 {i+1}",
                "launch_position": {"lat": -30.0 - i*5, "lon": 120.0 + i*5, "alt": 0},
                "target_position": {"lat": 30.0 + i*5, "lon": -120.0 - i*5, "alt": 0},
                "trajectory_params": {"max_altitude": 600, "flight_time": 900},
                "launch_time": simulation_end + timedelta(hours=i+1),  # 超出仿真时间
                "launch_sequence": i+4
            }
            
            result = missile_manager.create_single_missile_target(missile_scenario)
            if result:
                test_missiles.append(missile_scenario["missile_id"])
                logger.info(f"✅ 创建无效导弹: {missile_scenario['missile_id']}")
        
        logger.info(f"📊 创建测试导弹完成: {len(test_missiles)}个")
        
        # 测试2: 检查导弹时间范围
        logger.info("\n=== 测试2: 检查导弹时间范围 ===")
        
        missile_check = missile_manager.check_missiles_in_simulation_range(simulation_start, simulation_end)
        
        logger.info(f"📊 导弹时间检查结果:")
        logger.info(f"   总导弹数: {missile_check['total_missiles']}")
        logger.info(f"   有效导弹: {missile_check['valid_count']}个")
        logger.info(f"   无效导弹: {missile_check['invalid_count']}个")
        logger.info(f"   有效导弹列表: {missile_check['valid_missiles']}")
        logger.info(f"   无效导弹列表: {missile_check['invalid_missiles']}")
        
        # 测试3: 导弹数量管理
        logger.info("\n=== 测试3: 导弹数量管理 ===")
        
        management_result = missile_manager.manage_missile_count(
            simulation_start=simulation_start,
            simulation_end=simulation_end,
            target_min=5,
            target_max=20
        )
        
        if management_result.get("management_success", False):
            logger.info("✅ 导弹数量管理成功")
            logger.info(f"📊 管理结果:")
            logger.info(f"   初始总数: {management_result.get('initial_total', 0)}")
            logger.info(f"   初始有效: {management_result.get('initial_valid', 0)}")
            logger.info(f"   删除无效: {management_result.get('removed_invalid', 0)}")
            logger.info(f"   目标数量: {management_result.get('target_count', 0)}")
            logger.info(f"   需要添加: {management_result.get('missiles_to_add', 0)}")
            logger.info(f"   成功添加: {management_result.get('successfully_added', 0)}")
            logger.info(f"   最终数量: {management_result.get('final_count', 0)}")
            
            added_missiles = management_result.get("added_missile_ids", [])
            if added_missiles:
                logger.info(f"🆕 新添加的导弹: {', '.join(added_missiles)}")
            
            # 验证最终结果
            final_count = management_result.get('final_count', 0)
            if 5 <= final_count <= 20:
                logger.info(f"✅ 导弹数量在目标范围内: {final_count}")
                return True
            else:
                logger.warning(f"⚠️ 导弹数量超出目标范围: {final_count}")
                return False
        else:
            logger.error(f"❌ 导弹数量管理失败: {management_result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试异常: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始导弹管理功能测试")
    
    success = test_missile_management()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("🎉 导弹管理功能测试成功！")
        logger.info("✅ 系统已准备好进行智能导弹管理")
        logger.info("✅ 支持以下功能:")
        logger.info("   - 检测导弹发射和结束时间")
        logger.info("   - 删除仿真时间范围外的导弹")
        logger.info("   - 随机添加全球导弹威胁")
        logger.info("   - 维持5-20颗导弹数量")
    else:
        logger.error("❌ 导弹管理功能测试失败！")
        logger.error("❌ 需要检查导弹管理逻辑")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
