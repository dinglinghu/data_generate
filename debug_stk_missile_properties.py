#!/usr/bin/env python3
"""
调试STK导弹对象属性和方法
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config_manager import ConfigManager
from src.stk_interface.stk_manager import STKManager
from src.utils.time_manager import UnifiedTimeManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_missile_properties():
    """调试导弹对象的所有属性和方法"""
    logger.info("=== 调试STK导弹对象属性 ===")
    
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
        
        # 获取第一个导弹对象
        scenario = stk_manager.scenario
        missile_obj = None
        missile_name = None
        
        for i in range(scenario.Children.Count):
            child = scenario.Children.Item(i)
            if child.ClassName == "Missile":
                missile_obj = child
                missile_name = child.InstanceName
                logger.info(f"🎯 找到导弹对象: {missile_name}")
                break
        
        if not missile_obj:
            logger.error("❌ 没有找到导弹对象")
            return False
        
        logger.info(f"\n=== 调试导弹对象: {missile_name} ===")
        
        # 1. 检查导弹对象的所有属性
        logger.info("📋 导弹对象的所有属性:")
        missile_attrs = dir(missile_obj)
        for attr in sorted(missile_attrs):
            if not attr.startswith('_'):
                try:
                    value = getattr(missile_obj, attr)
                    attr_type = type(value).__name__
                    logger.info(f"   {attr}: {attr_type}")
                    
                    # 特别关注时间相关的属性
                    if 'time' in attr.lower() or 'start' in attr.lower() or 'stop' in attr.lower():
                        logger.info(f"   ⭐ 时间相关属性: {attr} = {value}")
                        
                except Exception as e:
                    logger.debug(f"   {attr}: 无法访问 ({e})")
        
        # 2. 检查轨迹对象
        logger.info(f"\n📡 检查轨迹对象:")
        try:
            trajectory = missile_obj.Trajectory
            logger.info(f"✅ 轨迹对象类型: {type(trajectory).__name__}")
            
            # 检查轨迹对象的属性
            traj_attrs = dir(trajectory)
            logger.info("📋 轨迹对象的所有属性:")
            for attr in sorted(traj_attrs):
                if not attr.startswith('_'):
                    try:
                        value = getattr(trajectory, attr)
                        attr_type = type(value).__name__
                        logger.info(f"   {attr}: {attr_type}")
                        
                        # 特别关注时间相关的属性
                        if 'time' in attr.lower() or 'start' in attr.lower() or 'stop' in attr.lower() or 'launch' in attr.lower() or 'impact' in attr.lower():
                            logger.info(f"   ⭐ 时间相关属性: {attr} = {value}")
                            
                    except Exception as e:
                        logger.debug(f"   {attr}: 无法访问 ({e})")
            
            # 3. 检查轨迹的具体类型
            logger.info(f"\n🔍 检查轨迹的具体类型:")
            
            # 检查是否有Ballistic属性
            if hasattr(trajectory, 'Ballistic'):
                logger.info("✅ 轨迹有Ballistic属性")
                ballistic = trajectory.Ballistic
                logger.info(f"   Ballistic对象类型: {type(ballistic).__name__}")
                
                # 检查Ballistic对象的属性
                ballistic_attrs = dir(ballistic)
                logger.info("📋 Ballistic对象的所有属性:")
                for attr in sorted(ballistic_attrs):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(ballistic, attr)
                            attr_type = type(value).__name__
                            logger.info(f"   {attr}: {attr_type}")
                            
                            # 特别关注时间相关的属性
                            if 'time' in attr.lower() or 'launch' in attr.lower() or 'impact' in attr.lower():
                                logger.info(f"   ⭐ 时间相关属性: {attr} = {value}")
                                
                        except Exception as e:
                            logger.debug(f"   {attr}: 无法访问 ({e})")
            else:
                logger.warning("⚠️ 轨迹没有Ballistic属性")
            
            # 检查其他可能的轨迹类型
            possible_types = ['GreatArc', 'TwoBodyKeplerian', 'J2Perturbation', 'J4Perturbation', 'HPOP', 'Astrogator']
            for prop_type in possible_types:
                if hasattr(trajectory, prop_type):
                    logger.info(f"✅ 轨迹有{prop_type}属性")
                    prop_obj = getattr(trajectory, prop_type)
                    logger.info(f"   {prop_type}对象类型: {type(prop_obj).__name__}")
                    
                    # 检查这个传播器的属性
                    prop_attrs = dir(prop_obj)
                    time_attrs = [attr for attr in prop_attrs if 'time' in attr.lower() or 'start' in attr.lower() or 'stop' in attr.lower() or 'launch' in attr.lower() or 'impact' in attr.lower()]
                    if time_attrs:
                        logger.info(f"   📅 {prop_type}时间相关属性: {time_attrs}")
                        for attr in time_attrs:
                            try:
                                value = getattr(prop_obj, attr)
                                logger.info(f"   ⭐ {attr} = {value}")
                            except Exception as e:
                                logger.debug(f"   {attr}: 无法访问 ({e})")
                                
        except Exception as e:
            logger.error(f"❌ 轨迹对象访问失败: {e}")
        
        # 4. 检查DataProvider
        logger.info(f"\n📊 检查DataProvider:")
        try:
            data_providers = missile_obj.DataProviders
            logger.info(f"✅ DataProvider集合类型: {type(data_providers).__name__}")
            logger.info(f"   DataProvider数量: {data_providers.Count}")
            
            # 列出所有DataProvider
            logger.info("📋 所有可用的DataProvider:")
            for i in range(data_providers.Count):
                dp = data_providers.Item(i)
                logger.info(f"   {i}: {dp.Name}")
                
                # 特别关注时间相关的DataProvider
                if 'time' in dp.Name.lower() or 'state' in dp.Name.lower() or 'position' in dp.Name.lower():
                    logger.info(f"   ⭐ 重要DataProvider: {dp.Name}")
                    
                    # 尝试获取这个DataProvider的信息
                    try:
                        dp_info = dp.QueryInterface()
                        logger.info(f"      接口类型: {type(dp_info).__name__}")
                    except Exception as e:
                        logger.debug(f"      接口查询失败: {e}")
                        
        except Exception as e:
            logger.error(f"❌ DataProvider访问失败: {e}")
        
        # 5. 尝试Connect命令获取详细信息
        logger.info(f"\n🔗 尝试Connect命令:")
        try:
            # 尝试各种可能的命令
            commands = [
                f'GetValue */Missile/{missile_name} StartTime',
                f'GetValue */Missile/{missile_name} StopTime',
                f'GetValue */Missile/{missile_name} LaunchTime',
                f'GetValue */Missile/{missile_name} ImpactTime',
                f'GetValue */Missile/{missile_name}.StartTime',
                f'GetValue */Missile/{missile_name}.StopTime',
                f'GetValue */Missile/{missile_name}.LaunchTime',
                f'GetValue */Missile/{missile_name}.ImpactTime',
                f'GetValue */Missile/{missile_name}/Trajectory StartTime',
                f'GetValue */Missile/{missile_name}/Trajectory StopTime',
                f'ShowNames */Missile/{missile_name}',
                f'ShowNames */Missile/{missile_name}/Trajectory'
            ]
            
            for cmd in commands:
                try:
                    result = stk_manager.root.ExecuteCommand(cmd)
                    if result:
                        logger.info(f"✅ 命令成功: {cmd}")
                        logger.info(f"   结果: {result}")
                    else:
                        logger.debug(f"❌ 命令失败: {cmd}")
                except Exception as e:
                    logger.debug(f"❌ 命令异常: {cmd}, {e}")
                    
        except Exception as e:
            logger.error(f"❌ Connect命令测试失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 调试异常: {e}")
        return False

def main():
    """主函数"""
    logger.info("🔍 开始STK导弹对象属性调试")
    
    success = debug_missile_properties()
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ STK导弹对象属性调试完成")
    else:
        logger.error("❌ STK导弹对象属性调试失败")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
