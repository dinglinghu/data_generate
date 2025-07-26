#!/usr/bin/env python3
"""
STK卫星星座数据采集主程序
基于STK软件结合Python代码进行数据生成，严格按照ADK官网文档实现

主要功能：
1. 创建Walker星座和载荷配置
2. 随机添加导弹目标
3. 采集卫星位置姿态、载荷参数、导弹轨迹、可见性时间窗口数据
4. 定期保存数据为JSON格式

使用统一时间管理器，禁止使用系统时间
"""

import logging
import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stk_data_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 导入项目模块
from src.utils.config_manager import get_config_manager
from src.utils.time_manager import get_time_manager
from src.stk_interface.stk_manager import STKManager
from src.stk_interface.missile_manager import MissileManager
from src.stk_interface.visibility_calculator import VisibilityCalculator
from src.constellation.constellation_manager import ConstellationManager
from src.data_collection.data_collector import DataCollector

class STKDataCollectionSystem:
    """STK数据采集系统主类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化数据采集系统
        
        Args:
            config_path: 配置文件路径
        """
        logger.info("🚀 STK数据采集系统启动...")
        
        # 初始化配置和时间管理器
        self.config_manager = get_config_manager(config_path)
        self.time_manager = get_time_manager(self.config_manager)
        
        # 初始化STK管理器
        stk_config = self.config_manager.get_stk_config()
        self.stk_manager = STKManager(stk_config)
        
        # 初始化其他组件
        self.missile_manager = None
        self.visibility_calculator = None
        self.constellation_manager = None
        self.data_collector = None
        
        # 导弹管理
        self.active_missiles = {}
        self.missile_counter = 0
        
        logger.info("✅ STK数据采集系统初始化完成")
    
    async def run(self):
        """运行数据采集流程"""
        try:
            logger.info("🎯 开始执行数据采集流程...")
            
            # 1. 连接STK
            if not await self._connect_stk():
                logger.error("❌ STK连接失败，程序退出")
                return False
            
            # 2. 初始化组件
            if not self._initialize_components():
                logger.error("❌ 组件初始化失败，程序退出")
                return False
            
            # 3. 创建星座（如果需要）
            if not await self._setup_constellation():
                logger.error("❌ 星座设置失败，程序退出")
                return False

            # 4. 创建首批导弹目标进行验证
            if not await self._create_initial_missiles():
                logger.error("❌ 首批导弹目标创建失败，程序退出")
                return False

            # 5. 等待用户确认场景创建是否正确
            if not self._wait_for_user_confirmation():
                logger.info("👤 用户选择退出，程序结束")
                return True

            # 6. 执行数据采集循环
            await self._data_collection_loop()
            
            logger.info("✅ 数据采集流程完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据采集流程异常: {e}")
            return False
        finally:
            self._cleanup()
    
    async def _connect_stk(self) -> bool:
        """连接STK"""
        try:
            logger.info("🔗 连接STK...")
            success = self.stk_manager.connect()
            if success:
                logger.info("✅ STK连接成功")
                return True
            else:
                logger.error("❌ STK连接失败")
                return False
        except Exception as e:
            logger.error(f"❌ STK连接异常: {e}")
            return False
    
    def _initialize_components(self) -> bool:
        """初始化所有组件"""
        try:
            logger.info("🔧 初始化系统组件...")
            
            # 初始化导弹管理器
            self.missile_manager = MissileManager(
                self.stk_manager, 
                self.config_manager.config,
                None  # output_manager暂时为None
            )
            
            # 初始化可见性计算器
            self.visibility_calculator = VisibilityCalculator(self.stk_manager)
            
            # 初始化星座管理器
            self.constellation_manager = ConstellationManager(
                self.stk_manager, 
                self.config_manager
            )
            
            # 初始化数据采集器
            self.data_collector = DataCollector(
                self.stk_manager,
                self.missile_manager,
                self.visibility_calculator,
                self.constellation_manager,
                self.config_manager,
                self.time_manager
            )
            
            logger.info("✅ 系统组件初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 组件初始化失败: {e}")
            return False
    
    async def _setup_constellation(self) -> bool:
        """设置星座"""
        try:
            logger.info("🌟 设置Walker星座...")

            # 验证星座参数
            if not self.constellation_manager.validate_constellation_parameters():
                logger.error("❌ 星座参数验证失败")
                return False

            # 创建星座
            success = self.constellation_manager.create_walker_constellation()
            if success:
                logger.info("✅ Walker星座创建成功")

                # 显示星座信息
                constellation_info = self.constellation_manager.get_constellation_info()
                logger.info("📊 Walker星座配置信息:")
                logger.info(f"   星座类型: {constellation_info['type']}")
                logger.info(f"   轨道面数: {constellation_info['planes']}")
                logger.info(f"   每面卫星数: {constellation_info['satellites_per_plane']}")
                logger.info(f"   总卫星数: {constellation_info['total_satellites']}")
                logger.info(f"   卫星列表: {', '.join(constellation_info['satellite_list'])}")

                return True
            else:
                logger.error("❌ Walker星座创建失败")
                return False

        except Exception as e:
            logger.error(f"❌ 星座设置异常: {e}")
            return False

    async def _create_initial_missiles(self) -> bool:
        """创建首批导弹目标进行验证"""
        try:
            logger.info("🚀 创建首批导弹目标进行验证...")

            # 获取导弹配置
            missile_config = self.config_manager.get_missile_config()

            # 创建测试导弹目标
            system_config = self.config_manager.get_system_config()
            initial_missile_count = system_config["testing"]["initial_missile_count"]
            logger.info(f"📊 计划创建 {initial_missile_count} 个测试导弹目标")

            success_count = 0
            for i in range(initial_missile_count):
                missile_id = f"TestMissile{i+1:02d}"

                try:
                    # 生成随机发射和目标位置
                    launch_position = self._generate_random_position(missile_config["global_launch_positions"])
                    target_position = self._generate_random_position(missile_config["global_target_positions"])

                    # 生成轨迹参数
                    trajectory_params = self._generate_trajectory_params(missile_config["trajectory_params"])

                    # 计算发射时间（从当前仿真时间开始）
                    launch_time = self.time_manager.current_simulation_time

                    # 创建导弹场景配置
                    missile_scenario = {
                        "missile_id": missile_id,
                        "launch_position": launch_position,
                        "target_position": target_position,
                        "trajectory_params": trajectory_params,
                        "launch_time": launch_time
                    }

                    logger.info(f"🎯 创建测试导弹 {missile_id}:")
                    logger.info(f"   发射位置: 纬度{launch_position['lat']:.2f}°, 经度{launch_position['lon']:.2f}°")
                    logger.info(f"   目标位置: 纬度{target_position['lat']:.2f}°, 经度{target_position['lon']:.2f}°")
                    logger.info(f"   最大高度: {trajectory_params['max_altitude']:.1f} km")
                    logger.info(f"   飞行时间: {trajectory_params['flight_time']:.0f} 秒")

                    # 创建导弹目标
                    result = self.missile_manager.create_single_missile_target(missile_scenario)

                    if result is not None:
                        logger.info(f"✅ 测试导弹 {missile_id} 创建成功")
                        success_count += 1
                    else:
                        logger.warning(f"⚠️ 测试导弹 {missile_id} 创建失败")

                except Exception as missile_error:
                    logger.error(f"❌ 创建测试导弹 {missile_id} 异常: {missile_error}")

            logger.info(f"📊 首批导弹目标创建完成: {success_count}/{initial_missile_count} 成功")

            if success_count > 0:
                logger.info("✅ 首批导弹目标创建成功，导弹配置方法验证通过")
                return True
            else:
                logger.error("❌ 所有测试导弹创建失败，导弹配置方法需要修复")
                return False

        except Exception as e:
            logger.error(f"❌ 创建首批导弹目标异常: {e}")
            return False

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

    def _wait_for_user_confirmation(self) -> bool:
        """等待用户确认场景创建是否正确"""
        try:
            logger.info("=" * 60)
            logger.info("🎯 场景创建完成！请检查STK软件中的场景")
            logger.info("=" * 60)

            # 显示场景摘要
            constellation_info = self.constellation_manager.get_constellation_info()
            logger.info("📋 场景摘要:")
            logger.info(f"   ✅ Walker星座: {constellation_info['total_satellites']}颗卫星")
            logger.info(f"   ✅ 轨道面配置: {constellation_info['planes']}个轨道面")
            logger.info(f"   ✅ 载荷配置: 光学传感器 (锥形模式)")
            logger.info(f"   ✅ 测试导弹: 3个测试导弹目标")
            logger.info(f"   ✅ 仿真时间: {self.time_manager.start_time} - {self.time_manager.end_time}")

            # 显示轨道参数
            ref_sat = constellation_info['reference_satellite']
            logger.info("🛰️ 参考卫星轨道参数:")
            logger.info(f"   轨道高度: {ref_sat['altitude']} km")
            logger.info(f"   轨道倾角: {ref_sat['inclination']}°")
            logger.info(f"   偏心率: {ref_sat['eccentricity']}")
            logger.info(f"   近地点幅角: {ref_sat['arg_of_perigee']}°")
            logger.info(f"   RAAN偏移: {ref_sat['raan_offset']}°")
            logger.info(f"   平近点角偏移: {ref_sat['mean_anomaly_offset']}°")

            # 显示载荷配置信息
            payload_config = self.config_manager.get_payload_config()
            logger.info("📡 载荷传感器配置:")
            logger.info(f"   传感器模式: {payload_config.get('sensor_pattern', 'Conic')} (锥形)")
            logger.info(f"   内锥角: {payload_config.get('inner_cone_half_angle', 66.1)}°")
            logger.info(f"   外锥角: {payload_config.get('outer_cone_half_angle', 85.0)}°")
            logger.info(f"   指向方位角: {payload_config.get('pointing', {}).get('azimuth', 0.0)}°")
            logger.info(f"   指向俯仰角: {payload_config.get('pointing', {}).get('elevation', 90.0)}°")

            logger.info("=" * 60)
            logger.info("🔍 请在STK软件中检查以下内容:")
            logger.info("   1. 卫星轨道是否符合Walker星座配置")
            logger.info("   2. 载荷传感器是否正确配置为锥形模式")
            logger.info("   3. 测试导弹目标是否正确创建")
            logger.info("   4. 仿真时间范围是否正确")
            logger.info("   5. 场景整体是否符合预期")
            logger.info("=" * 60)

            # 等待用户按回车确认
            input("✋ 请检查STK场景，确认无误后按回车键继续数据采集...")
            logger.info("👍 用户确认场景正确，开始数据采集...")
            return True

        except KeyboardInterrupt:
            logger.info("⚠️ 用户中断确认过程")
            return False
        except Exception as e:
            logger.error(f"❌ 用户确认过程异常: {e}")
            return False
    
    async def _data_collection_loop(self):
        """数据采集主循环 - 使用固定仿真场景时间范围"""
        try:
            logger.info("📊 开始数据采集循环...")
            logger.info("🕐 采集策略: 使用固定的仿真场景时间范围进行数据采集")

            missile_config = self.config_manager.get_missile_config()
            max_concurrent = missile_config.get("max_concurrent_missiles", 5)

            collection_count = 0

            while not self.time_manager.is_simulation_finished() and not self.time_manager.is_collection_finished():
                # 获取下一次采集时间
                next_collection_time = self.time_manager.get_next_collection_time()

                collection_count += 1
                # 移除旧的日志，新的详细日志在data_collector中输出
                logger.info(f"🎯 采集策略: 使用固定场景时间范围，当前采集时间点: {next_collection_time}")

                # 数据采集前进行导弹管理
                await self._manage_missiles_before_collection()

                # 执行数据采集 - 使用固定场景时间范围
                data_snapshot = self.data_collector.collect_data_at_time(next_collection_time)

                if data_snapshot:
                    progress = self.time_manager.get_simulation_progress()
                    scenario_time_fixed = data_snapshot.get("metadata", {}).get("scenario_time_fixed", False)
                    time_status = "✅ 使用固定场景时间范围" if scenario_time_fixed else "⚠️ 场景时间配置异常"

                    logger.info(f"📊 数据采集成功: 进度 {progress:.1f}%, {time_status}")
                    logger.info(f"📈 采集数据: {len(data_snapshot.get('satellites', []))}颗卫星, "
                               f"{len(data_snapshot.get('missiles', []))}个导弹, "
                               f"{len(data_snapshot.get('visibility', []))}个可见性记录")

                # 检查是否需要保存数据
                if self.time_manager.should_save_data():
                    saved_file = self.data_collector.save_collected_data()
                    if saved_file:
                        logger.info(f"💾 数据已保存: {saved_file}")

                # 清理过期导弹
                self._cleanup_expired_missiles()

                # 推进到下一次采集时间
                self.time_manager.advance_simulation_time(next_collection_time)

                # 短暂延迟以确保STK处理完成
                system_config = self.config_manager.get_system_config()
                delay = system_config["delays"]["collection_loop"]
                await asyncio.sleep(delay)

            # 保存最后的数据
            final_file = self.data_collector.save_collected_data()
            if final_file:
                logger.info(f"💾 最终数据已保存: {final_file}")

            # 输出采集摘要
            summary = self.data_collector.get_collection_summary()
            progress = self.time_manager.get_collection_progress()

            logger.info("=" * 80)
            logger.info("🎉 【数据采集任务完成】")
            logger.info(f"📈 数据采集摘要: {summary}")
            logger.info(f"📊 最终进度: {progress['current_count']}/{progress['total_count']} ({progress['progress_percentage']}%)")
            logger.info(f"🏁 数据采集循环完成，共进行{collection_count}次采集")

            # 检查完成原因
            if self.time_manager.is_collection_finished():
                logger.info("✅ 完成原因: 达到目标采集次数")
            elif self.time_manager.is_simulation_finished():
                logger.info("⏰ 完成原因: 仿真时间结束")

            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"❌ 数据采集循环异常: {e}")

    async def _manage_missiles_before_collection(self):
        """数据采集前的导弹管理"""
        try:
            logger.info("🚀 开始数据采集前导弹管理...")

            # 获取仿真时间范围
            simulation_start = self.time_manager.start_time
            simulation_end = self.time_manager.end_time

            logger.info(f"📅 仿真时间范围: {simulation_start} - {simulation_end}")

            # 执行导弹数量管理
            system_config = self.config_manager.get_system_config()
            mgmt_range = system_config["missile_management_range"]
            management_result = self.missile_manager.manage_missile_count(
                simulation_start=simulation_start,
                simulation_end=simulation_end,
                target_min=mgmt_range["target_min"],
                target_max=mgmt_range["target_max"]
            )

            if management_result.get("management_success", False):
                logger.info("✅ 导弹管理完成")
                logger.info(f"📊 导弹管理结果:")
                logger.info(f"   初始导弹: {management_result.get('initial_total', 0)}个")
                logger.info(f"   删除无效: {management_result.get('removed_invalid', 0)}个")
                logger.info(f"   添加新导弹: {management_result.get('successfully_added', 0)}个")
                logger.info(f"   最终导弹: {management_result.get('final_count', 0)}个")

                # 显示新添加的导弹
                added_missiles = management_result.get("added_missile_ids", [])
                if added_missiles:
                    logger.info(f"🆕 新添加的导弹: {', '.join(added_missiles)}")
            else:
                logger.error(f"❌ 导弹管理失败: {management_result.get('error', '未知错误')}")

        except Exception as e:
            logger.error(f"❌ 导弹管理异常: {e}")

    async def _maybe_add_missile(self):
        """随机添加导弹目标"""
        try:
            missile_config = self.config_manager.get_missile_config()
            max_concurrent = missile_config.get("max_concurrent_missiles", 5)
            
            # 检查当前导弹数量
            current_count = len(self.active_missiles)
            
            if current_count < max_concurrent:
                # 随机决定是否添加导弹
                system_config = self.config_manager.get_system_config()
                add_probability = system_config["testing"]["missile_add_probability"]
                if random.random() < add_probability:
                    await self._add_random_missile()
                    
        except Exception as e:
            logger.error(f"❌ 添加导弹检查失败: {e}")
    
    async def _add_random_missile(self):
        """添加随机导弹目标"""
        try:
            self.missile_counter += 1
            missile_id = f"ICBM_Threat_{self.missile_counter:02d}"
            
            # 生成随机发射和目标位置
            missile_config = self.config_manager.get_missile_config()
            launch_pos_config = missile_config.get("global_launch_positions", {})
            target_pos_config = missile_config.get("global_target_positions", {})
            
            launch_position = {
                "lat": random.uniform(*launch_pos_config.get("lat_range", [-60, 60])),
                "lon": random.uniform(*launch_pos_config.get("lon_range", [-180, 180])),
                "alt": random.uniform(*launch_pos_config.get("alt_range", [0, 100]))
            }
            
            target_position = {
                "lat": random.uniform(*target_pos_config.get("lat_range", [-60, 60])),
                "lon": random.uniform(*target_pos_config.get("lon_range", [-180, 180])),
                "alt": random.uniform(*target_pos_config.get("alt_range", [0, 100]))
            }
            
            # 创建导弹场景
            missile_scenario = {
                "missile_id": missile_id,
                "missile_type": "ballistic_missile",
                "description": f"随机生成的导弹威胁 {missile_id}",
                "threat_level": "高",
                "launch_position": launch_position,
                "target_position": target_position,
                "launch_sequence": self.missile_counter
            }
            
            # 创建导弹
            missile_info = self.missile_manager.create_single_missile_target(missile_scenario)
            
            if missile_info:
                self.active_missiles[missile_id] = missile_info
                logger.info(f"🚀 随机导弹添加成功: {missile_id}")
                logger.info(f"   发射位置: ({launch_position['lat']:.2f}°, {launch_position['lon']:.2f}°)")
                logger.info(f"   目标位置: ({target_position['lat']:.2f}°, {target_position['lon']:.2f}°)")
            else:
                logger.warning(f"⚠️ 随机导弹添加失败: {missile_id}")
                
        except Exception as e:
            logger.error(f"❌ 添加随机导弹失败: {e}")
    
    def _cleanup_expired_missiles(self):
        """清理过期的导弹"""
        try:
            current_time = self.time_manager.current_simulation_time
            expired_missiles = []
            
            for missile_id, missile_info in self.active_missiles.items():
                launch_time = missile_info.get("launch_time")
                if isinstance(launch_time, datetime):
                    # 使用配置的导弹飞行时间
                    missile_mgmt_config = self.config_manager.get_missile_management_config()
                    flight_minutes = missile_mgmt_config["flight_time"]["default_minutes"]
                    impact_time = launch_time + timedelta(minutes=flight_minutes)
                    
                    if current_time > impact_time:
                        expired_missiles.append(missile_id)
            
            # 移除过期导弹
            for missile_id in expired_missiles:
                del self.active_missiles[missile_id]
                logger.info(f"🗑️ 导弹已过期移除: {missile_id}")
                
        except Exception as e:
            logger.error(f"❌ 清理过期导弹失败: {e}")
    
    def _cleanup(self):
        """清理资源"""
        try:
            logger.info("🧹 清理系统资源...")
            
            if self.stk_manager:
                self.stk_manager.disconnect()
                
            logger.info("✅ 资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理失败: {e}")

async def main():
    """主函数"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 STK卫星星座数据采集系统启动")
        logger.info("=" * 60)
        
        # 创建数据采集系统
        system = STKDataCollectionSystem()
        
        # 运行系统
        success = await system.run()
        
        if success:
            logger.info("✅ 数据采集系统运行完成")
        else:
            logger.error("❌ 数据采集系统运行失败")
            
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断程序")
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
    finally:
        logger.info("=" * 60)
        logger.info("🏁 STK卫星星座数据采集系统结束")
        logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
