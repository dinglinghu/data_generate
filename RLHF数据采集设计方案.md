# STK场景数据采集系统 - RLHF强化学习设计方案

## 1. 项目背景与目标

### 1.1 项目背景
基于现有的STK卫星星座数据采集系统，设计一个专门用于大模型强化学习（RLHF）的数据采集系统，实现卫星星座对导弹目标的跟踪任务规划强化学习。

### 1.2 设计目标
- **状态空间建模**：构建完整的卫星-导弹跟踪状态表示
- **动作空间设计**：定义卫星任务规划和资源分配动作
- **奖励函数设计**：建立多目标优化的奖励机制
- **场景多样性**：生成丰富的训练和评估场景
- **数据质量保证**：确保训练数据的质量和一致性

## 2. 系统架构设计

### 2.1 整体架构
```
现有STK数据采集系统
├── 基础数据采集 (已实现)
│   ├── 卫星位置姿态
│   ├── 导弹轨迹信息
│   ├── 可见性计算
│   └── 载荷状态
└── RLHF扩展模块 (新增)
    ├── RLHF数据采集器
    ├── 场景生成器
    ├── 奖励函数计算器
    ├── 状态空间编码器
    └── 数据质量控制器
```

### 2.2 核心组件

#### 2.2.1 RLHF数据采集器 (`RLHFDataCollector`)
- **功能**：采集强化学习所需的状态-动作-奖励数据
- **输入**：基础STK数据 + 动作指令
- **输出**：结构化的RLHF训练数据

#### 2.2.2 场景生成器 (`RLHFScenarioGenerator`)
- **功能**：生成多样化的训练和评估场景
- **特点**：支持难度分级、场景类型多样化
- **输出**：场景配置文件

#### 2.2.3 状态空间编码器
- **功能**：将原始STK数据转换为标准化的状态向量
- **包含**：卫星状态、导弹状态、环境状态、任务状态

## 3. 强化学习数据结构设计

### 3.1 状态空间 (State Space)

#### 3.1.1 卫星状态
```python
satellite_state = {
    "position": [x, y, z],           # 3D位置坐标 (km)
    "velocity": [vx, vy, vz],        # 3D速度向量 (km/s)
    "attitude": [q0, q1, q2, q3],    # 姿态四元数
    "orbital_elements": {            # 轨道根数
        "semi_major_axis": float,
        "eccentricity": float,
        "inclination": float,
        "raan": float,
        "arg_perigee": float,
        "mean_anomaly": float
    },
    "power_status": {
        "battery_level": float,      # 电池电量 (0-1)
        "solar_panel_efficiency": float,
        "power_consumption": float
    },
    "payload_status": {
        "operational": bool,         # 载荷运行状态
        "pointing_accuracy": float,  # 指向精度
        "detection_range": float     # 探测距离
    }
}
```

#### 3.1.2 导弹威胁状态
```python
missile_state = {
    "position": [x, y, z],           # 当前位置
    "velocity": [vx, vy, vz],        # 速度向量
    "trajectory_prediction": {       # 轨迹预测
        "predicted_path": [[x,y,z], ...],
        "impact_point": [x, y, z],
        "time_to_impact": float
    },
    "threat_level": int,             # 威胁等级 (1-4)
    "flight_phase": str,             # 飞行阶段
    "remaining_time": float          # 剩余飞行时间
}
```

#### 3.1.3 环境状态
```python
environment_state = {
    "time_info": {
        "simulation_time": datetime,
        "time_of_day": float,        # 0-1编码
        "season": int                # 季节编码
    },
    "space_environment": {
        "sun_position": [x, y, z],
        "earth_shadow": bool,
        "solar_activity": int,
        "atmospheric_density": float
    }
}
```

### 3.2 动作空间 (Action Space)

#### 3.2.1 卫星控制动作
```python
satellite_actions = {
    "attitude_control": {
        "target_attitude": [q0, q1, q2, q3],
        "control_mode": str          # "manual", "auto", "tracking"
    },
    "payload_pointing": {
        "target_coordinates": [lat, lon, alt],
        "pointing_mode": str,        # "fixed", "tracking", "scanning"
        "scan_pattern": str
    },
    "power_management": {
        "power_allocation": {
            "payload": float,        # 功率分配比例
            "communication": float,
            "attitude_control": float
        }
    }
}
```

#### 3.2.2 任务规划动作
```python
mission_actions = {
    "target_assignment": {
        "satellite_id": str,
        "target_id": str,
        "priority": int,
        "assignment_duration": float
    },
    "resource_allocation": {
        "communication_bandwidth": dict,
        "computational_resources": dict,
        "observation_time": dict
    },
    "coordination_commands": {
        "handover_instructions": list,
        "collaborative_tracking": bool,
        "formation_adjustment": dict
    }
}
```

### 3.3 奖励函数 (Reward Function)

#### 3.3.1 多目标奖励设计
```python
reward_components = {
    "tracking_performance": {
        "coverage_time": 0.4,       # 权重40%
        "tracking_accuracy": 0.3,
        "detection_success": 0.3
    },
    "resource_efficiency": {
        "power_efficiency": 0.4,    # 权重30%
        "communication_efficiency": 0.3,
        "computational_efficiency": 0.3
    },
    "mission_completion": {
        "threat_neutralization": 0.5, # 权重30%
        "response_time": 0.3,
        "coordination_effectiveness": 0.2
    }
}
```

#### 3.3.2 奖励计算公式
```
总奖励 = Σ(权重i × 归一化奖励i) - 惩罚项

其中：
- 跟踪性能奖励 = 覆盖时间比例 × 跟踪精度 × 检测成功率
- 资源效率奖励 = 1 - (实际消耗 / 最大可用资源)
- 任务完成奖励 = 威胁中和率 × 响应时间因子 × 协调效果
- 惩罚项 = 虚警惩罚 + 资源浪费惩罚 + 任务失败惩罚
```

## 4. 场景多样性设计

### 4.1 威胁场景分类

#### 4.1.1 单一威胁场景
- **导弹数量**：1个
- **难度等级**：简单-中等
- **训练目标**：基础跟踪能力

#### 4.1.2 多威胁场景
- **导弹数量**：2-8个
- **发射模式**：同时发射、协调攻击
- **训练目标**：资源分配和优先级管理

#### 4.1.3 饱和攻击场景
- **导弹数量**：10-20个
- **时间窗口**：60-300秒内密集发射
- **训练目标**：高压力下的快速决策

#### 4.1.4 对抗性场景
- **特点**：规避机动、诱饵部署、电子干扰
- **训练目标**：鲁棒性和适应性

### 4.2 星座配置变化

#### 4.2.1 规模变化
- **小型星座**：2×2 (4颗卫星)
- **中型星座**：3×3 (9颗卫星)
- **大型星座**：4×4 (16颗卫星)

#### 4.2.2 轨道参数变化
- **高度范围**：800-2000 km
- **倾角范围**：45-98度
- **偏心率范围**：0.0-0.1

### 4.3 环境条件变化

#### 4.3.1 时间条件
- **时段**：黎明、白天、黄昏、夜晚
- **季节**：春夏秋冬

#### 4.3.2 空间环境
- **太阳活动**：低、中、高
- **大气密度**：0.8-1.2倍标准值

## 5. 数据采集流程

### 5.1 场景初始化
```python
# 1. 生成场景配置
scenario_config = scenario_generator.generate_single_scenario(
    scenario_type="multiple_threats",
    difficulty_level="medium"
)

# 2. 初始化STK环境
stk_manager.setup_scenario(scenario_config)

# 3. 开始RLHF回合
episode_id = rlhf_collector.start_episode(
    scenario_type=scenario_config.scenario_type,
    scenario_params=scenario_config.__dict__
)
```

### 5.2 数据采集循环
```python
while not episode_done:
    # 1. 获取当前状态
    current_state = rlhf_collector._extract_state_vector(base_data)
    
    # 2. 智能体决策 (外部提供)
    action = agent.select_action(current_state)
    
    # 3. 执行动作并采集数据
    data_point = rlhf_collector.collect_rlhf_data_point(action)
    
    # 4. 检查回合结束条件
    episode_done = data_point.done
    
    # 5. 推进仿真时间
    time_manager.advance_simulation_time()
```

### 5.3 数据保存
```python
# 结束回合
completed_episode = rlhf_collector.end_episode(success=mission_success)

# 保存数据
rlhf_collector.save_rlhf_data(format_type="json")  # 或 "hdf5", "numpy"
```

## 6. 数据质量控制

### 6.1 数据验证规则
- **位置边界检查**：确保坐标在合理范围内
- **速度限制检查**：验证速度符合物理约束
- **时间一致性检查**：保证时间序列的连续性
- **状态完整性检查**：确保必要字段不缺失

### 6.2 异常检测
- **统计异常值检测**：识别超出3σ范围的数据点
- **物理约束检查**：验证是否违反物理定律
- **时间异常检测**：发现时间序列中的突变

### 6.3 数据增强策略
- **噪声注入**：添加高斯噪声提高鲁棒性
- **场景扰动**：轻微修改场景参数增加多样性
- **时间抖动**：在发射时间上添加随机扰动

## 7. 输出数据格式

### 7.1 JSON格式 (人类可读)
```json
{
  "metadata": {
    "collection_time": "2025-07-31T10:00:00",
    "total_episodes": 100,
    "total_data_points": 50000
  },
  "episodes": [
    {
      "episode_id": "episode_001",
      "scenario_type": "multiple_threats",
      "total_reward": 85.6,
      "success": true,
      "data_points": [
        {
          "timestamp": "2025-07-31T10:05:00",
          "state": {...},
          "action": {...},
          "reward": 0.8,
          "next_state": {...},
          "done": false
        }
      ]
    }
  ]
}
```

### 7.2 HDF5格式 (高效存储)
```
rlhf_data.h5
├── metadata/
│   ├── collection_info
│   └── scenario_configs
├── episodes/
│   ├── episode_000/
│   │   ├── states (array)
│   │   ├── actions (array)
│   │   ├── rewards (array)
│   │   └── metadata
│   └── episode_001/
└── statistics/
    ├── reward_distribution
    └── state_statistics
```

### 7.3 NumPy格式 (机器学习友好)
```python
# 保存为.npz文件
np.savez('rlhf_data.npz',
    states=states_array,      # (N, state_dim)
    actions=actions_array,    # (N, action_dim)
    rewards=rewards_array,    # (N,)
    next_states=next_states_array,
    dones=dones_array,
    episode_ids=episode_ids
)
```

## 8. 评估指标

### 8.1 性能指标
- **跟踪精度**：目标跟踪的准确性
- **覆盖率**：时间和空间覆盖比例
- **响应时间**：从检测到跟踪的延迟
- **虚警率**：误报的频率

### 8.2 效率指标
- **功率效率**：单位功耗的跟踪效果
- **计算效率**：算法执行时间
- **通信效率**：数据传输效率

### 8.3 鲁棒性指标
- **噪声容忍度**：在噪声环境下的性能
- **故障恢复**：设备故障后的恢复能力
- **适应性**：对新场景的适应速度

## 9. 使用示例

### 9.1 生成训练数据
```python
# 初始化系统
rlhf_system = RLHFDataCollectionSystem()

# 生成训练场景
scenarios = scenario_generator.generate_training_scenarios(
    num_scenarios=1000,
    difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3}
)

# 采集训练数据
for scenario in scenarios:
    rlhf_system.collect_episode_data(scenario)

# 保存数据
rlhf_system.save_all_data(format_type="hdf5")
```

### 9.2 生成评估数据
```python
# 生成评估场景
eval_scenarios = scenario_generator.generate_evaluation_scenarios(100)

# 采集评估数据
eval_data = rlhf_system.collect_evaluation_data(eval_scenarios)

# 计算评估指标
metrics = rlhf_system.calculate_evaluation_metrics(eval_data)
```

## 10. 扩展性设计

### 10.1 模块化设计
- 每个组件都可以独立替换和升级
- 支持新的状态特征和动作类型
- 可扩展的奖励函数设计

### 10.2 配置驱动
- 所有参数通过配置文件管理
- 支持运行时配置修改
- 版本化的配置管理

### 10.3 分布式支持
- 支持多进程并行数据采集
- 可扩展到集群环境
- 数据流水线处理

这个设计方案基于现有的STK数据采集系统，通过添加RLHF专用模块，实现了完整的强化学习数据采集功能，为卫星星座导弹跟踪任务的强化学习训练提供了高质量的数据支持。
