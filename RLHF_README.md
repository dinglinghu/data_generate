# STK场景数据采集系统 - RLHF强化学习扩展

## 概述

本项目在现有STK卫星星座数据采集系统基础上，扩展了专门用于强化学习（RLHF）的数据采集功能，实现卫星星座对导弹目标的跟踪任务规划强化学习数据生成。

## 核心特性

### 🎯 强化学习数据结构
- **状态空间**：卫星状态、导弹威胁、环境条件、任务状态
- **动作空间**：卫星控制、任务规划、资源分配
- **奖励函数**：多目标优化（跟踪性能、资源效率、任务完成）

### 🎭 场景多样性
- **威胁场景**：单一威胁、多威胁、饱和攻击、对抗性场景
- **星座配置**：不同规模和轨道参数的星座
- **环境变化**：时间条件、空间环境、干扰条件

### 📊 数据质量保证
- **数据验证**：位置边界、速度限制、时间一致性检查
- **异常检测**：统计异常、物理约束、时间异常
- **数据增强**：噪声注入、场景扰动、时间抖动

## 系统架构

```
STK基础系统 (已有)
├── 卫星星座管理
├── 导弹目标管理
├── 可见性计算
└── 基础数据采集

RLHF扩展模块 (新增)
├── RLHFDataCollector      # RLHF数据采集器
├── RLHFScenarioGenerator  # 场景生成器
├── RLHFDataCollectionSystem # 系统集成
└── 配置文件和示例
```

## 快速开始

### 1. 环境准备

确保已安装基础系统的依赖：
```bash
# 安装Python依赖
pip install -r requirements.txt

# 额外的RLHF依赖
pip install numpy h5py
```

### 2. 配置文件

RLHF配置文件位于 `config/rlhf_scenarios.yaml`，包含：
- 场景多样性配置
- 强化学习数据标注
- 训练数据生成策略
- 评估指标配置

### 3. 运行示例

```bash
# 运行RLHF数据采集示例
python rlhf_data_collection_example.py
```

### 4. 自定义使用

```python
from main import STKDataCollectionSystem
from src.rlhf_data_collection.rlhf_system import RLHFDataCollectionSystem

# 初始化系统
base_system = STKDataCollectionSystem()
rlhf_system = RLHFDataCollectionSystem(base_system)

# 生成训练数据
training_stats = await rlhf_system.generate_training_dataset(
    num_scenarios=100,
    difficulty_distribution={"easy": 0.3, "medium": 0.4, "hard": 0.3}
)

# 生成评估数据
eval_stats = await rlhf_system.generate_evaluation_dataset(50)
```

## 数据格式

### 状态空间示例
```python
state = {
    "satellite_positions": [[x1,y1,z1], [x2,y2,z2], ...],  # 卫星位置
    "satellite_velocities": [[vx1,vy1,vz1], ...],          # 卫星速度
    "missile_positions": [[x1,y1,z1], ...],                # 导弹位置
    "threat_levels": [1, 2, 3, ...],                       # 威胁等级
    "visibility_matrix": [[1,0,1], [0,1,0], ...],          # 可见性矩阵
    "mission_progress": 0.65,                               # 任务进度
    "active_targets": 3                                     # 活跃目标数
}
```

### 动作空间示例
```python
action = {
    "satellite_actions": {
        "Satellite01": {
            "payload_pointing": {
                "target_coordinates": [lat, lon, alt],
                "pointing_mode": "tracking"
            },
            "power_management": {
                "power_allocation": {
                    "payload": 0.6,
                    "communication": 0.2,
                    "attitude_control": 0.2
                }
            }
        }
    },
    "mission_actions": {
        "target_assignments": [
            {
                "satellite_id": "Satellite01",
                "target_id": "Missile01",
                "priority": 1,
                "assignment_duration": 300.0
            }
        ]
    }
}
```

### 奖励函数
```python
# 多目标奖励计算
total_reward = (
    0.4 * tracking_performance_reward +    # 跟踪性能 (40%)
    0.3 * resource_efficiency_reward +     # 资源效率 (30%)
    0.3 * mission_completion_reward        # 任务完成 (30%)
) - penalty_terms
```

## 输出数据

### JSON格式 (人类可读)
```json
{
  "metadata": {
    "dataset_type": "training",
    "generation_time": "2025-07-31T10:00:00",
    "total_episodes": 100,
    "total_data_points": 50000
  },
  "episodes": [
    {
      "episode_id": "episode_001",
      "scenario_type": "multiple_threats",
      "total_reward": 85.6,
      "success": true,
      "data_points": [...]
    }
  ]
}
```

### HDF5格式 (高效存储)
```
rlhf_data.h5
├── metadata/
├── episodes/
│   ├── episode_000/
│   │   ├── states (array)
│   │   ├── actions (array)
│   │   └── rewards (array)
└── statistics/
```

### NumPy格式 (机器学习)
```python
data = np.load('rlhf_data.npz')
states = data['states']        # (N, state_dim)
actions = data['actions']      # (N, action_dim)
rewards = data['rewards']      # (N,)
```

## 场景类型

### 1. 单一威胁场景
- **目标**：基础跟踪能力训练
- **导弹数量**：1个
- **难度**：简单-中等

### 2. 多威胁场景
- **目标**：资源分配和协调训练
- **导弹数量**：2-8个
- **特点**：同时发射、协调攻击

### 3. 饱和攻击场景
- **目标**：高压力决策训练
- **导弹数量**：10-20个
- **特点**：密集发射、时间窗口短

### 4. 对抗性场景
- **目标**：鲁棒性和适应性训练
- **特点**：规避机动、诱饵部署、电子干扰

## 评估指标

### 性能指标
- **跟踪精度**：目标跟踪准确性
- **覆盖率**：时空覆盖比例
- **响应时间**：检测到跟踪延迟
- **虚警率**：误报频率

### 效率指标
- **功率效率**：单位功耗跟踪效果
- **计算效率**：算法执行时间
- **通信效率**：数据传输效率

### 鲁棒性指标
- **噪声容忍度**：噪声环境性能
- **故障恢复**：设备故障恢复能力
- **适应性**：新场景适应速度

## 配置参数

### 场景多样性配置
```yaml
scenario_diversity:
  threat_scenarios:
    single_threat:
      missile_count: [1, 1]
      threat_level: ["low", "medium", "high"]
    multiple_threats:
      missile_count: [2, 8]
      simultaneous_launches: true
    saturation_attack:
      missile_count: [10, 20]
      launch_time_window: [60, 300]
```

### 数据质量控制
```yaml
data_quality:
  validation_rules:
    position_bounds: true
    velocity_limits: true
    temporal_consistency: true
  anomaly_detection:
    statistical_outliers: true
    physical_constraints: true
```

## 扩展功能

### 1. 自定义奖励函数
```python
def custom_reward_function(state, action, base_data):
    # 实现自定义奖励逻辑
    return reward_value
```

### 2. 自定义状态编码
```python
def custom_state_encoder(base_data):
    # 实现自定义状态编码
    return encoded_state
```

### 3. 专家演示数据
```python
# 集成专家策略
expert_action = expert_policy.get_action(state)
data_point = rlhf_collector.collect_rlhf_data_point(expert_action)
```

## 故障排除

### 常见问题

1. **STK连接失败**
   - 检查STK软件是否正常运行
   - 验证COM接口权限

2. **数据采集异常**
   - 检查配置文件格式
   - 验证时间管理器设置

3. **内存不足**
   - 减少并发场景数量
   - 使用HDF5格式存储大数据集

### 日志分析
```bash
# 查看详细日志
tail -f rlhf_data_collection.log

# 过滤错误信息
grep "ERROR" rlhf_data_collection.log
```

## 性能优化

### 1. 并行处理
```python
# 多进程数据采集
import multiprocessing
pool = multiprocessing.Pool(processes=4)
```

### 2. 内存管理
```python
# 批量保存数据
if len(collected_data) >= batch_size:
    save_batch_data(collected_data)
    collected_data.clear()
```

### 3. 数据压缩
```python
# 使用压缩存储
with h5py.File('data.h5', 'w', compression='gzip') as f:
    f.create_dataset('states', data=states, compression='gzip')
```

## 贡献指南

1. Fork项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request

## 许可证

本项目遵循相关开源许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues
- 技术文档
- 开发团队

---

**注意**：本系统基于现有STK数据采集系统扩展，需要先确保基础系统正常运行。
