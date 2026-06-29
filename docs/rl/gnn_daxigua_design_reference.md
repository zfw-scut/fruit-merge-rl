# GNN 状态图设计参考

## 1. 总体目标

采用异构图表示游戏状态：

```text
Graph(state) → [Q(s,a0), Q(s,a1), ..., Q(s,aK)]
```

即：输入当前状态图，输出所有候选投放动作的价值。

---

## 2. 状态组成

```text
state = {
    board_fruits,      # 地图内已有水果
    fruit_queue,       # 待投放水果序列，N = 4
    action_candidates, # 候选投放动作
    global_state,      # 全局状态
    boundary_state     # 地图边界
}
```

---

## 3. 节点类型

| 节点类型 | 含义 |
|---|---|
| board_fruit_node | 地图内已有水果 |
| queue_fruit_node | 待投放水果序列 |
| action_node | 候选投放动作 |
| global_node | 全局状态 |
| boundary_node | 地图边界 |

---

## 4. 节点状态量

### 4.1 board_fruit_node

表示地图中真实存在的水果。

| 状态量 | 含义 |
|---|---|
| x | 水果中心横坐标 |
| y | 水果中心纵坐标 |
| vx | 水平速度 |
| vy | 垂直速度 |
| level | 水果等级 |
| radius | 水果半径 |
| age | 水果存在时间 |
| stable_flag | 是否稳定 |
| distance_to_left_wall | 到左墙距离 |
| distance_to_right_wall | 到右墙距离 |
| distance_to_floor | 到底部距离 |
| distance_to_danger_line | 到死亡线距离 |

---

### 4.2 queue_fruit_node

表示待投放水果序列。

```text
fruit_queue = [q0, q1, q2, q3]
```

| 队列元素 | 含义 |
|---|---|
| q0 | 当前即将投放的水果 |
| q1 | 下一颗水果 |
| q2 | 下下颗水果 |
| q3 | 下下下颗水果 |

单个 queue_fruit_node 状态量：

| 状态量 | 含义 |
|---|---|
| level | 水果等级 |
| radius | 水果半径 |
| queue_index | 队列位置 |
| is_current | 是否为 q0 |

说明：queue_fruit_node 没有真实空间位置，不作为地图内物理水果处理。

---

### 4.3 action_node

表示一个候选投放位置。

| 状态量 | 含义 |
|---|---|
| drop_x | 投放横坐标 |
| action_index | 动作编号 |
| current_level | q0 的水果等级 |
| current_radius | q0 的水果半径 |

---

### 4.4 global_node

表示全局游戏状态。

| 状态量 | 含义 |
|---|---|
| score | 当前分数 |
| max_height | 当前最高堆叠高度 |
| fruit_count | 地图内水果数量 |
| max_level | 地图内最高水果等级 |
| danger_height | 危险高度 / 死亡线高度 |
| empty_space_ratio | 剩余空间比例 |
| step_count | 当前步数 |

---

### 4.5 boundary_node

表示地图边界。

| 节点 | 含义 |
|---|---|
| left_wall | 左边界 |
| right_wall | 右边界 |
| floor | 底部边界 |
| danger_line | 死亡线 |

单个 boundary_node 状态量：

| 状态量 | 含义 |
|---|---|
| boundary_type | 边界类型 |
| position | 边界位置 |

---

## 5. 边类型与边状态量

### 5.1 board_fruit ↔ board_fruit

表示地图内水果之间的空间、碰撞、合成关系。

| 状态量 | 含义 |
|---|---|
| dx | 横向相对距离 |
| dy | 纵向相对距离 |
| distance | 中心距离 |
| radius_sum | 半径和 |
| overlap_margin | 半径和与中心距离之差 |
| level_diff | 等级差 |
| abs_level_diff | 等级差绝对值 |
| same_level | 是否同等级 |
| relative_vx | 相对水平速度 |
| relative_vy | 相对垂直速度 |
| approaching_speed | 相互接近速度 |

---

### 5.2 action ↔ board_fruit

表示候选投放动作与地图内水果之间的潜在影响关系。

| 状态量 | 含义 |
|---|---|
| drop_x_minus_fruit_x | 投放横坐标与水果横坐标之差 |
| horizontal_distance | 横向距离 |
| vertical_distance | 纵向距离 |
| distance | 假想投放水果与已有水果距离 |
| level_diff | q0 与已有水果等级差 |
| abs_level_diff | 等级差绝对值 |
| same_level | 是否同等级 |
| radius_sum | q0 与已有水果半径和 |
| is_under_drop_path | 已有水果是否接近投放路径 |

---

### 5.3 queue_fruit → queue_fruit

表示待投放水果之间的时间顺序关系。

| 状态量 | 含义 |
|---|---|
| order_gap | 队列顺序间隔 |
| is_next | 是否为相邻下一个水果 |
| level_diff | 等级差 |
| same_level | 是否同等级 |

---

### 5.4 queue_fruit ↔ board_fruit

表示未来水果与地图内水果之间的等级匹配关系。

| 状态量 | 含义 |
|---|---|
| queue_index | 待投放水果队列位置 |
| level_diff | 队列水果与地图水果等级差 |
| abs_level_diff | 等级差绝对值 |
| same_level | 是否同等级 |
| board_fruit_x | 地图水果横坐标 |
| board_fruit_y | 地图水果纵坐标 |

---

### 5.5 action ↔ queue_fruit

表示候选动作与待投放序列之间的规划关系。

| 状态量 | 含义 |
|---|---|
| action_index | 动作编号 |
| queue_index | 队列位置 |
| is_current | 是否为 q0 |
| level_diff_to_current | 与 q0 的等级差 |
| same_as_current | 是否与 q0 同等级 |

---

### 5.6 board_fruit ↔ boundary

表示水果与边界之间的位置 / 风险关系。

| 状态量 | 含义 |
|---|---|
| boundary_type | 边界类型 |
| distance_to_boundary | 水果到边界距离 |
| is_near_boundary | 是否接近边界 |

---

### 5.7 global ↔ all

表示全局状态与其他节点之间的信息关系。

| 状态量 | 含义 |
|---|---|
| node_type | 被连接节点类型 |

---

## 6. 输出

模型输出：

```text
[Q(s,a0), Q(s,a1), ..., Q(s,aK)]
```

| 输出量 | 含义 |
|---|---|
| Q(s,ai) | 当前状态下选择第 i 个投放动作的价值 |

动作选择：

```text
argmax Q(s,ai)
```

---

## 7. 经验状态记录

单条经验状态包含：

| 字段 | 含义 |
|---|---|
| board_fruits | 当前地图内水果状态 |
| fruit_queue | 当前待投放水果序列 |
| action | 当前选择的投放动作 |
| reward | 当前动作获得的奖励 |
| next_board_fruits | 下一状态地图内水果状态 |
| next_fruit_queue | 下一状态待投放水果序列 |
| done | 是否终止 |

队列更新：

```text
[q0, q1, q2, q3] → [q1, q2, q3, new_q]
```
