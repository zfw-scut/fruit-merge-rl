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

当前实现使用统一节点特征矩阵。所有节点都会带节点类型 one-hot：

```text
is_board_fruit_node
is_queue_fruit_node
is_action_node
is_global_node
is_boundary_node
```

下表只列出各类节点实际填充的业务特征。未使用的通用字段保持为 `0`。

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
| stable | 是否基本稳定 |
| distance_to_left_wall | 到左墙距离 |
| distance_to_right_wall | 到右墙距离 |
| distance_to_floor | 到底部距离 |
| distance_to_danger_line | 到死亡线距离 |

已移除的弱特征：

```text
angle_sin
angle_cos
angular_velocity
age
```

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
| is_current_queue_fruit | 是否为 q0 |

说明：queue_fruit_node 没有真实空间位置，不作为地图内物理水果处理。

---

### 4.3 action_node

表示一个候选投放位置。

| 状态量 | 含义 |
|---|---|
| x | 候选投放横坐标 |
| action_index | 动作编号 |
| level | q0 的水果等级 |
| radius | q0 的水果半径 |

---

### 4.4 global_node

表示全局游戏状态。

| 状态量 | 含义 |
|---|---|
| max_height | 当前最高堆叠高度 |
| fruit_count | 地图内水果数量 |
| max_level | 地图内最高水果等级 |
| empty_space_ratio | 剩余空间比例 |

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
| is_left_wall | 是否为左墙 |
| is_right_wall | 是否为右墙 |
| is_floor | 是否为底部地板 |
| is_danger_line | 是否为死亡线 |
| x | 边界参考横坐标 |
| y | 边界参考纵坐标 |
| boundary_position | 边界在线性方向上的位置 |

---

## 5. 边类型与边状态量

当前实现使用统一边特征矩阵。所有边都会带边类型 one-hot：

```text
is_board_fruit_pair_edge
is_action_board_fruit_edge
is_queue_fruit_order_edge
is_queue_board_fruit_edge
is_action_queue_fruit_edge
is_board_boundary_edge
is_global_edge
```

下表只列出各类边实际填充的业务特征。未使用的通用字段保持为 `0`。

### 5.1 board_fruit ↔ board_fruit

表示地图内水果之间的空间、碰撞、合成关系。

| 状态量 | 含义 |
|---|---|
| dx | 横向相对距离 |
| dy | 纵向相对距离 |
| distance | 中心距离 |
| horizontal_distance | 横向距离绝对值 |
| vertical_distance | 纵向距离绝对值 |
| radius_sum | 半径和 |
| overlap_margin | 半径和与中心距离之差 |
| level_diff | 等级差 |
| abs_level_diff | 等级差绝对值 |
| same_level | 是否同等级 |
| relative_vx | 相对水平速度 |
| relative_vy | 相对垂直速度 |

---

### 5.2 action ↔ board_fruit

表示候选投放动作与地图内水果之间的潜在影响关系。

| 状态量 | 含义 |
|---|---|
| dx | 场上水果相对候选落点的横向有符号距离 |
| dy | 场上水果相对生成线的纵向有符号距离 |
| horizontal_distance | 横向距离 |
| vertical_distance | 纵向距离 |
| radius_sum | q0 与已有水果半径和 |
| path_overlap_margin | 投放路径和场上水果横向范围的重叠余量 |
| level_diff | q0 与已有水果等级差 |
| abs_level_diff | 等级差绝对值 |
| same_level | 是否同等级 |
| is_under_drop_path | 已有水果是否接近投放路径 |

---

### 5.3 queue_fruit → queue_fruit

表示待投放水果之间的时间顺序关系。

| 状态量 | 含义 |
|---|---|
| order_gap | 队列顺序间隔 |
| is_next | 是否为相邻下一个水果 |
| level_diff | 等级差 |
| abs_level_diff | 等级差绝对值 |
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

---

### 5.5 action ↔ queue_fruit

表示候选动作与待投放序列之间的规划关系。

| 状态量 | 含义 |
|---|---|
| queue_index | 队列位置 |

---

### 5.6 board_fruit ↔ boundary

表示水果与边界之间的位置 / 风险关系。

| 状态量 | 含义 |
|---|---|
| distance_to_boundary | 水果到边界距离 |
| is_near_boundary | 是否接近边界 |

---

### 5.7 global ↔ all

表示全局状态与其他节点之间的信息关系。

当前 `global ↔ all` 边只使用边类型 one-hot，不额外填充业务特征。

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

## 7. 当前最小 GNN-Q 模型

当前模型位于：

```text
src/daxigua_rl/models/gnn_q.py
```

模型输入：

```text
node_features        [num_nodes, 28]
edge_index           [2, num_edges]
edge_features        [num_edges, 26]
action_node_indices  [action_count]
```

模型结构：

```text
node_features -> node_encoder -> node_hidden
edge_features -> edge_encoder -> edge_hidden

message passing x 3:
    message = MLP([source_node, target_node, edge])
    aggregate = mean(messages_to_target)
    node = LayerNorm(node + update_mlp([node, aggregate]))

action_hidden = node_hidden[action_node_indices]
q_values = q_head(action_hidden)
```

默认参数：

```text
hidden_dim = 128
message_layers = 3
aggregation = mean
activation = SiLU
```

输出：

```text
q_values.shape = [action_count]
```

说明：

- 当前是统一图 message passing，不使用真正的异构图算子。
- 节点类型和边类型通过 one-hot 特征提供给模型。
- 当前模型只用于跑通前向链路，尚未接入 replay buffer、target network 或训练循环。

---

## 8. 经验状态记录

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
