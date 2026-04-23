# Engine Prototype Basic

本文定义新的独立体素引擎原型方向。

当前阶段不再使用 Godot 作为第一版运行时壳。
第一版技术栈改为：

- Python
- pyglet
- ModernGL

当前仓库已经落下：

- `engine core`
- 一个最小 `pyglet + ModernGL` 可交互 demo 壳

也就是：

- 核心模拟已经可以独立步进
- demo 可以开窗,实时渲染网格并用笔刷注入材质
- GPU 后端本身仍未实现,当前只是用 ModernGL 显示 CPU 模拟结果

## 1. 当前目标

第一版原型要先验证：

- `8 邻域` 的体素运动
- `fixpoint/platform` 支撑网络
- `platform` 失撑后逐步粉化，而不是整块刚体坍塌
- 每格速度向量与 `blocked_impulse`
- 热传导
- 家族内相变
- `fire/acid/poison/tar` 的基础反应

## 2. CellState

`CellState` 只存每个格子随时间变化的动态状态，不重复存静态材质参数。

固定字段：

- `family_id`
  - 当前材质家族。
- `variant_id`
  - 当前家族内的具体变体。
- `vel_x`
- `vel_y`
  - 当前主运动倾向向量。
- `blocked_x`
- `blocked_y`
  - 未实现运动残差向量。
  - 不是反弹，不是反向冲量。
  - 下一步会和 `vel` 共同决定新的主方向。
- `temperature`
  - 当前温度。
- `support_value`
  - 当前收到的支撑强度。
- `integrity`
  - 当前结构完整度。
- `generation`
  - 当前格子是第几代传播产物。
- `age`
  - 当前形态已持续多久。
  - 当前主要给 `fire` 使用。
- `flags`
  - 至少包含 `is_fixpoint`。

## 3. MaterialFamily / VariantDef

静态材质参数不进 `CellState`，而放到材质表中。

### 3.1 MaterialFamily

- `family_id`
- `name`
- `default_variant`
- `collapse_target`
- `variants`
- `phase_map`
- `reaction_profile`
- `render_profile`

其中：

- `collapse_target`
  - 当承重结构失撑或完整度归零时，转成的同家族粉体或碎屑变体。
- `phase_map`
  - 家族内或跨家族的相变规则。

### 3.2 VariantDef

`VariantDef` 定义一个具体变体的静态模拟参数。

- `variant_id`
- `sim_kind`
  - 例如：
    - `EMPTY`
    - `PLATFORM`
    - `POWDER`
    - `LIQUID`
    - `GAS`
    - `FIRE`
    - `MOLTEN`
- `density`
- `hardness`
- `friction`
- `viscosity`
- `thermal_conductivity`
- `heat_capacity`
- `support_bearing`
- `support_transmission`
- `base_temperature`
- 相变阈值：
  - `ignite_temperature`
  - `melt_temperature`
  - `freeze_temperature`
  - `boil_temperature`
  - `decompose_temperature`
- `integrity_decay_from_heat`
- `reaction_kind`
- `reaction_strength`
- `lifetime_mode`
- `render_color` 或 `palette_id`

## 4. 支撑系统

### 4.1 fixpoint 不是材质

`fixpoint` 不是独立材质，而是一个结构角色标志位。

- 主材质仍然可能是 `stone/glass/iron/...`
- `fixpoint` 通过 `flags` 表示
- 正式游戏里可以做到与普通 `platform` 视觉上不可区分

### 4.2 support_value

支撑信号采用连续值场，但只在承重网络内传播：

- `fixpoint` 持续注入 `support_value`
- 只有 `fixpoint/platform` 网络传播支撑
- `powder/liquid/gas/fire/molten` 不传播支撑

当 `support_value` 低于阈值时：

- 结构不会立刻塌
- 而是开始持续降低 `integrity`
- 当 `integrity` 降到阈值以下时，再转为同家族 `powder`

## 5. blocked_impulse

`blocked_impulse` 表示未实现运动残差。

它的语义是：

- 这格本来想沿某个方向流动
- 但这一轮没有完全释放出去
- 剩下的那部分方向性需求会积累到 `blocked_impulse`

它不是：

- 反向回弹
- 法向反作用力的直接存储

下一步的主方向由以下量合成：

```text
drive = vel + blocked_impulse + 重力/浮力偏置
```

候选方向使用 `8 邻域`，并按与 `drive` 夹角从小到大搜索。

## 6. 首批材质族闭环

第一版核心玩法集固定为：

- `stone_family`
  - `stone_platform`
  - `stone_powder`
  - `magma`
- `sand_family`
  - `sand_powder`
  - 高温转 `glass_family.molten_glass`
- `glass_family`
  - `glass_platform`
  - `glass_shards`
  - `molten_glass`
- `iron_family`
  - `iron_platform`
  - `iron_grit`
  - `molten_iron`
- `water_family`
  - `ice`
  - `water`
  - `steam`
- `acid_family`
  - `acid_liquid`
  - `acid_gas`
- `poison_family`
  - `poison_liquid`
  - `poison_gas`
- `tar_family`
  - `tar_liquid`
  - `tar_smoke`
- `fire_family`
  - `fire`

固定相变闭环：

- `stone -> magma -> stone`
- `sand -> molten_glass -> glass`
- `glass <-> molten_glass`
- `iron <-> molten_iron`
- `water <-> ice <-> steam`
- `acid_liquid -> acid_gas`
- `poison_liquid -> poison_gas`
- `tar_liquid -> tar_smoke/fire`

## 7. 反应语义

- `fire`
  - 使用 `age`
  - 作为加热源
- `acid`
  - 主打腐蚀
  - 可让结构掉 `integrity`
- `poison`
  - 主打毒液与毒雾扩散
  - 受热后可分解成更危险的毒雾与短火团
- `tar`
  - 高黏液体
  - 易燃

结构材质当前遵循：

- `stone/glass/iron/ice` 不持续燃烧
- 它们只会受热损伤或相变
- 也就是“先烧后塌”，不是着火瞬间失撑

## 8. 当前 demo 交互

当前 demo 已具备:

- 打开桌面窗口
- 实时步进模拟
- 鼠标绘制/擦除
- 切换 `stone/fixpoint/sand/water/fire/acid/tar/poison/ice`
- 暂停、单步、重置和清空场景

## 9. 第一版不做什么

- 不做 `fill_ratio`
- 不做整块刚体坍塌
- 不做 Godot 集成
- 不做实时模型 socket 接入
- 不做真正的 GPU 模拟后端

当前阶段是：

- CPU 做模拟
- `pyglet + ModernGL` 只做 demo 壳与显示
