# Magic Runtime Schema B

本文只定义游戏运行时可执行的魔法结构 `B`, 不讨论模型如何解析自然语言, 也不讨论训练方式.  
目标是把任意可识别的咒语, 最终编译为一组可被引擎稳定消费的槽位.

## 1. 总体结构

一个魔法由 5 个部分组成:

1. `subject`: 被召唤或被操作的主体.
2. `release`: 主体被释放到世界中的方式.
3. `motion`: 主体出生后的整体运动规则.
4. `targeting`: 释放起点, 朝向与目标选择规则.
5. `expression`: 本次咒语的表达风格特征.

当前第一阶段只支持:

- `subject_kind = summon_material`

## 2. 顶层字段

```json
{
  "subject_kind": "summon_material",
  "subject": {},
  "release": {},
  "motion": {},
  "targeting": {},
  "expression": {}
}
```

## 3. Subject

`subject` 表示被生成出来并参与模拟的物质主体.

### 3.1 核心字段

- `material_archetype`
  - 可选.
  - 表示一个预定义物质原型.
  - 例如: `holy_fire`, `acid_slime`, `poison_mist`, `ice_shard`.
  - 运行时语义: 原型本质上是一组默认槽位.

- `state`
  - 必填.
  - 取值: `solid`, `liquid`, `gas`.
  - 决定主体的基础体素行为.

- `color`
  - 第一阶段必填.
  - 使用颜色库分类, 不直接输出任意 RGB 或 HEX.
  - 运行时再由颜色库映射到实际渲染颜色.
  - 颜色既影响视觉反馈, 也应作为文本解析的一部分保留.

- `density`
  - 必填.
  - 表示质量与重力响应.
  - 决定沉浮, 冲击感和惯性表现.

- `temperature`
  - 必填.
  - 表示热量等级.
  - 决定与环境或目标接触时可能触发的高温或低温效果.

- `amount`
  - 必填.
  - 表示总生成量.
  - 决定生成体素数量, 持续时间上限或覆盖面积.

### 3.2 反应字段

不要把 `传染` 塞进单一字段, 而是拆为以下 4 个字段:

- `reaction_kind`
  - 可选.
  - 取值示例: `none`, `burn`, `corrode`, `freeze`, `poison`, `grow`.
  - 表示该主体接触其他对象后主要触发的转化类型.

- `reaction_rate`
  - 条件字段.
  - 当 `reaction_kind != none` 时生效.
  - 表示扩散或转化的速度.

- `reaction_mask`
  - 条件字段.
  - 当 `reaction_kind != none` 时生效.
  - 表示该转化允许作用于哪些目标.
  - 取值示例: `solid`, `liquid`, `gas`, `living`, `terrain`.
  - 可为单值或多值集合.

- `reaction_direction`
  - 可选.
  - 表示该反应是否有额外扩散偏置.
  - 取值示例: `none`, `up`, `down`, `forward`, `outward`.

### 3.3 条件字段

以下字段只在特定 `state` 下启用:

- `hardness`
  - 仅 `solid` 生效.
  - 表示固体抵抗破坏和替换的能力.

- `friction`
  - 仅 `solid` 生效.
  - 表示固体表面的摩擦表现.

- `viscosity`
  - 仅 `liquid` 生效.
  - 表示液体的流动阻力和附着倾向.

## 4. Release

`release` 表示主体出生时如何被抛出或铺开.

### 4.1 字段

- `release_profile`
  - 必填.
  - 取值示例: `burst`, `stream`, `spray`, `pool`, `beam`.
  - 表示体素生成的空间形状和初始发射方式.

- `release_speed`
  - 必填.
  - 表示主体出生瞬间获得的初始速度.

- `release_spread`
  - 可选.
  - 表示初始扩散角度或扩散宽度.

- `release_duration`
  - 可选.
  - 表示持续喷射或持续生成的时长.
  - 对 `stream`, `beam` 这类持续型释放尤其重要.

## 5. Motion

`motion` 表示主体在出生之后, 如何受一个统一的运动规则影响.

### 5.1 字段

- `motion_template`
  - 必填.
  - 取值示例: `fixed`, `flow`, `vortex`.
  - 表示整体运动场模板.

- `force_strength`
  - 必填.
  - 表示运动场施加给主体的力强度.

- `carrier_velocity`
  - 必填.
  - 表示运动场本身的整体平移速度.
  - 最终粒子受力, 可以理解为 `场内局部力 + 场整体移动`.

- `motion_direction`
  - 可选.
  - 表示运动场偏向的主要方向.
  - 取值示例: `forward`, `backward`, `up`, `down`, `target`, `self`, `none`.

## 6. Targeting

`targeting` 定义魔法从哪里出现, 朝哪里去, 以及目标如何被选中.

### 6.1 字段

- `origin`
  - 必填.
  - 取值示例: `self`, `front_enemy_random`, `back`, `front_up`, `front_down`.
  - 表示释放起点.

- `target_mode`
  - 必填.
  - 取值示例: `aim_enemy`, `aim_self`, `none`.
  - 表示是否存在明确瞄准对象.

- `direction_mode`
  - 必填.
  - 取值示例: `to_target`, `to_self`, `forward`, `none`.
  - 表示释放与运动参考的方向来源.

## 7. Archetype 展开规则

`material_archetype` 不是独立于槽位系统的第二套系统, 它只是槽位默认值模板.

例:

```json
{
  "material_archetype": "holy_fire"
}
```

运行时可展开为:

```json
{
  "state": "gas",
  "temperature": "high",
  "density": "low",
  "amount": "medium",
  "reaction_kind": "burn",
  "reaction_rate": "medium",
  "reaction_mask": ["living", "terrain"]
}
```

如果后续解析结果中还出现明确修饰, 则允许覆盖模板默认值.

例:

- `holy_fire` + `更粘稠` -> 若系统允许, 可强行转向更像燃烧液或凝胶火.
- `acid_slime` + `极寒` -> 允许覆盖温度字段.

## 8. Expression

`expression` 表示一次咒语在语言风格层面的附加特征.  
这些字段默认不直接改变物理执行结果, 但可用于驱动以下系统:

- 吟唱音效与 UI 反馈.
- 施法姿态或特效风格.
- 语义疲劳的风格偏置.
- 误判或反噬时的表现差异.
- 成就, 评分, 流派偏好, NPC 反应.

### 8.1 字段

- `curvature`
  - 必填.
  - 表示表达是否直接, 还是更曲折, 迂回, 修辞化.
  - 建议归一化到 `0.0 - 1.0`.
  - `0.0` 表示非常直白.
  - `1.0` 表示高度委婉或高度修辞化.

- `politeness`
  - 必填.
  - 表示表达的礼貌程度, 克制程度或祈请感.
  - 建议归一化到 `0.0 - 1.0`.
  - `0.0` 表示粗暴, 命令式.
  - `1.0` 表示恭敬, 请愿式.

- `elegance`
  - 必填.
  - 表示表达的高雅程度, 文学性或仪式感.
  - 建议归一化到 `0.0 - 1.0`.
  - `0.0` 表示口语化, 粗粝.
  - `1.0` 表示典雅, 庄重, 仪式化.

### 8.2 设计说明

- `expression` 是解析器输出的一部分, 但不应成为第一阶段物理施法成功的硬前置条件.
- 当 `subject/release/motion/targeting` 已足够完整时, 即使 `expression` 不稳定, 也不应阻止施法.
- `expression` 更适合作为风格层和反馈层的控制量, 而不是底层体素物理参数.

## 9. 第一阶段建议的最小必填集合

为了保证第一阶段可执行, 每个成功施法的运行时结果至少应包含:

- `subject.color`
- `subject.state`
- `subject.density`
- `subject.temperature`
- `subject.amount`
- `release.release_profile`
- `release.release_speed`
- `motion.motion_template`
- `motion.force_strength`
- `motion.carrier_velocity`
- `targeting.origin`
- `targeting.target_mode`
- `targeting.direction_mode`

如果依赖反应机制, 则至少还需要:

- `subject.reaction_kind`
- `subject.reaction_rate`
- `subject.reaction_mask`

第一阶段如需保留风格信息, 还建议输出:

- `expression.curvature`
- `expression.politeness`
- `expression.elegance`

## 10. 不在第一阶段支持的内容

以下内容暂不进入 `B`:

- 对现有玩家或敌人直接施加复杂动作.
- 对场上任意已有物体做精细语义选择.
- 多主体并列召唤.
- 条件语句, 因果链, 多段施法脚本.
- 需要世界知识才能稳定理解的隐喻解释.

第一阶段的 `B` 本质上是:

- 召唤一种物质.
- 以一种方式释放.
- 让它遵循一种运动模板.
- 根据目标规则向外作用.
