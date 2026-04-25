# materials.py

这个文件定义第一版体素引擎原型使用的材质族表。

## 对外接口

- `build_material_registry()`
  - 返回完整的 `MaterialRegistry`。

## 依赖的对外接口

- `engine.types.MaterialFamily`
- `engine.types.MaterialRegistry`
- `engine.types.MatterState`
- `engine.types.MotionMode`
- `engine.types.PhaseRule`
- `engine.types.ReactionKind`
- `engine.types.VariantDef`

## 主要功能

- 定义 `stone/sand/glass/iron/water/acid/poison/tar/fire` 等核心家族。
- 定义每个家族的:
  - 默认变体
  - 坍塌目标
  - 变体静态参数
  - 相变闭环
  - 反应与渲染配置
- 用 `matter_state + motion_mode + density` 组合描述每个变体的运动与交换语义。
- 为反应变体提供“是否在成功反应后保留自身”的静态开关。
- 为反应变体区分“反应效果强度”和“反应热量”两类参数; 其中 `reaction_energy` 用于表达反应时吸热或放热。
- 给 `empty`、`platform/fixpoint` 等固体以及流体都配置导热率,让热量能经空气和材质一起扩散。
- `empty` 当前使用较高于早期版本的导热率,避免热空气在热力图上看起来像几乎不和周围空气交换。
- `empty` 当前也作为空气格参与气体式运动,用于表达热空气上浮和空气布朗运动。
- `base_temperature` 只表示创建 cell 时的默认初始温度,不表示已有 cell 的温度回归目标。
- 水家族当前用相变滞后和较高等效热容量表达潜热: 蒸汽在普通空气中不会离开热源几帧内立刻凝结,冰在常温空气中不会几帧内立刻融化。
- 当前水家族阈值采用滞后:
  - `water -> ice` 低于约 -3 度
  - `ice -> water` 高于约 8 度
  - `steam -> water` 低于约 65 度
- `fire` 当前通过 `reaction_energy` 在每步先加热自身,再由热传导把热量扩散到周围空气和材质。
- 让 `engine` 其他模块通过查表获得静态材质参数,而不是把这些参数复制到每个 cell。
