# materials.py

这个文件定义第一版体素引擎原型使用的材质族表。

## 对外接口

- `build_material_registry()`
  - 返回完整的 `MaterialRegistry`。

## 依赖的对外接口

- `engine.types.MaterialFamily`
- `engine.types.MaterialRegistry`
- `engine.types.MatterState`
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
- 用 `_default_motion_profile()` 按 `matter_state + support_bearing` 生成默认运动参数,再写入 `VariantDef`。
- 默认运动参数不再使用 `powder/static/fluid` 这样的离散标签,而改成通用字段组合:
  - `mobility`
  - `pressure_response`
  - `gravity_scale`
  - `buoyancy_scale`
  - `thermal_motion_scale`
  - `wind_coupling`
  - `wind_vertical_factor`
  - `downward_blocked_diagonal_fallback`
  - `velocity_decay`
- 允许单个变体覆盖默认运动参数,方便以后扩展新的固体/液体/气体而不改求解器分支。
- 为反应变体提供“是否在成功反应后保留自身”的静态开关。
- 为反应变体区分“反应效果强度”和“反应热量”两类参数; 其中 `reaction_energy` 用于表达反应时吸热或放热。
- 给 `empty`、承重固体以及流体都配置导热率,让热量能经空气和材质一起扩散。
- `empty` 的 `base_temperature` 只定义环境空气的中性基准温度; 真实空域会在运行时叠加背景层结。
- `base_temperature` 只表示创建 cell 时的默认初始温度,不表示已有 cell 的温度回归目标。
- 水家族通过 `liquid_contact_heat_exchange_multiplier` 和 `same_variant_heat_exchange_multiplier` 表达蒸汽界面传热加速与蒸汽团内热扩散加速,不再把 `steam` 特判写在热传导求解器里。
