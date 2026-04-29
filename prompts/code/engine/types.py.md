# types.py

这个文件定义体素引擎原型的核心类型系统。

## 对外接口

- 枚举:
  - `MatterState`
  - `ReactionKind`
  - `LifetimeMode`
  - `CellFlag`
- 数据类:
  - `PhaseRule`
  - `VariantDef`
  - `MaterialFamily`
  - `MaterialRegistry`
  - `CellState`
- `empty_cell()`
  - 构造一个空格子状态。

## 依赖的对外接口

- Python 标准库 `dataclasses`
- Python 标准库 `enum`

## 主要功能

- 把静态材质定义和动态格子状态分层。
- 用 `MatterState` 只表达 `solid/liquid/gas` 三种物态,不再额外维护 `MotionMode` 这类第二层运动分类。
- 让 `VariantDef` 通过通用运动参数描述同一物态内部的差异,包括:
  - `mobility`
  - `pressure_response`
  - `gravity_scale`
  - `buoyancy_scale`
  - `thermal_motion_scale`
  - `wind_coupling`
  - `wind_vertical_factor`
  - `downward_blocked_diagonal_fallback`
  - `velocity_decay`
- 让求解器只按 `matter_state` 分支,其余行为全部查 `VariantDef` 参数。
- `VariantDef` 同时承载 `reaction_strength` 和 `reaction_energy`,分别表达非热反应强度与反应吸放热。
- `VariantDef` 也承载导热加速倍率 `liquid_contact_heat_exchange_multiplier` 与 `same_variant_heat_exchange_multiplier`,用于数据驱动的热交换增强。
- 给 `support_value`、`generation`、`blocked_impulse`、相变规则和反应语义提供统一类型。
- 明确 `fixpoint` 是标志位而不是独立材质。
