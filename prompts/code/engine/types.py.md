# types.py

这个文件定义体素引擎原型的核心类型系统。

## 对外接口

- 枚举:
  - `MatterState`
  - `MotionMode`
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
- 把“物态”和“运动方式”拆开,不再把它们混在一个旧的粗分类字段里。
- 给 `support_value` 支撑新鲜度、`generation` 支撑波次、`blocked_impulse`、相变规则和反应语义提供统一类型。
- `VariantDef` 当前同时承载 `reaction_strength` 和 `reaction_energy`,分别表达非热反应强度与反应吸放热。
- 让变体可以声明“反应后是否保留自身”等静态反应语义。
- 明确 `fixpoint` 是标志位而不是独立材质。
