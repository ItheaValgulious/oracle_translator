# types.py

这个文件定义体素引擎原型的核心类型系统。

## 对外接口

- 枚举:
  - `SimKind`
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
- 给 `support_value`、`blocked_impulse`、相变规则和反应语义提供统一类型。
- 明确 `fixpoint` 是标志位而不是独立材质。
