# support.py

这个文件实现 `fixpoint/platform` 支撑网络。

## 对外接口

- 常量:
  - `SUPPORT_SOURCE_VALUE`
  - `SUPPORT_DIFFUSION_RATE`
  - `SUPPORT_DECAY_RATE`
  - `SUPPORT_FAILURE_THRESHOLD`
  - `INTEGRITY_DECAY_UNSUPPORTED`
- `apply_support(grid, registry, dt)`
  - 更新 `support_value`,并在失撑时逐步降低 `integrity`。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.CellFlag`
- `engine.types.MaterialRegistry`

## 主要功能

- 让 `fixpoint` 持续注入支撑。
- 让只有可承重网络中的格子传播支撑值。
- 当支撑不足时,让结构先掉完整度,而不是瞬间转成粉体。
