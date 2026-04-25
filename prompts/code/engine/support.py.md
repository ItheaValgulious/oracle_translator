# support.py

这个文件实现 `fixpoint/platform` 支撑网络。

## 对外接口

- 常量:
  - `SUPPORT_TIMEOUT_SECONDS`
  - `SUPPORT_SOURCE_VALUE`
  - `SUPPORT_FAILURE_THRESHOLD`
  - `INTEGRITY_DECAY_UNSUPPORTED`
- `apply_support(grid, registry, dt)`
  - 按一格一帧传播支撑波次,更新 `support_value`,并在超时失撑时逐步降低 `integrity`。

## 依赖的对外接口

- `engine.grid.Grid`
- `engine.types.CellFlag`
- `engine.types.MaterialRegistry`

## 主要功能

- 让 `fixpoint` 持续注入支撑。
- 每个模拟步只把支撑波次向相邻 `platform` 推进一格,远距离支撑靠多帧抵达。
- `generation` 记录已收到的最新波次 id,只有比当前格更新的邻居波次才能刷新支撑。
- `support_value` 记录距离失撑还剩多少秒,收到新波次时重置为 10 秒。
- 断链后没有新波次进入的结构只会消耗本地倒计时,不会在孤岛内永久互相续命。
- 当支撑不足时,让结构先掉完整度,而不是瞬间转成粉体。
