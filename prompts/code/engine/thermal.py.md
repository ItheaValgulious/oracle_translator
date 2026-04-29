# thermal.py

这个文件实现热传导。

## 对外接口

- `apply_thermal(grid, registry, dt)`
  - 更新相邻格子的热交换。

## 依赖的对外接口

- `engine.atmosphere.AMBIENT_AIR_RESTORE_RATE`
- `engine.atmosphere.ambient_air_temperature_for_row`
- `engine.grid.Grid`
- `engine.types.MaterialRegistry`
- `engine.types.MatterState`

## 主要功能

- 按变体的导热率和热容量做局部热交换,包括承重固体、液体、气体和 `empty` 空空气。
- 允许热量通过空气和固体一起传播。
- 空空气在相邻导热之后,还会缓慢向“近地略暖、高空略冷”的背景环境温度回归。
- 相界面传热加速不再硬编码 `steam` / `water` 组合,而是从变体参数读取:
  - `liquid_contact_heat_exchange_multiplier`
  - `same_variant_heat_exchange_multiplier`
- 当前实现里:
  - 气体和液体相邻时,会读取该气体变体的 `liquid_contact_heat_exchange_multiplier`
  - 同一变体的气体团内部,会读取双方 `same_variant_heat_exchange_multiplier` 的较大值
- 这样蒸汽界面冷却、蒸汽团内部热扩散等行为都由材质表控制,不需要在热传导求解器里再写材质名特判。
- `base_temperature` 只作为创建 cell 时的初始温度来源,热传导不会把已有 cell 拉回材质基础温度。
- 让不同材质以不同速度升温和降温,并为相变和结构热损伤提供温度来源。
