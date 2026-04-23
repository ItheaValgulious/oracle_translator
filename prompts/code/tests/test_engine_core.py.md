# test_engine_core.py

这个文件提供第一版体素引擎原型的基础验证。

## 对外接口

- 无库接口。
- 这是 `unittest` 测试文件。

## 依赖的对外接口

- `engine.grid.create_grid`
- `engine.materials.build_material_registry`
- `engine.phases.apply_phase_transitions`
- `engine.sim.inject_cells`
- `engine.sim.step`
- `engine.types.CellFlag`
- `engine.types.CellState`
- `slm.data_generation`
- `slm.io_utils`
- `slm.model_socket_schema`

## 主要功能

- 验证 `slm` 包改名后导入仍可用。
- 验证 `powder` 下落。
- 验证支撑网络断开后结构逐步粉化。
- 验证水相变与 `sand -> molten_glass`。
- 验证 `fire` 使用 `age` 衰减。
- 验证 RGBA 帧输出长度与 demo 场景中关键物质分布。
