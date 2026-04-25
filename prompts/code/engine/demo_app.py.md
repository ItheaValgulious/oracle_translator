# demo_app.py

这个文件提供第一版可运行体素 demo 的桌面应用壳。

## 对外接口

- `ToolSpec`
  - 定义一个绘制工具的家族、变体和附加覆盖。
- `VoxelDemoWindow`
  - pyglet 窗口,负责交互、调度、缩放和绘制。
- `run_demo(grid_width=160, grid_height=96, cell_scale=8, window_width=None, window_height=None, simulation_substeps=2, liquid_brownian_enabled=True, blocked_impulse_enabled=True, vsync=True)`
  - 按指定网格分辨率和初始窗口大小启动 demo。

## 依赖的对外接口

- 第三方:
  - `moderngl`
  - `pyglet`
- `engine.gpu_backend.GpuSimulator`
- `engine.grid.create_grid`
- `engine.materials.build_material_registry`
- `engine.render.build_rgba_frame`
- `engine.scenarios.populate_demo_scene`
- `engine.sim.inject_cells`
- `engine.sim.step`
- `engine.types.CellFlag`

## 主要功能

- 建立 `pyglet + ModernGL` 窗口和全屏纹理渲染管线。
- 提供材质笔刷、`fixpoint` 笔刷、暂停、单步、重置、清空、材质/温度/压力三态视图切换、液体布朗运动运行时开关、子步进调节和窗口缩放等交互。
- `B` 键可在运行时开关液体布朗运动,并把状态同步到 CPU grid 和 GPU backend。
- `I` 键可在运行时开关 `blocked_impulse`,并把状态同步到 CPU grid 和 GPU backend。
- 优先创建 `GpuSimulator`,让 demo 在 GPU 上步进并直接输出调试纹理。
- 当 compute shader 不可用时,回退到 CPU reference 模拟和 `build_rgba_frame()` 上传路径。
- 在 overlay 中实时显示刷新频率、当前 backend、网格分辨率、窗口尺寸、当前视图、液体布朗开关状态、`blocked_impulse` 开关状态和模拟子步进数。
