# demo_app.py

这个文件提供第一版可运行体素 demo 的桌面应用壳。

## 对外接口

- `ToolSpec`
  - 定义一个绘制工具的家族、变体和附加覆盖。
- `VoxelDemoWindow`
  - pyglet 窗口,负责相机、交互、调度、缩放和绘制。
- `run_demo(grid_width=160, grid_height=96, world_width=None, world_height=None, halo_cells=64, page_shift_cells=64, cell_scale=8, window_width=None, window_height=None, simulation_substeps=2, liquid_brownian_enabled=True, blocked_impulse_enabled=True, directional_fallback_enabled=True, vsync=True)`
  - 按指定 viewport/world 分辨率和初始窗口大小启动 demo。

## 依赖的对外接口

- 第三方:
  - `moderngl`
  - `pyglet`
- `engine.gpu_backend.ComputeBackendUnavailable`
- `engine.materials.build_material_registry`
- `engine.render.DebugViewMode`
- `engine.scenarios.populate_demo_scene`
- `engine.types.CellFlag`
- `engine.world.ActiveWorldWindow`
- `engine.world.WorldChunkStore`

## 主要功能

- 建立 `pyglet + ModernGL` 窗口和全屏纹理渲染管线。
- 通过 `WorldChunkStore + ActiveWorldWindow` 把“大世界”和“当前活动模拟窗”拆开。
- 片元着色阶段只采样活动窗里对应 viewport 的 UV 子矩形,窗口不再直接等于整张模拟纹理。
- 提供材质笔刷、`fixpoint` 笔刷、暂停、单步、重置、清空、材质/温度/压力三态视图切换、液体布朗运动运行时开关、子步进调节和窗口缩放等交互。
- `W/A/S/D` 和方向键可以在 world space 里平移相机,并在需要时触发活动窗分页平移。
- `B` 键可在运行时开关液体布朗运动,并同步到当前活动模拟窗的 CPU/GPU backend。
- `I` 键可在运行时开关 `blocked_impulse`,并同步到当前活动模拟窗的 CPU/GPU backend。
- `F` 键可在运行时开关“首选方向受阻时是否退而求其次找最近可移动方向”,并同步到当前活动模拟窗的 CPU/GPU backend。
- 优先创建 GPU 活动窗 backend,不可用时回退到 CPU 活动窗路径。
- overlay 当前会实时显示 world 尺寸、活动窗尺寸、viewport 尺寸、相机位置、活动窗原点、运行时物理开关和子步进数。
