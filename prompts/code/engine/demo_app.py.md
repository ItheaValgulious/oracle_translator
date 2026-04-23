# demo_app.py

这个文件提供第一版可运行体素 demo 的桌面应用壳。

## 对外接口

- `ToolSpec`
  - 定义一个绘制工具的家族、变体和附加覆盖。
- `VoxelDemoWindow`
  - pyglet 窗口,负责交互、调度和绘制。
- `run_demo()`
  - 启动 demo。

## 依赖的对外接口

- 第三方:
  - `moderngl`
  - `pyglet`
- `engine.grid.create_grid`
- `engine.materials.build_material_registry`
- `engine.render.build_rgba_frame`
- `engine.scenarios.populate_demo_scene`
- `engine.sim.inject_cells`
- `engine.sim.step`
- `engine.types.CellFlag`

## 主要功能

- 建立 `pyglet + ModernGL` 窗口和全屏纹理渲染管线。
- 提供材质笔刷、暂停、单步、重置和清空等交互。
- 将 `engine` 的 CPU 模拟结果实时上传成纹理并显示。
