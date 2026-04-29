# run_engine_demo.py

这个脚本是第一版体素 demo 的命令行入口。

## 如何使用

```bash
.env\Scripts\python.exe scripts/run_engine_demo.py --grid-width 200 --grid-height 120 --window-width 1440 --window-height 900 --substeps 3 --no-liquid-brownian --no-blocked-impulse --no-directional-fallback
```

## 对外接口

- 无库接口。
- 这是命令行入口程序。
- `parse_args(argv=None)`
  - 解析 demo 的分辨率、窗口尺寸、子步进、液体布朗运动开关、`blocked_impulse` 开关、方向 fallback 开关和 `vsync` 参数。

## 依赖的对外接口

- `engine.demo_app.run_demo`

## 主要功能

- 把 `src` 加入 `sys.path`
- 启动 `pyglet + ModernGL` 体素 demo
- demo 默认优先使用 GPU compute backend,不可用时自动回退 CPU reference
- 允许通过命令行参数配置初始网格分辨率、窗口大小、每帧模拟子步进数以及是否默认关闭液体布朗运动、`blocked_impulse` 和方向 fallback
