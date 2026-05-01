# benchmark_world_paging.py

这个脚本提供大世界分页的固定配置 benchmark。

## 如何使用

```bash
.venv\Scripts\python.exe scripts/benchmark_world_paging.py
```

## 对外接口

- 无库接口。
- 这是命令行 benchmark 脚本。

## 依赖的对外接口

- 第三方:
  - `moderngl`
  - `pyglet`
- `engine.materials.build_material_registry`
- `engine.world.ActiveWorldWindow`
- `engine.world.WorldChunkStore`

## 主要功能

- 创建固定的 `2560x1440` world 和 `640x360` viewport。
- 预先用 `populate_demo_scene()` 填充 demo 场景,默认测 populated world,而不是空世界。
- 使用 `halo_cells=32`、`page_shift_cells=16`、`pending_writeback_limit=256` 复现当前分页压力场景。
- 在 GPU 路径下重复横向推动相机,统计分页发生时的:
  - 总切页耗时
  - evict stage
  - overlap copy
  - overlap transient copy
  - incoming load
  - incoming transient clear
  - anchor build
  - anchor upload
- 同时统计整帧链路:
  - `pan_camera`
  - `world.step`
  - `service_background_io`
  - `world.render`
  - frame total
- 在移动阶段结束后,还会继续统计 idle 状态下逐个 pending writeback 的 `service_background_io` 耗时,用于验证“停下后写回”是否会产生新的长阻塞。
- 直接把 count / avg / median / max 打到标准输出,用于对比不同优化轮次。
- 输出里会额外标明 `scene=populate_demo_scene`,避免把空世界 benchmark 结果误当成真实 demo 体验。
