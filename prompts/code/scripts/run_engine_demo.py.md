# run_engine_demo.py

这个脚本是第一版体素 demo 的命令行入口。

## 如何使用

```bash
.env/bin/python scripts/run_engine_demo.py
```

## 对外接口

- 无库接口。
- 这是命令行入口程序。

## 依赖的对外接口

- `engine.demo_app.run_demo`

## 主要功能

- 把 `src` 加入 `sys.path`
- 启动 `pyglet + ModernGL` 体素 demo
