# smoke_test.py

这个文件是最小前向冒烟脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\smoke_test.py
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.model.build_model`

## 主要功能

- 加载一个可用的 backbone.
- 对两条固定文本跑一次前向.
- 打印前向耗时和各 head 的输出形状.
- 支持通过 `--backbone` 和 `--device` 覆盖默认加载行为.
