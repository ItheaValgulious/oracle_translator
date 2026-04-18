# train_parser.py

这个文件是训练入口脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\train_parser.py --train data\processed\train_v1.jsonl --val data\processed\val_v1.jsonl --output-dir artifacts\run_name
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.train_eval.train_model`

## 主要功能

- 从命令行读取训练参数.
- 调用 `train_model`.
- 打印训练摘要.

