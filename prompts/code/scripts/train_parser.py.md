# train_parser.py

这个文件是训练入口脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\train_parser.py --train data\processed\train_v1.jsonl --val data\processed\val_v1.jsonl --output-dir artifacts\run_name --max-epochs 300 --patience 3 --min-delta 0.0 --big-head --lora-rank 8 --unfreeze-last-n-layers 8
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.train_eval.train_model`

## 主要功能

- 从命令行读取训练参数.
- 支持用 `--max-epochs`, `--patience`, `--min-delta` 控制基于 `val.loss` 的 early stopping.
- 支持独立配置:
  - `--big-head`
  - `--lora-rank/--lora-alpha/--lora-dropout/--lora-target-modules`
  - `--train-backbone/--unfreeze-last-n-layers`
- 训练时打印每个头的 `train_loss` 和 `val_loss`.
- 调用 `train_model`.
- 打印训练摘要.
