# eval_parser.py

这个文件是验证集评测入口脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\eval_parser.py --val data\processed\val_v1.jsonl --checkpoint artifacts\run_name\best_model.pt
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.dataset.SpellDataset`
- `oracle_translator.dataset.SpellCollator`
- `oracle_translator.model.build_model`
- `oracle_translator.train_eval.evaluate_model`

## 主要功能

- 加载 checkpoint.
- 在验证集上运行评测.
- 写出指标文件.
