# overfit_parser.py

这个文件是小样本过拟合测试入口脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\overfit_parser.py --subset-size 32 --device cuda
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.io_utils.write_jsonl`
- `oracle_translator.train_eval.train_model`

## 主要功能

- 从训练集里抽取一小批 `success` 样本.
- 把同一批样本同时作为 train 和 val, 做可记忆性测试.
- 复用正式训练入口, 输出每个头的 train/val loss.
- 支持独立切换 `big_head`, `LoRA`, `train_backbone`, `unfreeze_last_n_layers`.
- 产出独立的过拟合调试 artifacts.
