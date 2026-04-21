# interactive_parser.py

这个文件是交互式推理入口脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\interactive_parser.py --checkpoint artifacts\qwen06b_v1_val_loss_es\best_model.pt
```

单句测试:

```powershell
.\.env\Scripts\python scripts\interactive_parser.py --text "圣火昭昭, 涤荡前路."
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.model.build_model`
- `oracle_translator.ontology.CATEGORICAL_SPECS`
- `oracle_translator.ontology.BINNED_SPECS`
- `oracle_translator.ontology.STYLE_SPECS`
- `oracle_translator.ontology.REACTION_MASK_LABELS`
- `oracle_translator.ontology.STATUS_LABELS`

## 主要功能

- 加载本地 backbone 和训练好的 checkpoint.
- 支持 `--text` 单句预测.
- 在不传 `--text` 时进入终端 REPL.
- 始终输出完整的 `runtime_preview`.
- 输出 `status`, `status_confidence`, 各槽位置信度和 reaction mask 分数.
