# train_eval.py

这个文件提供训练和评测主流程.

## 对外接口

- `evaluate_model(model, dataloader, device)`
  - 在一个 dataloader 上跑评测.
  - 输出聚合指标字典.

- `train_model(...)`
  - 完整训练入口.
  - 负责:
    - 构造 tokenizer
    - 构造 dataset 和 dataloader
    - 构造模型
    - 训练循环
    - 验证
    - 保存最佳 checkpoint
    - 写出 metrics

## 依赖的对外接口

- `oracle_translator.dataset.SpellDataset`
- `oracle_translator.dataset.SpellCollator`
- `oracle_translator.losses.compute_loss`
- `oracle_translator.metrics.build_prediction_records`
- `oracle_translator.metrics.compute_epoch_metrics`
- `oracle_translator.model.build_model`

## 主要功能

- 提供一个不依赖外部训练框架的最小 PyTorch 训练闭环.
- 按 parse quality 而不是单看 loss 选择最佳 checkpoint.

