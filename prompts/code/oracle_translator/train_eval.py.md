# train_eval.py

这个文件提供训练和评测主流程.

## 对外接口

- `evaluate_model(model, dataloader, device)`
  - 在一个 dataloader 上跑评测.
  - 输出聚合指标字典.
  - 包含总 `loss` 和各头 `loss_parts`.

- `train_model(...)`
  - 完整训练入口.
  - 负责:
    - 构造 tokenizer
    - 构造 dataset 和 dataloader
    - 构造模型
    - 接收 `big_head`, `LoRA`, `unfreeze_last_n_layers` 等结构开关
    - 训练循环
    - 验证
    - 输出每个头的 train/val loss
    - 按 `val.loss` 保存最佳 checkpoint
    - 按 `val.loss` 做 early stopping
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
- 每个 epoch 输出 `status`, `categorical`, `binned`, `style`, `reaction_mask` 的 train/val loss.
- 监控 `val.loss`, 当连续若干个 epoch 没有下降时提前停止训练.
- 保存最低 `val.loss` 对应的 `best_model.pt`.
- 在训练摘要中写出 `best_epoch`, `best_val_loss` 和 early stopping 信息.
- 在训练摘要中写出本次使用的 `big_head`, `LoRA`, `train_backbone`, `unfreeze_last_n_layers` 配置.
