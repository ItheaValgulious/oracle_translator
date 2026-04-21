# Magic Parser Model Realized

本文记录当前仓库中已经实际实现并跑通过的一版 text -> data -> train -> eval 流程.  
它不是 `model.md` 的完整实现, 而是一个可运行的第一版落地.

## 1. 已实现范围

当前已实现的链路如下:

1. 手工构造 `200` 条高质量 curated 文本样本.
2. 通过 `minimax-m2.5` API 继续扩写出 `1471` 条增广样本.
3. 合并数据, 自动加入 prefix 样本.
4. 使用 `Qwen/Qwen3-0.6B-Base` 作为 backbone.
5. 在 backbone 后接自定义多头分类器.
6. 使用 PyTorch 完成训练和评测.
7. 已经在 CPU 上跑通一轮 quick pipeline.

当前仓库内对应文件:

- 数据定义: `src/oracle_translator/ontology.py`
- 数据生成: `src/oracle_translator/data_generation.py`
- 数据集与 collator: `src/oracle_translator/dataset.py`
- 模型结构: `src/oracle_translator/model.py`
- 损失函数: `src/oracle_translator/losses.py`
- 指标计算: `src/oracle_translator/metrics.py`
- 训练与评测: `src/oracle_translator/train_eval.py`
  - 训练支持按 `val.loss` 做 early stopping, 并保存最低 `val.loss` 的 checkpoint
  - 训练日志会输出 `status`, `categorical`, `binned`, `style`, `reaction_mask` 的 train/val loss
  - 训练结构支持独立切换 `big_head`, `LoRA`, `train_backbone`, `unfreeze_last_n_layers`
- 脚本入口:
  - `scripts/build_curated_dataset.py`
  - `scripts/generate_api_dataset.py`
  - `scripts/prepare_dataset.py`
  - `scripts/train_parser.py`
  - `scripts/eval_parser.py`
  - `scripts/interactive_parser.py`
  - `scripts/overfit_parser.py`
  - `scripts/smoke_test.py`

## 2. 当前模型结构

实际实现的 backbone:

- `Qwen/Qwen3-0.6B-Base`

输入:

- `input_ids: int64[B, T]`
- `attention_mask: int64[B, T]`

实际实现方式:

- 取 backbone 最后 4 层 hidden states.
- 用可学习的 layer mix 做加权求和.
- 再统一 cast 到 `float32`, 避免 CPU 下 `bfloat16` 与 head 的 dtype 冲突.

全局表示:

- `z_last`: 最后有效 token 的表示
- `z_pool`: attention pooling 的全局表示
- `z_global = MLP([z_last ; z_pool])`

槽位表示:

- 每个槽位单独做一个 slot attention pooling.

输出 heads:

- `status_head`
- categorical heads
- binned heads
- style heads
- `reaction_mask_head`

## 3. 当前监督目标

### 3.1 status

3 分类:

- `success`
- `unstable`
- `backfire`

### 3.2 categorical 槽位

当前实现了以下 categorical heads:

- `subject.color`
- `subject.state`
- `subject.reaction_kind`
- `subject.reaction_direction`
- `release.release_profile`
- `motion.motion_template`
- `motion.motion_direction`
- `targeting.origin`
- `targeting.target_mode`
- `targeting.direction_mode`

### 3.3 binned 槽位

以下字段被离散成 7 档:

- `subject.density`
- `subject.temperature`
- `subject.amount`
- `subject.reaction_rate`
- `subject.hardness`
- `subject.friction`
- `subject.viscosity`
- `release.release_speed`
- `release.release_spread`
- `release.release_duration`
- `motion.force_strength`
- `motion.carrier_velocity`

### 3.4 style heads

风格字段被离散成 5 档:

- `expression.curvature`
- `expression.politeness`
- `expression.elegance`

### 3.5 multi-label head

- `subject.reaction_mask`

## 4. 当前损失函数

当前总损失实现为:

```text
L = 1.0 * L_status
  + 1.0 * L_categorical
  + 0.8 * L_binned
  + 0.5 * L_style
  + 0.7 * L_reaction_mask
```

具体损失:

- `status`: cross entropy
- categorical heads: masked cross entropy
- binned heads: masked cross entropy
- style heads: masked cross entropy
- `reaction_mask`: masked BCE with logits

当前没有做的内容:

- 更严格的 ordinal loss
- focal loss
- class-balanced weighting
- calibration-specific objective

## 5. 当前数据情况

当前落盘的数据:

- curated: `200`
- API augment: `1471`

合计原始文本样本:

- `1671`

在 `prepare_dataset.py` 中:

- 会额外把 prefix labels 展开成 `unstable` 样本
- 当前得到:
  - train: `4204`
  - val: `467`

为了快速跑通完整链路, 还额外构造了 quick split:

- `train_quick.jsonl`: `256`
- `val_quick.jsonl`: `64`

### 5.1 当前仓库本地数据现状

当前仓库里实际存在并被这次本地训练直接使用的 `manifest_v1.json` 为:

- `curated_raw = 200`
- `api_raw = 0`
- `train = 490`
- `val = 54`

这意味着当前本地可直接复现的训练规模, 明显小于上面那版包含 API augment 的完整数据描述.
如果直接用当前仓库落盘数据训练, 模型表现应按这组较小数据量来预期, 不能按 `4204/467` 的训练规模估计.

## 6. 已完成的实跑结果

已完成:

- `prepare_dataset.py`
- `smoke_test.py`
- `train_parser.py` on quick split
- `eval_parser.py` on quick split

运行条件:

- device: `cpu`
- backbone frozen
- epochs: `1`
- batch size: `2`
- max length: `96`

quick eval 指标:

- `status_accuracy = 0.8125`
- `success_exact_match = 0.0`
- `categorical_accuracy = 0.4167`
- `binned_accuracy = 0.3371`
- `style_mae = 0.9740`
- `reaction_mask_f1 = 0.7879`

解读:

- `status` 头和 `reaction_mask` 头已经能学到一些东西.
- 细粒度槽位预测仍然偏弱.
- style 预测明显不够好.
- `success_exact_match = 0.0` 表明整套槽位一起命中的能力还远远不够.

### 6.1 最新本地 full split 实跑

基于当前仓库里的本地数据:

- device: `cuda`
- train: `490`
- val: `54`
- early stopping: monitor `val.loss`, `patience=3`, `min_delta=0.0`, `max_epochs=300`

实跑结果:

- 实际停止 epoch: `27`
- 最佳 epoch: `24`
- best `val.loss = 0.9595`
- `status_accuracy = 0.9630`
- `success_exact_match = 0.0625`

逐字段上, 当前模型对 `color` 和 `state` 仍然不够可靠.

## 7. 与 model.md 的差别

### 7.1 已经实现的部分

- 已按 `Qwen3-0.6B-Base + 自定义 heads` 的主线实现.
- 已实现多头监督, 包括 status, 槽位, 风格, reaction mask.
- 已实现 text-only 数据生成, API augment, dataset split, train, eval.

### 7.2 暂未实现的部分

- 没有实现 `Qwen3.5-0.8B-Base` 对照组.
- 没有实现 `Qwen3-Embedding-0.6B` 和 `BGE-M3` baseline.
- 没有实现 `ASR` 接入.
- 没有实现 `TTS -> ASR` 回环数据.
- 没有实现 `LoRA/QLoRA`.
- 没有实现 decoder `KV cache` 的实时前缀推理.
- 没有实现 full-size 训练集实跑.

### 7.3 主动收缩掉的内容

- `color` 已进入第一阶段, 并使用颜色库分类.

- 第一阶段不再单独训练 `confidence_head`.
- 当前代码已改为由 `status` logits 推导 `status_confidence`, 不再把它作为独立监督目标.

- 当前没有实现严格的 `ordinal loss`.
  - 目前先用普通 cross entropy 跑通.
  - 后续如果风格和数值槽位长期不稳, 再升级为 ordinal 训练.

- 当前没有做 judge 过滤的二阶段 teacher pipeline.
  - 目前 API 数据只做了格式层面的恢复和断点续跑.
  - 语义质量控制还不够强.

## 8. 当前最主要的问题

这次实跑暴露出的核心问题不是代码通不通, 而是数据质量:

1. 当前增广更像同义改写, 生活化和语体跨度不足.
2. 风格标签虽然有标, 但训练信号仍然偏弱.
3. success 样本内部模式相对集中, 导致模型容易学到表面模板.
4. 细槽位很多, 但每个槽位真正的语言覆盖还不够宽.

## 9. 下一步最值得做的事

如果继续迭代, 优先级建议如下:

1. 先补数据多样性, 再扩模型.
2. 增加更生活化, 更口语化, 更混杂的表达.
3. 增加更多 near-miss 和 confusing negatives.
4. 对 API 生成样本增加更严格的质量过滤.
5. 在数据变好后, 再尝试:
   - 解冻最后若干层
   - LoRA
   - full training set
   - embedding / encoder baseline 对比
