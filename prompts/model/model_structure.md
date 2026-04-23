# Model Structure

本文针对 [model_socket.md](/mnt/d/project/oracle_translator/prompts/model/model_socket.md), 定义第一版模型结构.

## 1. 目标

第一版模型不是直接预测完整 runtime 字段.
而是预测一个更收缩的 `model socket`.

也就是:

- `material_template`
- `reaction_template`
- `release_template`
- `motion_template`
- `motion_direction`
- `origin`
- `target`
- `politeness`

## 2. Backbone

第一版仍推荐:

- `Qwen3-0.6B-Base`

原因:

- 工程实现简单
- 本地部署容易
- 做模板分类足够用

## 3. 输入

当前阶段只处理完整句子.

输入:

- `completed_utterance_text`

不训练:

- `backfire`
- `unstable`
- prefix 完成度判断

## 4. 输出 heads

### 4.1 Template Heads

模型应直接输出这些分类头:

- `material_template_head`
- `reaction_template_head`
- `release_template_head`
- `motion_template_head`
- `motion_direction_head`
- `origin_head`
- `target_head`

### 4.2 Style Head

风格头只保留:

- `politeness_head`

推荐:

- 做二分类:
  - `0`
  - `1`
- 训练时用 BCE 或 2-class CE
- 推理时取概率作为连续 `politeness`

## 5. 表征方式

仍建议:

- backbone 最后 4 层做 layer mix
- `z_last + z_pool` 做全局表示
- 各分类头直接接 `z_global`

因为第一版头数已经明显收缩, 不需要像旧版那样每个小字段都做一套复杂 slot attention.

如果后续发现:

- `material_template`
- `reaction_template`
- `motion_template`

之间干扰严重, 再给这些头补 slot-specific pooling.

## 6. Loss

第一版总 loss 可写为:

```text
L = λ_material * L_material
  + λ_reaction * L_reaction
  + λ_release * L_release
  + λ_motion_template * L_motion_template
  + λ_motion_direction * L_motion_direction
  + λ_origin * L_origin
  + λ_target * L_target
  + λ_politeness * L_politeness
```

推荐:

- 全部模板头: cross entropy
- `politeness`: BCE loss 或 2-class cross entropy

## 7. 执行链路

执行时顺序应为:

1. 模型输出 `model socket`
2. 模板展开器把:
   - `material_template`
   - `reaction_template`
   - `release_template`
   - `motion_template`
   等展开成完整 runtime 结构
3. 展开器再根据:
   - `motion_direction`
   - `origin`
   - `target`
   写入完整 runtime
4. 游戏侧消费展开后的完整 `magic_socket`

## 8. 为什么这样更稳

因为第一版最难的不是语言理解本身, 而是:

- 字段太多
- 字段边界太细
- reaction 执行逻辑太复杂
- release 数值细节也没必要第一版直接学

先做模板选择, 再做模板展开, 更符合第一版的工程分层.
