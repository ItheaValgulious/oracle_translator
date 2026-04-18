# Magic Parser Model Plan

本文定义 `magic_socket.md` 对应的语义解析模型方案.  
目标不是让模型完整理解任意文学表达, 而是把大范围的玩家表达空间 `A`, 映射到较小的运行时魔法空间 `B`, 或映射到 `backfire`.

## 1. 设计目标

第一阶段模型必须满足以下目标:

1. 支持诗性, 中二, 仪式化表达.
2. 不要求理解所有表达, 允许低置信输入触发 `backfire`.
3. 本地可运行.
4. 可增量处理 `ASR` 的稳定前缀.
5. 最终输出必须直接对应 `magic_socket.md` 中的槽位结构.

## 2. 模型总览

系统分为 4 类模型:

1. `ASR` 模型. 把音频转成稳定前缀文本.
2. `Runtime Parser` 模型. 把文本前缀映射到 `B` 或 `backfire`.
3. `Teacher LLM` 模型. 离线生成训练语料, 不进入实时链路.
4. `TTS` 模型. 离线生成合成语音, 让训练数据更像真实语音输入.

推荐的第一阶段组合如下:

- `ASR`: `paraformer-zh-streaming`, 来自 FunASR.
- `Runtime Parser` 主方案 A: `Qwen3.5-0.8B-Base`, 接自定义分类头.
- `Runtime Parser` 主方案 B: `Qwen3-0.6B-Base`, 接自定义分类头.
- `Runtime Parser` 对照组: `Qwen3-Embedding-0.6B` 或 `BGE-M3`, 接相同分类头.
- `Teacher LLM`: `Qwen3-32B-Instruct` 或同等级更强模型.
- `TTS`: `CosyVoice`.

这里故意保留两个 decoder 主方案:

- `Qwen3.5-0.8B-Base` 的世界知识和风格潜力更强, 但它是带 vision encoder 的 multimodal backbone, 部署和训练实现更复杂.
- `Qwen3-0.6B-Base` 是更干净的 text-only causal LM, 更适合作为工程第一版.

如果目标是尽快做出可跑原型, 先用 `Qwen3-0.6B-Base`.  
如果目标是尽量提高诗性表达理解和风格保留, 再做 `Qwen3.5-0.8B-Base` 对照实验.

## 3. 每个模型的输入输出

### 3.1 ASR

推荐模型: `paraformer-zh-streaming`.

系统级输入:

- `audio_chunk`: `float32[B, samples]`
- 采样率建议 `16kHz`, 单声道.
- 实时流式输入, 建议配合 `VAD`.

系统级输出:

- `partial_text`: 当前 chunk 的实时识别文本.
- `stable_prefix_text`: 已确认较稳定的前缀文本.
- `unstable_suffix_text`: 仍可能被后续修正的尾部文本.
- `timestamps`: 可选.

备注:

- FunASR 官方示例中, `chunk_size=[0,10,5]` 表示约 `600ms` 实时显示粒度, 并带 `300ms` lookahead.
- 如果后续发现 `600ms` 太慢, 可以尝试更短 chunk, 但识别稳定性会下降.

### 3.2 Runtime Parser Backbone

#### 方案 A. `Qwen3.5-0.8B-Base`

推荐用途:

- 做高质量主实验.
- 更看重诗性表达, 风格, 世界知识, 语义泛化半径.

已确认配置:

- 类型: `Causal Language Model with Vision Encoder`
- 语言侧 hidden size: `1024`
- 语言侧层数: `24`
- 语言侧 attention heads: `8` for Q, `2` for KV
- intermediate size: `3584`
- context length: `262144`

模型输入:

- `input_ids`: `int64[B, T]`
- `attention_mask`: `int64[B, T]`
- `past_key_values`: 流式增量推理时启用.
- `output_hidden_states=True`
- 只走 text path, 不喂图像.

模型输出:

- `last_hidden_state`: `float[B, T, 1024]`
- `hidden_states`: 长度为 `L + 1` 的 tuple, 每层形状 `float[B, T, 1024]`
- `past_key_values`: 供下一次前缀更新复用.

#### 方案 B. `Qwen3-0.6B-Base`

推荐用途:

- 作为第一版工程主线.
- 更容易部署, 更容易量化, 也更容易做流式前缀重算.

已确认配置:

- 类型: `Causal Language Model`
- hidden size: `1024`
- 层数: `28`
- attention heads: `16` for Q, `8` for KV
- intermediate size: `3072`
- context length: `32768`

模型输入:

- `input_ids`: `int64[B, T]`
- `attention_mask`: `int64[B, T]`
- `past_key_values`
- `output_hidden_states=True`

模型输出:

- `last_hidden_state`: `float[B, T, 1024]`
- `hidden_states`: 长度为 `L + 1` 的 tuple, 每层形状 `float[B, T, 1024]`
- `past_key_values`

#### 对照组. `Qwen3-Embedding-0.6B`

推荐用途:

- 只作为对照实验.
- 用来验证 decoder backbone 是否真的比 embedding backbone 更擅长保留风格和世界知识.

已确认配置:

- 类型: `Text Embedding`
- hidden size: `1024`
- 层数: `28`
- embedding dimension: `1024`
- context length: `32768`

模型输入:

- `input_ids`: `int64[B, T]`
- `attention_mask`: `int64[B, T]`

模型输出:

- token hidden states, 以及最终 pooled embedding.
- 如果只用 pooled embedding, 会更容易丢失风格信息.
- 如果要认真比较, 应尽量取 token-level hidden states 后再接同构 head.

#### 对照组. `BGE-M3`

推荐用途:

- 只作为 encoder baseline.
- 验证 lexical + dense 混合能力, 是否对神秘学词面更友好.

已确认配置:

- 类型: multilingual embedding model.
- dimension: `1024`
- sequence length: `8192`
- 同时支持 dense, sparse, multi-vector 三种 retrieval 表示.

模型输入:

- `input_ids`: `int64[B, T]`
- `attention_mask`: `int64[B, T]`

模型输出:

- dense embedding.
- sparse lexical weights.
- multi-vector representation.

它更适合做对照组或辅助检索, 不适合作为第一版实时主干.

### 3.3 Teacher LLM

推荐模型: `Qwen3-32B-Instruct` 或同级更强 teacher.

输入:

- 一条 `B` 的 canonical seed.
- 目标风格约束, 如 `curvature`, `politeness`, `elegance`.
- 世界观词表, 禁词表, 可接受误解范围.

输出:

- 正样本咒语文本.
- 近义变体.
- 中二变体.
- 古风或仪式化变体.
- 难负样本.
- `backfire` 负样本.

Teacher LLM 只在离线阶段使用, 不进入实时链路.

### 3.4 TTS

推荐模型: `CosyVoice`.

输入:

- 生成好的咒语文本.
- 说话人 id.
- 语速, 情绪, 音色等可选控制字段.

输出:

- 合成语音波形 `float32[samples]`

用途:

- 生成多说话人, 多语速, 多情绪训练音频.
- 再回灌到 `ASR`, 得到更真实的 noisy transcript.

## 4. Runtime Parser 的最终结构

### 4.1 输入

在每次实时更新时, parser 接收:

- `stable_prefix_text`
- 当前是否已到 `utterance_end`
- 可选的 `speaker_id` 或音色 id, 第一阶段可不接

tokenize 后得到:

- `input_ids`: `int64[B, T]`
- `attention_mask`: `int64[B, T]`

建议:

- `T` 先限制到 `96` 或 `128` tokens.
- 咒语通常很短, 没必要放得太长.

### 4.2 Backbone 输出

记 backbone hidden size 为 `H = 1024`.

- `H_l`: 第 `l` 层 hidden state, 形状 `float[B, T, H]`
- 取最后 4 层做 layer mix:

```text
H_mix = w1 * H_-1 + w2 * H_-2 + w3 * H_-3 + w4 * H_-4
```

其中 `w1..w4` 为可学习标量, softmax 归一化.

### 4.3 全局表示

先构造两种全局向量:

- `z_last = H_mix[:, -1, :]`, 形状 `float[B, 1024]`
- `z_pool = AttnPool(H_mix, attention_mask)`, 形状 `float[B, 1024]`

然后拼接并投影:

```text
z_global = MLP([z_last ; z_pool]) -> float[B, 1024]
```

### 4.4 槽位特定表示

不同槽位关注的 token 往往不同.  
因此不建议所有 head 都只看 `z_global`.  
更适合的做法是给每个槽位一个可学习 query, 做 slot attention:

```text
q_slot[i] in R[1024]
a_slot[i] = softmax(q_slot[i] * H_mix^T)
z_slot[i] = a_slot[i] * H_mix
```

于是每个槽位都有自己的表示 `z_slot[i]`, 形状都是 `float[B, 1024]`.

### 4.5 输出 heads

#### 状态 head

- `status_head(z_global) -> float[B, 3]`
- 3 类:
  - `success`
  - `unstable`
  - `backfire`

含义:

- `success`: 槽位信息足够完整, 可以提交施法.
- `unstable`: 当前前缀信息不足, 先只做预览.
- `backfire`: 低置信或明显不可解释, 直接反噬.

#### 颜色 head

第一阶段应训练 `color_head`.

原因:

- `color` 是玩家表达中非常高频, 非常直观的语义维度.
- `color` 不依赖最终体素物理模板定稿.
- `color` 既影响视觉反馈, 也能帮助模型保留一部分意象信息.

建议:

- `color_head(z_slot[subject]) -> float[B, N_color_palette]`
- 第一阶段使用固定颜色库分类, 不直接回归 RGB 或 HEX

#### 物质原型 head

第一阶段不训练 `material_archetype_head`.

原因:

- 最终 `material_archetype` 依赖体素物理与模板设计定稿.
- 当前阶段更值得先验证 `text -> stable slots/backfire` 是否成立.
- 因此第一阶段把 `material_archetype` 视为后绑定字段.

第二阶段如需补训, 再加入:

- `material_archetype_head(z_slot[subject]) -> float[B, N_archetype + 1]`
- `+1` 表示 `none`

#### 核心 categorical 槽位 heads

每个 head 输出一个 logits 向量:

- `color_head -> float[B, N_color_palette]`
- `state_head -> float[B, 3]`
- `reaction_kind_head -> float[B, N_reaction]`
- `reaction_direction_head -> float[B, N_reaction_dir]`
- `release_profile_head -> float[B, N_release]`
- `motion_template_head -> float[B, N_motion]`
- `motion_direction_head -> float[B, N_motion_dir]`
- `origin_head -> float[B, N_origin]`
- `target_mode_head -> float[B, N_target_mode]`
- `direction_mode_head -> float[B, N_direction_mode]`

#### multi-label 槽位 head

- `reaction_mask_head -> float[B, N_mask]`
- 训练时用 `BCEWithLogitsLoss`.

#### 连续物理量 heads

为了提高训练稳定性, 不直接回归实数, 而是先做 ordinal bins.

建议把以下字段量化成 7 档:

- `density`
- `temperature`
- `amount`
- `reaction_rate`
- `hardness`
- `friction`
- `viscosity`
- `release_speed`
- `release_spread`
- `release_duration`
- `force_strength`
- `carrier_velocity`

因此每个连续字段对应:

- `slot_head -> float[B, 7]`

运行时再把 7 档映射回 `[0, 1]` 或引擎实际数值范围.

#### 风格 heads

风格字段也建议先做 ordinal 分类, 再映射成 `[0, 1]`.

- `curvature_head -> float[B, 5]`
- `politeness_head -> float[B, 5]`
- `elegance_head -> float[B, 5]`

其中 5 档分别表示:

- very low
- low
- mid
- high
- very high

#### 置信度

第一阶段不建议单独训练 `confidence_head`.

原因:

- 如果没有独立可靠的置信度标签, 单靠正样本和普通状态标签训练出来的往往只是伪置信度.
- 第一阶段更合理的做法是直接从 `status_head` 的 logits 推导:
  - max softmax
  - margin
  - entropy

第二阶段如果补充了:

- hard negatives
- near misses
- prefix `unstable`
- 独立 audit 结果

再考虑加入:

- `confidence_head(z_global) -> float[B, 1]`

### 4.6 最终提交结构

如果 `status = success`, 则模型输出会被整理为:

```json
{
  "status": "success",
  "subject_kind": "summon_material",
  "subject": {},
  "release": {},
  "motion": {},
  "targeting": {},
  "expression": {},
  "confidence": 0.93
}
```

如果 `status = unstable`, 只更新预览, 不真正提交施法.  
如果 `status = backfire`, 直接进入误施法逻辑.

## 5. 推理流程

### 5.1 实时阶段

1. `VAD` 判断玩家是否仍在说话.
2. `ASR` 按 chunk 输出 `stable_prefix_text`.
3. 每次 stable prefix 增长时, parser 重新前向一次.
4. 如果 `status = unstable`, 只更新 UI 预览.
5. 句尾再做一次最终判定.
6. 若 `status = success`, 输出 `B`.
7. 若 `status = backfire`, 触发反噬.

### 5.2 decoder backbone 的增量方式

如果使用 `Qwen3-0.6B-Base` 或 `Qwen3.5-0.8B-Base`, 推荐做法是:

- 对 `ASR` 的稳定前缀使用 `KV cache`.
- 只追加新确认的 token.
- 如果 `ASR` 回改了尾部, 则回滚到最近稳定边界重新算.

这比每次整句从头生成更适合实时链路.

## 6. 训练数据格式

建议统一使用 `jsonl`.

### 6.1 正样本格式

```json
{
  "id": "spell_000001",
  "text": "圣火昭昭, 涤荡前路",
  "status": "success",
  "runtime_b": {
    "subject_kind": "summon_material",
    "subject": {
      "material_archetype": "holy_fire",
      "state": "gas",
      "temperature": "high",
      "density": "low",
      "amount": "mid",
      "reaction_kind": "burn",
      "reaction_rate": "mid",
      "reaction_mask": ["living", "terrain"]
    },
    "release": {
      "release_profile": "stream",
      "release_speed": "mid_high"
    },
    "motion": {
      "motion_template": "flow",
      "force_strength": "mid",
      "carrier_velocity": "mid_high",
      "motion_direction": "forward"
    },
    "targeting": {
      "origin": "self",
      "target_mode": "none",
      "direction_mode": "forward"
    },
    "expression": {
      "curvature": "high",
      "politeness": "mid",
      "elegance": "high"
    }
  },
  "prefix_labels": [
    {
      "text": "圣火昭昭",
      "status": "unstable"
    },
    {
      "text": "圣火昭昭, 涤荡前路",
      "status": "success"
    }
  ],
  "meta": {
    "source": "teacher_llm",
    "speaker": null
  }
}
```

### 6.2 负样本格式

```json
{
  "id": "spell_neg_000001",
  "text": "太阳第四课行星对应之秘, 在我命盘中点燃",
  "status": "backfire",
  "runtime_b": null,
  "prefix_labels": [
    {
      "text": "太阳第四课",
      "status": "unstable"
    },
    {
      "text": "太阳第四课行星对应之秘, 在我命盘中点燃",
      "status": "backfire"
    }
  ],
  "meta": {
    "source": "teacher_llm_hard_negative"
  }
}
```

## 7. 大模型语料生成方案

### 7.1 第一步. 先采样 canonical seeds

不要直接对 `B` 的全笛卡尔积做全量展开.  
应该先定义:

- `30 - 50` 个 `material_archetype`
- `8 - 12` 个 `release_profile`
- `6 - 8` 个 `motion_template`
- `5 - 8` 个 `origin`
- `3 - 5` 个 `direction_mode`
- `20 - 40` 个常见 modifier 组合

然后从中采样 `30k - 60k` 条 canonical seed.

每条 seed 是一个完整但简洁的 `B`.

### 7.2 第二步. 用 Teacher LLM 生成正样本

对每条 seed, 让 teacher 生成:

- `6 - 12` 条直白但不程序化的表达.
- `6 - 12` 条诗性或中二表达.
- `2 - 4` 条高雅/仪式化表达.

每条样本都带风格标签:

- `curvature`
- `politeness`
- `elegance`

目标是让 teacher 输出:

- 表达文本
- 风格标签
- 该文本对应的 `B`
- 关键语义锚点说明

### 7.3 第三步. 生成负样本

负样本至少分 4 类:

1. `near_miss`
   - 看起来接近正确咒语, 但关键槽位缺失或错位.
2. `arcane_unknown`
   - 需要太强世界知识才能解释的表达.
3. `contradictory`
   - 句内语义互相打架.
4. `ornate_but_unmappable`
   - 很华丽, 但没有稳定落点.

对每个正样本 seed, 建议至少再配:

- `3 - 6` 条负样本.

### 7.4 第四步. 用 judge 过滤

Teacher 生成的文本不能直接用.  
需要再过一轮 judge:

1. 让 judge 反向把文本解析回 `B`.
2. 对比 mandatory slots 是否一致.
3. 风格标签是否落在允许误差内.
4. 不一致的样本直接丢弃或降权.

### 7.5 第五步. 构造 ASR 噪声

在文本层面额外生成:

- 同音字替换
- 漏字
- 口癖
- 停顿
- 标点去除
- 局部断句错误

同时对每条正样本抽取 `2 - 4` 个前缀切片, 标成 `unstable` 或 `success`.

### 7.6 第六步. TTS 和 ASR 回环

从高质量文本样本中抽一部分做音频 round-trip:

1. 用 `CosyVoice` 生成多说话人音频.
2. 混入少量环境噪声, 混响, 语速变化.
3. 跑过真实 `ASR`.
4. 把得到的 transcript 和 partial prefix 再喂给 parser 训练.

这样 parser 学到的是:

- teacher 文本
- noisy text
- real ASR text

而不是只会吃干净文本.

## 8. 训练流程

### 8.1 Stage 0. 头部预热

先冻结 backbone, 只训练 heads:

- `status_head`
- 全部 slot heads
- style heads

目的:

- 先验证标签体系是否自洽.
- 先看哪些槽位最难学.

### 8.2 Stage 1. Text-only LoRA

对 decoder backbone 做 `LoRA` 或 `QLoRA`:

- 输入是 teacher 生成的 clean text.
- 任务是学 `success` 样本和 `backfire` 样本.

这一阶段先不加真实 ASR 噪声, 先把语义学稳.

### 8.3 Stage 2. Prefix 和 noisy text

加入:

- partial prefixes
- homophone noise
- deletion noise
- punctuation noise

这一阶段的关键目标是:

- 让模型学会在前缀阶段输出 `unstable`.
- 让模型学会不要过早 commit.

### 8.4 Stage 3. ASR transcript fine-tune

把 TTS -> ASR round-trip 数据加入训练.

这一步的目标是:

- 把 parser 从 clean text 适配到真实 ASR 输出.
- 尤其修正中文同音字和断句问题.

### 8.5 Stage 4. 校准

最后单独做:

- `confidence` 校准
- `success / unstable / backfire` 阈值校准
- 不同说话人和麦克风条件下的阈值评估

## 9. Loss 设计

总 loss 可写为:

```text
L = λ_status * L_status
  + λ_cat * Σ L_categorical_slots
  + λ_bin * Σ L_binned_slots
  + λ_mask * L_reaction_mask
  + λ_style * Σ L_style
```

推荐:

- `status_head`: cross entropy
- categorical slots: cross entropy
- `reaction_mask`: BCE with logits
- binned continuous slots: ordinal cross entropy
- style heads: ordinal cross entropy

注意:

- 当 `status = backfire` 时, 多数槽位 loss 应 mask 掉.
- 当样本是 prefix 且标签为 `unstable` 时, 已知槽位可以保留监督, 未知槽位应 mask 掉.

## 10. 训练成本预估

以下预估以第一阶段推荐规模为例:

- canonical seeds: `40k`
- 正样本: 平均每 seed `12` 条, 共 `480k`
- 负样本: 平均每 seed `6` 条, 共 `240k`
- prefix/noisy 扩增后, 总训练实例约 `1.5M - 2.0M`
- 平均长度按 `64` tokens 估算

则每个 epoch 的 token 量级约为:

- `96M - 128M` tokens

### 10.1 Teacher 语料生成成本

如果用大模型 API 生成文本, 更适合按 token 预算估算:

- 总 prompt + completion token 量级约 `120M - 250M`

实际金钱成本取决于具体 provider, 这里不写死.

### 10.2 TTS 成本

如果抽 `100k - 200k` 条文本做音频合成, 每条做 `4` 个说话人版本, 平均 `3s`:

- 总音频时长约 `333h - 666h`

如果 TTS 实际合成速度在 `0.1x - 0.3x` realtime 左右, 则:

- 单卡合成 wall-clock 大约 `33h - 200h`

这部分通常不是最贵的算力, 但会占明显的离线时间.

### 10.3 Parser 训练成本

#### 只训 heads

- 设备: `1 x 16GB` 或 `1 x 24GB` GPU
- 时间: `1h - 3h`

#### `Qwen3-0.6B-Base` + LoRA / QLoRA

- 设备: `1 x 24GB` GPU 最舒服, `1 x 16GB` 也可做但 batch 更小
- 时间: `8h - 20h`, 取决于 batch size 和 epoch 数

#### `Qwen3.5-0.8B-Base` + LoRA / QLoRA

- 设备: `1 x 24GB` GPU 可做, 但实现更复杂
- 时间: `10h - 28h`

#### full fine-tune

- 不建议第一阶段做
- 需要更高显存, 更长时间, 收益未必比 LoRA 明显

### 10.4 CPU 推理预估

训练不建议走 CPU.  
推理可以走 CPU, 但必须做真实 benchmark.  
对于量化后的 `0.6B - 0.8B` decoder backbone, 短句前缀分类在桌面 CPU 上是有希望的, 但是否稳定落在 `0.5s` 内, 必须按目标机器实测.

## 11. 第一阶段实验顺序

推荐的实验顺序:

1. `Qwen3-0.6B-Base + heads`, 做文本监督 baseline.
2. 加 `prefix` 和 `backfire` 标签.
3. 加 noisy text.
4. 加 TTS -> ASR round-trip.
5. 再跑 `Qwen3.5-0.8B-Base` 对照.
6. 最后再拿 `Qwen3-Embedding-0.6B` 和 `BGE-M3` 做基线对比.

这样可以最快知道:

- decoder 路线是否真的比 embedding 路线更适合这个项目.
- 风格字段是否真的可学.
- `backfire` 阈值是否足够稳定.

## 12. 参考资料

- Qwen3.5-0.8B-Base: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
- Qwen3-0.6B-Base: https://huggingface.co/Qwen/Qwen3-0.6B-Base
- Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding
- FunASR: https://github.com/modelscope/FunASR
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- CosyVoice: https://github.com/FunAudioLLM/CosyVoice
- PEFT LoRA docs: https://huggingface.co/docs/peft/developer_guides/lora
- Transformers quantization docs: https://huggingface.co/docs/transformers/quantization/bitsandbytes
- TRL SFTTrainer docs: https://huggingface.co/docs/trl/sft_trainer
