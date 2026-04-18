# Model Data Generation Plan

本文定义第一阶段 `text -> stable slots/backfire` 训练所使用的数据生成方案.  
这份文档只讨论数据生成, 翻译, 审阅, 审核, 并发和落盘结构.  
不讨论体素物理实现细节, 也不要求第一阶段训练 `material_archetype_head`.

## 1. 目标

当前数据生成要解决的核心问题不是数量, 而是分布.

现阶段训练数据必须同时覆盖:

1. 诗意化. 这是最终魔法效果最强, 最符合世界观的表达.
2. 生活化. 这是玩家临场最可能说出来的话, 通常不那么华丽, 但更真实.
3. 整活化. 玩家一定会故意用神秘语句, 乱码, 歌词, 笑话, 脏话, 爆论, 或者故意试探系统边界.
4. 多语言化. 第一阶段至少同时覆盖中文和英文.

第一阶段的目标不是直接得到完美大数据集, 而是得到一套可持续扩展, 可审阅, 可回溯, 可重跑的数据生产流水线.

## 2. 第一原则

第一阶段不训练 `material_archetype_head`.

原因:

- `material_archetype` 依赖最终体素物理模板定稿.
- 当前真正需要先验证的是 `text -> stable slots/backfire`.
- 因此第一阶段的数据标注重点是:
  - `status`
  - stable slots
  - style slots
  - 可选的中间 `proto_material_id`, 但不要求最终模板定稿

数据生成系统必须优先产出这些标签, 而不是被最终模板设计卡住.

## 3. 目录约束

从现在开始, 数据生成, 数据清洗, 审核, 训练, 评测相关代码应放在:

- `src/slm`

建议目录结构如下:

```text
src/slm/
  __init__.py
  prompts/
  schemas/
  generation/
  translation/
  review/
  audit/
  dataset/
  training/
  evaluation/
  utils/
```

其中:

- `generation/` 负责生成 seed spell 或 seed json.
- `translation/` 负责 spell -> json, json -> spell.
- `review/` 负责生成供人工审阅的汇总文件.
- `audit/` 负责格式审核, schema 检查, 重试和过滤.
- `dataset/` 负责合并和切分 train/val/test.
- `training/` 负责 PyTorch 训练脚本.
- `evaluation/` 负责评测脚本和报告.

当前仓库里已有的 `src/oracle_translator` 属于过渡实现.  
后续如果继续开发, 应整体迁移到 `src/slm`.

## 4. 数据组成

第一阶段数据由 4 部分混合构成.

### 4.1 Part A. 用户手写 30 条玩家咒语

流程:

1. 用户手写 `30` 条原始咒语.
2. 系统不直接把它们放进训练集.
3. 先调用翻译模型, 把每条咒语翻译成目标 `json`.
4. 再生成一份汇总审阅文件, 供人工检查.

这部分的核心产物不是训练样本本身, 而是:

- 玩家原始表达样本
- 对应翻译结果
- 可审阅的 `咒语 -> json` 对照表

建议文件:

```text
data/source/manual_player_spells_v1.jsonl
data/review/manual_player_spells_v1_review.md
data/review/manual_player_spells_v1_review.jsonl
```

每条记录建议字段:

```json
{
  "id": "manual_0001",
  "language": "zh",
  "text": "圣火昭昭, 涤荡前方的路",
  "status": "pending_review",
  "translated_json": {},
  "translator_meta": {},
  "review_meta": {}
}
```

审阅文件必须清楚展示:

- 原始咒语
- 翻译后的 `json`
- 必填槽位摘要
- 风格摘要
- 人工批注位

### 4.2 Part B. 基于用户 30 条样本, 调用 GPT-5.4 生成 300 条

这部分不是无条件自由生成.  
每次调用的上下文必须包含:

1. 当前 generation prompt
2. 用户手写的 `30` 条样本
3. 项目的 [magic_socket.md](C:/Projects/oracle_translator/prompts/magic_socket.md)

流程:

1. 读取用户 `30` 条手写样本.
2. 把这 `30` 条样本和 `magic_socket.md` 一起喂给 `gpt-5.4`.
3. 让它扩写出 `300` 条咒语.
4. 扩写出的 `300` 条咒语, 不能直接作为训练数据.
5. 再使用独立上下文和另一套 translation prompt, 把这 `300` 条咒语翻译成 `json`.
6. 最后生成审阅文件.

这里必须强调 "独立上下文":

- 生成咒语的模型上下文, 不能和翻译 `json` 的模型上下文混在一起.
- 否则模型会偷看目标结构, 导致数据过于理想化.

建议文件:

```text
data/source/gpt54_spell_gen_v1.jsonl
data/source/gpt54_spell_gen_v1_translated.jsonl
data/review/gpt54_spell_gen_v1_review.md
```

### 4.3 Part C. 先随机出一个 json, 再让 GPT-5.4 反推一个咒语

流程:

1. 先随机采样一个合法 `json`.
2. 输入这个 `json`.
3. 让 `gpt-5.4` 思考一个自然语言咒语, 使其能够表达这个 `json`.

这一部分的意义是:

- 从结构空间反推表达空间.
- 避免数据完全被现有手写咒语风格牵引.
- 更容易覆盖低频槽位组合.

这类数据尤其适合补足:

- 少见的 `release_profile`
- 少见的 `motion_template`
- 少见的 `reaction_mask`
- 不常见的 style 组合

建议文件:

```text
data/source/random_json_to_spell_v1.jsonl
data/review/random_json_to_spell_v1_review.md
```

### 4.4 Part D. 先随机一个咒语, 再独立翻译成 json

流程:

1. 先随机生成一个咒语文本.
2. 生成这条咒语时, 不同步生成 `json`.
3. 再单独开一次独立上下文调用, 让模型翻译成 `json`.
4. 这一步使用的 prompt, 不是直接引用项目文件, 而是使用我们自己写的一份非常详细的 translation 文档.

这一部分的意义是:

- 模拟更真实的开放输入.
- 使用廉价模型
- 避免生成模型和翻译模型共享同一个隐含答案.
- 检查 translation prompt 本身是否足够稳.

建议文件:

```text
data/source/random_spell_to_json_v1.jsonl
data/review/random_spell_to_json_v1_review.md
```

## 5. 四部分混合的目的

4 部分不是重复建设, 而是分别解决不同偏差.

Part A 解决:

- 紧贴真实玩家表达
- 人工可控
- 可快速审阅

Part B 解决:

- 在用户风格附近扩张
- 放大用户定义的世界观
- 更容易得到成规模样本

Part C 解决:

- 从结构空间覆盖长尾组合
- 减少只围绕少数热门 spell 的偏置

Part D 解决:

- 模拟开放输入
- 检查翻译 prompt 的稳健性
- 让 generation 和 translation 真正解耦

## 6. 语体配额

每一部分的数据都必须尽量兼顾 4 种大分布:

1. 诗意化
2. 生活化
3. 整活化
4. 多语言化

建议在数据 manifest 中显式记录:

```json
{
  "language": "zh",
  "tone_bucket": "poetic",
  "style_bucket": "ritual",
  "chaos_bucket": "serious",
  "source_part": "B"
}
```

推荐的标签维度:

- `language`: `zh`, `en`
- `tone_bucket`: `poetic`, `everyday`, `meme`, `mixed`
- `register_bucket`: `battle`, `casual`, `ritual`, `classical`, `abusive`
- `chaos_bucket`: `serious`, `playful`, `chaotic`, `provocative`

注意:

- `整活化` 不是简单加脏话.
- 它应该覆盖:
  - 乱码试探
  - 歌词试探
  - 笑话试探
  - 脏话试探
  - 爆论试探
  - 故意玄学化试探

这类样本很多都应该落到:

- `unstable`
- `backfire`

而不是强行映射到成功施法.

## 7. Prompt 体系

整个数据生产不能只用一个 prompt.

至少需要 5 类 prompt:

1. `spell_generation_prompt`
   - 从样本或上下文中生成咒语文本
2. `spell_translation_prompt`
   - 从咒语翻译到 `json`
3. `json_to_spell_prompt`
   - 从 `json` 反推咒语
4. `audit_prompt`
   - 检查 `咒语 <-> json` 是否一致
5. `review_render_prompt`
   - 生成供人工阅读的摘要或审阅材料

其中:

- generation prompt 要强调语体跨度和多样化.
- translation prompt 要强调稳定结构化和拒识.
- audit prompt 要强调一致性检查.

## 8. 上下文要求

### 8.1 生成咒语时的上下文

Part B 中, 每次调用必须包含:

1. generation prompt
2. 用户手写的 `30` 条样本
3. [magic_socket.md](C:/Projects/oracle_translator/prompts/magic_socket.md)

### 8.2 翻译 json 时的上下文

translation 任务必须独立开上下文.

并且分两种:

1. 标准 translation 上下文
   - 用于 Part A 和 Part B
2. 超详细 translation 上下文
   - 用于 Part D
   - 使用我们单独写的详细翻译文档

### 8.3 审核时的上下文

审核模型建议再单独独立上下文, 避免 generation prompt 污染 audit 判断.

## 9. 格式审核与重试

调用 API 时必须默认启用:

1. JSON schema 检查
2. 必填字段检查
3. label 空间检查
4. 逻辑一致性检查
5. 失败重试

审核顺序建议如下:

1. 原始响应是否可解析
2. JSON 顶层结构是否正确
3. 各字段类型是否正确
4. 枚举值是否在 label space 中
5. 条件字段是否满足依赖约束
6. `status` 与 `json` 是否自洽
7. `咒语` 和 `json` 是否明显冲突

重试建议:

- 格式错误时, 优先原 prompt 重试 `2-3` 次
- 连续失败后, 切到 fallback prompt
- fallback 仍失败时, 记录为 bad case, 不自动吞掉

每次失败都必须保留:

- request id
- prompt hash
- source part
- raw response
- parse error
- retry count

## 10. 并发策略

API 调用必须支持可配置并发数.

建议配置:

```yaml
api:
  max_concurrency: 8
  max_retries: 3
  request_timeout_sec: 300
  backoff_base_sec: 2
  jitter: true
```

并发原则:

- generation 可并发
- translation 可并发
- audit 可并发
- 同一个样本链条内部尽量串行
  - 先生成
  - 再翻译
  - 再审核

原因:

- 保持样本生命周期清晰
- 方便定位哪一层坏掉

## 11. 审阅产物

每一部分数据都必须生成人工审阅文件.

审阅文件建议同时提供:

1. `jsonl`
2. `md`

Markdown 审阅文件建议每条都包含:

- 样本 id
- source part
- language
- tone bucket
- 原始咒语
- 翻译 `json`
- 风格摘要
- 审核结果
- 人工批注位

这样你可以快速扫出:

- 哪些太单调
- 哪些太像模板改写
- 哪些生活化不足
- 哪些整活化过头

## 12. 数据比例建议

第一阶段不需要一开始就特别大, 但要控制来源比例.

建议第一版:

- Part A: `30`
- Part B: `300`
- Part C: `300`
- Part D: `300`

后续再滚动扩张.

语言比例建议:

- 中文 `70%`
- 英文 `30%`

风格比例建议:

- 诗意化 `30%`
- 生活化 `25%`
- 整活化 `25%`
- 混合型 `20%`

这里的比例不是死规定, 但必须写进 manifest, 不能最后完全凭感觉.

## 13. 训练前过滤

不是所有生成出来的样本都直接进训练集.

建议至少过滤掉:

1. 纯同义词替换, 几乎没信息增量的样本
2. 明显编程式表达
3. 乱码过高, 不具备可读性的样本
4. 审核失败的样本
5. `咒语` 与 `json` 强冲突的样本
6. 语言标签错误的样本

## 14. 需要额外写的配套文档

除了本文件, 后续还应补:

1. `model_translation_prompt.md`
   - spell -> json 的详细翻译规范
2. `model_audit_prompt.md`
   - 一致性审核规范
3. `model_review_schema.md`
   - 人工审阅文件格式
4. `model_sampling_schema.md`
   - random json 和 random spell 的采样规则

其中 Part D 明确要求使用的 "十分详细的文档", 就应落在:

- `prompts/model_translation_prompt.md`

## 15. 当前阶段的实施顺序

建议按这个顺序推进:

1. 先落 Part A
   - 用户写 30 条
   - 系统翻译
   - 生成汇总审阅文件
2. 再落 Part B
   - 带上 30 条样本和 `magic_socket.md`
   - 生成 300 条
   - 独立上下文翻译
   - 生成审阅文件
3. 再做 Part C
   - random json -> spell
4. 最后做 Part D
   - random spell -> independent translation

原因:

- Part A 和 Part B 更贴近真实玩家分布
- Part C 和 Part D 更像扩展覆盖和稳健性补强

## 16. 核心结论

这套系统的重点不是 "多调几次 API", 而是:

- 多来源混合
- generation 和 translation 解耦
- 审核独立
- 并发可控
- 格式审查严格
- 审阅产物完整
- 目录结构清晰

如果这几件事不同时成立, 数据会很快再次滑回:

- 同义改写过多
- 语体太窄
- 生活化不足
- 审核不可追踪
- 后续训练问题难以定位
