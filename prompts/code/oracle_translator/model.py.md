# model.py

这个文件提供文本解析模型 `SpellParserModel`.

## 对外接口

- `ParserModelOutput`
  - 模型前向的输出结构.
  - 包含:
    - `status_logits`
    - `status_confidence`
    - `categorical_logits`
    - `binned_logits`
    - `style_logits`
    - `reaction_mask_logits`

- `SpellParserModel`
  - 主模型类.
  - 使用 `Qwen/Qwen3-0.6B-Base` 的 hidden states 作为 backbone 表征.
  - 在 backbone 后接:
    - 全局池化
    - slot attention 池化
    - 多个 MLP heads
  - 支持:
    - `big_head`
    - `LoRA`
    - 部分解冻或全量解冻 backbone
  - 当前仍是旧版风格头实现, 只输出:
    - `expression.curvature`
    - `expression.politeness`
    - `expression.elegance`
  - 尚未实现最新目标中的:
    - `material_template`
    - `reaction_template`
    - `release_template`
    - `motion_template`
    - `motion_direction`
    - `origin`
    - `target`
    - `powerness`

- `build_model(**kwargs)`
  - 构造模型的便捷入口.

## 依赖的对外接口

- `transformers.AutoModel`
- `transformers.AutoConfig`
- `oracle_translator.ontology.CATEGORICAL_SPECS`
- `oracle_translator.ontology.BINNED_SPECS`
- `oracle_translator.ontology.STYLE_SPECS`
- `oracle_translator.ontology.REACTION_MASK_LABELS`

## 主要功能

- 从 tokenizer 输出的 `input_ids` 和 `attention_mask` 中提取 token hidden states.
- 用最后 4 层 hidden states 做 layer mix.
- 构造全局表示和槽位特定表示.
- 输出状态, 槽位, 风格和 multi-label 反应 mask.
- 通过 `status_logits` 推导 `status_confidence`, 不单独训练置信度头.
- 支持冻结 backbone, 或只解冻 backbone 的最后若干层.
- 支持在 backbone 内部目标线性层上挂接 LoRA 适配器.
- 支持把 heads 切换为更大的 MLP 结构.
