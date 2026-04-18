# dataset.py

这个文件负责把 `jsonl` 样本转换成训练 batch.

## 对外接口

- `EncodedSample`
  - 单条样本的已编码表示.

- `encode_row(row)`
  - 把一条原始样本编码为:
    - `status_id`
    - categorical labels
    - binned labels
    - style labels
    - reaction mask

- `SpellDataset`
  - 训练集/验证集的 `torch.utils.data.Dataset`.

- `SpellCollator`
  - 对一批 `EncodedSample` 做 tokenizer 和张量拼装.

## 依赖的对外接口

- `oracle_translator.io_utils.read_jsonl`
- `oracle_translator.ontology.CATEGORICAL_SPECS`
- `oracle_translator.ontology.BINNED_SPECS`
- `oracle_translator.ontology.STYLE_SPECS`
- `oracle_translator.ontology.REACTION_MASK_LABELS`

## 主要功能

- 对 success 样本保留槽位监督.
- 对 unstable/backfire 样本只保留适合的监督.
- 为训练代码提供统一的 flat tensor dictionary.

