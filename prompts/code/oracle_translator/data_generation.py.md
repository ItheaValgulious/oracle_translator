# data_generation.py

这个文件负责第一阶段文本数据的构造, 扩写, 合并和切分.

## 对外接口

- `build_curated_dataset()`
  - 生成手工模板构造的 curated 样本.

- `write_curated_dataset(output_path)`
  - 把 curated 样本写成 `jsonl`.

- `load_api_credentials(path='api.txt')`
  - 读取 API 地址和 key.

- `generate_api_augmentations(...)`
  - 调用外部模型生成扩写数据.
  - 支持断点续跑和重试.

- `merge_and_split_datasets(...)`
  - 合并 raw 数据并切分为 train/val.
  - 会按配置展开 prefix 样本.

## 依赖的对外接口

- `oracle_translator.io_utils.read_jsonl`
- `oracle_translator.io_utils.write_json`
- `oracle_translator.io_utils.write_jsonl`
- `oracle_translator.ontology.validate_runtime_b`

## 主要功能

- 维护第一阶段咒语模板和 runtime json 的对应关系.
- 负责 API 扩写时的 prompt 构造.
- 负责 raw 数据层面的落盘和 split.

