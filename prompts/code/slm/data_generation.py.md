# data_generation.py

这个文件负责新的完整句子 `text <-> model socket` 数据生成流水线.

## 对外接口

- 常量:
  - `STYLE_RECIPES`
  - `CONTENT_MOTIFS`
  - `MODEL_SOCKET_BLUEPRINTS`

- `load_api_credentials(path=ROOT / 'api.txt')`
  - 读取 API 地址和 key.

- `load_prompt_template(filename)`
  - 从 `src/prompts/` 目录读取 prompt 模板.

- `generate_spells(...)`
  - 按 style recipe 和 content motif 生成完整咒语.
  - 支持重试, 去重, 断点续跑.

- `translate_spells_to_json(...)`
  - 把咒语翻译成 `model_socket`.
  - 调用后会做 schema 规范化和校验.

- `write_model_socket_seed_samples(...)`
  - 随机采样合法 `model_socket` 种子并落盘.

- `json_rows_to_spells(...)`
  - 把输入 JSONL 里的 `model_socket` 反推成咒语.

- `generate_json_to_spell_dataset(...)`
  - 先采样随机 `model_socket`, 再调用 `json_rows_to_spells`.

- `build_random_spell_to_json_dataset(...)`
  - 先生成咒语, 再翻译成 `model_socket`.

## 依赖的对外接口

- `slm.io_utils.append_jsonl`
- `slm.io_utils.read_jsonl`
- `slm.model_socket_schema.normalize_model_socket`
- `slm.model_socket_schema.validate_model_socket`
- `httpx`

## 主要功能

- 维护 style recipe 和内容物象覆盖策略.
- 当前 recipe 只显式携带:
  - `politeness`
  - 文风引导说明
- 当前 recipe 已显式覆盖:
  - 半文半白
  - 文学描写
  - 生活化
  - 口语化
  - 现实参照
  - 细节型文学描写
  - 人类钻空子/化学式/漏洞试探
- 读取独立 prompt 文件, 而不是把 prompt 写死在 CLI 入口里.
  - 当前代码使用的 prompt 文件位于 `src/prompts/`.
- 为长时间运行任务提供:
  - 每成功一条立刻落盘
  - JSONL 追加日志
  - 重试
  - 断点续跑
- 把当前 `model socket` 流程与旧的 `curated/api augment` 流程隔离开.
