# json_to_spell_v2.py

这个脚本把输入 JSONL 里的 `model_socket` 反推成咒语.

## 如何使用

```powershell
.\.env\Scripts\python scripts\json_to_spell_v2.py --input data\source\runtime_seed_v2.jsonl --output data\source\json_to_spell_v2.jsonl --log data\logs\json_to_spell_v2_log.jsonl
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.data_generation_v2.json_rows_to_spells`

## 主要功能

- 调用 `src/prompts/json_to_spell_prompt_v2.md`
- 支持重试和断点续跑
