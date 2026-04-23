# json_to_spell.py

这个脚本把输入 JSONL 里的 `model_socket` 反推成咒语.

## 如何使用

```powershell
.\.env\Scripts\python scripts\json_to_spell.py --input data\source\model_socket_seed.jsonl --output data\source\model_socket_to_spell.jsonl --log data\logs\model_socket_to_spell_log.jsonl
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `slm.data_generation.json_rows_to_spells`

## 主要功能

- 调用 `src/prompts/json_to_spell_prompt.md`
- 支持重试和断点续跑
