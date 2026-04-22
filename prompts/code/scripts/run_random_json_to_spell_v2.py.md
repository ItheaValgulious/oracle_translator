# run_random_json_to_spell_v2.py

这个脚本串起完整的 `random json -> spell` 流程.

## 如何使用

```powershell
.\.env\Scripts\python scripts\run_random_json_to_spell_v2.py --output data\source\random_json_to_spell_v2.jsonl --log data\logs\random_json_to_spell_v2_log.jsonl --target-count 100
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.data_generation_v2.generate_json_to_spell_dataset`

## 主要功能

- 先采样随机 `model_socket`
- 再反推成咒语
- 整体支持重试和断点续跑
