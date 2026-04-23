# run_random_json_to_spell.py

这个脚本串起完整的 `random model socket -> spell` 流程.

## 如何使用

```powershell
.\.env\Scripts\python scripts\run_random_json_to_spell.py --output data\source\random_model_socket_to_spell.jsonl --log data\logs\random_model_socket_to_spell_log.jsonl --target-count 100
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `slm.data_generation.generate_json_to_spell_dataset`

## 主要功能

- 先采样随机 `model_socket`
- 再反推成咒语
- 整体支持重试和断点续跑
