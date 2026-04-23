# run_random_spell_to_json.py

这个脚本串起完整的 `random spell -> model socket` 流程.

## 如何使用

```powershell
.\.env\Scripts\python scripts\run_random_spell_to_json.py --seed-examples data\source\manual_spell_seeds.jsonl --generated-spells data\source\random_spells.jsonl --output data\source\random_spell_to_model_socket.jsonl --target-count 100
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `slm.data_generation.build_random_spell_to_json_dataset`

## 主要功能

- 先生成随机咒语
- 再把这些咒语翻译成 `model_socket`
- 整体支持重试和断点续跑
