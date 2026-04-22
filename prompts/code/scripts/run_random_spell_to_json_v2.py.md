# run_random_spell_to_json_v2.py

这个脚本串起完整的 `random spell -> json` 流程.

## 如何使用

```powershell
.\.env\Scripts\python scripts\run_random_spell_to_json_v2.py --seed-examples data\source\manual_spell_seeds_v2.jsonl --generated-spells data\source\random_spells_v2.jsonl --output data\source\random_spell_to_json_v2.jsonl --target-count 100
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.data_generation_v2.build_random_spell_to_json_dataset`

## 主要功能

- 先生成随机咒语
- 再把这些咒语翻译成 `model_socket`
- 整体支持重试和断点续跑
