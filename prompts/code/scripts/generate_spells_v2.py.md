# generate_spells_v2.py

这个脚本按 style recipe 和内容物象生成完整咒语.

## 如何使用

```powershell
.\.env\Scripts\python scripts\generate_spells_v2.py --seed-examples data\source\manual_spell_seeds_v2.jsonl --output data\source\generated_spells_v2.jsonl --log data\logs\generated_spells_v2_log.jsonl --target-count 100
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.data_generation_v2.generate_spells`

## 主要功能

- 使用 `src/prompts/spell_generation_prompt_v2.md` 生成完整咒语.
- 支持重试, 去重和断点续跑.
