# build_seed_spells.py

这个脚本把我手写的种子咒语落盘成 JSONL.

## 如何使用

```powershell
.\.env\Scripts\python scripts\build_seed_spells.py
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `slm.data_generation.STYLE_RECIPES`
- `slm.io_utils.write_jsonl`

## 主要功能

- 生成 `data/source/manual_spell_seeds.jsonl`
- 为后续 spell generation prompt 提供 example bank
- 把每条手写种子附上:
  - `recipe_id`
  - `motif_id`
  - `politeness_target`
