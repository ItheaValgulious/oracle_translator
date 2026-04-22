# build_seed_spells_v2.py

这个脚本把我手写的种子咒语落盘成 JSONL.

## 如何使用

```powershell
.\.env\Scripts\python scripts\build_seed_spells_v2.py
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.data_generation_v2.STYLE_RECIPES`
- `oracle_translator.io_utils.write_jsonl`

## 主要功能

- 生成 `data/source/manual_spell_seeds_v2.jsonl`
- 为后续 spell generation prompt 提供 example bank
- 把每条手写种子附上:
  - `recipe_id`
  - `motif_id`
  - `powerness_target`
- 当前版本在原有基础上又补了 100 条, 重点拉高:
  - `everyday_clean`
  - `colloquial_with_lift`
  - `detail_literary`
  - `modern_reference_heavy`
  - `hacky_clever`
