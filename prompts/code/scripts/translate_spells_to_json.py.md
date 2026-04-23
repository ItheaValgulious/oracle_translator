# translate_spells_to_json.py

这个脚本把一组咒语翻译成 `model_socket`.

## 如何使用

```powershell
.\.env\Scripts\python scripts\translate_spells_to_json.py --input data\source\manual_spell_seeds.jsonl --output data\source\spell_to_model_socket.jsonl --log data\logs\spell_to_model_socket_log.jsonl
```

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `slm.data_generation.translate_spells_to_json`

## 主要功能

- 调用 `src/prompts/spell_to_json_prompt.md`
- 对输出做 schema 规范化与校验
- 支持重试和断点续跑
