# prepare_dataset.py

这个文件是 raw 数据合并与切分入口脚本.

## 如何使用

```powershell
.\.env\Scripts\python scripts\prepare_dataset.py --curated data\raw\curated_v1.jsonl --api data\raw\api_v1.jsonl --train data\processed\train_v1.jsonl --val data\processed\val_v1.jsonl
```

如果 `api_v1.jsonl` 不存在, 当前实现会按空数据处理.

## 对外接口

- 无库接口.
- 这是命令行入口程序.

## 依赖的对外接口

- `oracle_translator.data_generation.merge_and_split_datasets`

## 主要功能

- 把 raw 数据合并为训练集和验证集.
- 写出 manifest.

