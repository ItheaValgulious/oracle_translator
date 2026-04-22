# io_utils.py

这个文件提供仓库里通用的 JSON 和 JSONL 读写工具.

## 对外接口

- `ensure_parent(path)`
  - 确保目标路径的父目录存在.

- `write_json(path, payload)`
  - 把对象写成格式化 JSON.

- `read_json(path)`
  - 读取 JSON 文件.

- `write_jsonl(path, rows)`
  - 覆盖写出一组 JSONL 记录.

- `append_jsonl(path, row)`
  - 以追加方式写入一条 JSONL 记录.
  - 适合长时间运行的数据生成任务做断点续跑日志.

- `read_jsonl(path)`
  - 读取 JSONL 文件为列表.

## 依赖的对外接口

- Python 标准库 `json`
- Python 标准库 `pathlib`

## 主要功能

- 统一仓库里的 UTF-8 JSON 文件读写方式.
- 给数据生成和日志落盘提供最基础的 I/O 能力.
