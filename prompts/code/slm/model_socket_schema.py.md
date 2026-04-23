# model_socket_schema.py

这个文件定义新一版第一阶段 `model socket` 的 schema.

## 对外接口

- 常量:
  - `SCHEMA_VERSION`
  - `SUBJECT_KINDS`
  - `MATERIAL_TEMPLATE_LABELS`
  - `REACTION_TEMPLATE_LABELS`
  - `RELEASE_TEMPLATE_LABELS`
  - `MOTION_TEMPLATE_LABELS`
  - `MOTION_DIRECTION_LABELS`
  - `ORIGIN_LABELS`
  - `TARGET_LABELS`

- `SlotSpec`
  - 描述一个模板字段的标签空间.

- `get_nested(mapping, path)`
- `set_nested(mapping, path, value)`

- `normalize_model_socket(model_socket)`
  - 对模型常见自由写法做规范化.

- `validate_model_socket(model_socket, allow_none=False)`
  - 校验一条 `model_socket` 是否符合当前 schema.

## 依赖的对外接口

- Python 标准库 `copy`
- Python 标准库 `dataclasses`

## 主要功能

- 定义当前第一版真正训练的中间结构.
- 与旧的完整 runtime schema 解耦.
- 让 `spell -> model socket` 和 `model socket -> spell` 共用同一套校验器.
