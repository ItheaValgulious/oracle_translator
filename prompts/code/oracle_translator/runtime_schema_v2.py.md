# runtime_schema_v2.py

这个文件定义旧版 v2 流程里使用的完整 runtime schema.

## 对外接口

- 常量:
  - `SCHEMA_VERSION`
  - `SUBJECT_KINDS`
  - `COLOR_LABELS`
  - `STATE_LABELS`
  - `REACTION_KIND_LABELS`
  - `REACTION_DIRECTION_LABELS`
  - `REACTION_MASK_LABELS`
  - `RELEASE_PROFILE_LABELS`
  - `MOTION_TEMPLATE_LABELS`
  - `MOTION_DIRECTION_LABELS`
  - `ORIGIN_LABELS`
  - `DIRECTION_MODE_LABELS`
  - `VALUE_BINS_7`
  - `STYLE_AXIS_FIELDS`
  - `MANDATORY_RUNTIME_FIELDS`

- `SlotSpec`
  - 描述一个 categorical 或 binned 槽位的标签空间.

- `get_nested(mapping, path)`
- `set_nested(mapping, path, value)`

- `normalize_runtime_b(runtime_b)`
  - 对模型常见的自由枚举写法做规范化.
  - 当前会处理:
    - `medium -> mid`
    - `very_hot -> very_high`
    - `to_target -> to_enemy`
    - `target_mode` 遗留字段去除
    - 若干 `reaction_mask` 同义映射

- `validate_runtime_b(runtime_b, allow_none=False)`
  - 校验一条 `runtime_b` 是否符合 v2 schema.

## 依赖的对外接口

- Python 标准库 `copy`
- Python 标准库 `dataclasses`

## 主要功能

- 定义旧版 v2 数据生成脚本实际使用的 label space.
- 与旧的 `ontology.py` 解耦, 避免打断旧训练代码.
- 当前已经不再是最新目标.
- 最新第一版结构已切到 `model_socket_schema_v2.py`.
