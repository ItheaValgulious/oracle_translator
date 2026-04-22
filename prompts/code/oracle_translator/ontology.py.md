# ontology.py

这个文件定义训练和解析时共享的 label space.

## 对外接口

- 常量:
  - `STATUS_LABELS`
  - `COLOR_LABELS`
  - `STATE_LABELS`
  - `REACTION_KIND_LABELS`
  - `REACTION_DIRECTION_LABELS`
  - `REACTION_MASK_LABELS`
  - `RELEASE_PROFILE_LABELS`
  - `MOTION_TEMPLATE_LABELS`
  - `MOTION_DIRECTION_LABELS`
  - `ORIGIN_LABELS`
  - `TARGET_MODE_LABELS`
  - `DIRECTION_MODE_LABELS`
  - `STYLE_BINS`
  - `VALUE_BINS_7`

- `validate_runtime_b(runtime_b, allow_none=False)`
  - 校验一条 runtime json 是否落在当前 ontology 内.

- `get_nested(mapping, path)`
  - 读取二级字段.

- `set_nested(mapping, path, value)`
  - 写入二级字段.

- `SlotSpec`
  - 描述一个训练槽位及其标签空间.

- `CATEGORICAL_SPECS`
- `BINNED_SPECS`
- `STYLE_SPECS`

## 依赖的对外接口

- 无外部业务模块依赖.

## 主要功能

- 定义第一阶段真正训练的字段集合.
- 明确哪些字段是 categorical, 哪些是 7 档数值, 哪些是 5 档风格.
- 统一数据生成, 数据集编码, 模型 head 数量, 指标统计时使用的标签空间.
- 当前文件里的风格空间仍是旧版:
  - `STYLE_BINS`
  - `STYLE_SPECS`
  - 对应 `curvature/politeness/elegance`
- 它尚未表达最新目标中的:
  - `material_template`
  - `reaction_template`
  - `release_template`
  - `motion.origin`
  - `motion.target`
  - `expression.powerness`
