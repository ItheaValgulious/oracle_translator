# materials.py

这个文件定义第一版体素引擎原型使用的材质族表。

## 对外接口

- `build_material_registry()`
  - 返回完整的 `MaterialRegistry`。

## 依赖的对外接口

- `engine.types.MaterialFamily`
- `engine.types.MaterialRegistry`
- `engine.types.PhaseRule`
- `engine.types.ReactionKind`
- `engine.types.SimKind`
- `engine.types.VariantDef`

## 主要功能

- 定义 `stone/sand/glass/iron/water/acid/poison/tar/fire` 等核心家族。
- 定义每个家族的:
  - 默认变体
  - 坍塌目标
  - 变体静态参数
  - 相变闭环
  - 反应与渲染配置
- 让 `engine` 其他模块通过查表获得静态材质参数,而不是把这些参数复制到每个 cell。
