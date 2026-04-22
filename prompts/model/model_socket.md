# Model Socket

本文定义第一版 parser 直接输出的模型侧结构.
它不是完整 runtime 执行结构, 而是一个更收缩, 更容易学的中间层.

当前第一版模型只输出:

```json
{
  "subject_kind": "summon_material",
  "subject": {
    "material_template": "..."
  },
  "reaction": {
    "reaction_template": "..."
  },
  "release": {
    "release_template": "..."
  },
  "motion": {
    "motion_template": "...",
    "motion_direction": "...",
    "origin": "...",
    "target": "..."
  },
  "expression": {
    "powerness": 0.0
  }
}
```

## 1. 顶层原则

- 第一版不直接预测细粒度物理属性头.
- 第一版不直接预测完整 reaction 属性头.
- 第一版不直接预测完整 release 数值细节.
- 第一版的目标是:
  - 先把文本稳定映射到模板选择
  - 再由模板扩展成完整 runtime 字段

## 2. `subject.material_template`

表示“召出来的是什么物质模板”.

第一版建议先收缩到约 20 类:

1. `granite`
2. `earth`
3. `obsidian`
4. `sand`
5. `water`
6. `ice`
7. `steam`
8. `fire`
9. `light`
10. `wind`
11. `lightning`
12. `acid`
13. `poison_slurry`
14. `tar`
15. `explosive_slurry`
16. `grass`
17. `wood`
18. `glass`
19. `iron`
20. `quicksilver`

这些模板各自带默认的:

- `color`
- `state`
- `density`
- `temperature`
- `amount`
- 可选的 `hardness/friction/viscosity`
- 可选的默认 `release` 参数

## 3. `reaction.reaction_template`

表示“接触到别的对象以后, 按什么反应模板扩展”.

第一版先保持直观模板名:

- `none`
- `burn`
- `corrode`
- `freeze`
- `poison`
- `grow`

但它们在完整 runtime 里应展开成更底层的机制.

### 3.1 底层含义

reaction 的本质不是文学标签, 而是:

- 接触后把对象转化成什么
- 转化速度多快
- 向哪扩散
- 能感染几代

因此完整 runtime 里应至少展开成:

- `convert_mode`
  - `self`
  - `empty`
- `reaction_speed`
- `reaction_mask`
- `reaction_direction`
- `generation_limit`

推荐映射:

- `burn -> self`
- `freeze -> self`
- `grow -> self`
- `poison -> self`
- `corrode -> empty`
- `none -> none`

## 4. `release.release_template`

第一版模型还应直接输出:

- `release_template`

允许值:

- `spray`
- `appear`

含义:

- `spray`
  - 快速逐渐释放
  - 展开后主要补成:
    - `release_speed`
    - `release_spread`
- `appear`
  - 瞬间全部出现
  - 展开后也补成:
    - `release_speed`
    - `release_spread`

注意:

- 第一版不单独预测 `release_duration`
- `duration` 不应作为独立字段预测
- 它更像由 `amount / speed` 推导出来的次级量

## 5. `motion`

第一版不再保留独立的 `targeting` 区块.
相关字段全部并进 `motion`.

### 4.1 `motion.motion_template`

允许值:

- `none`
- `fixed`
- `flow`
- `vortex`
- `rotation`
- `vibration`

### 4.2 `motion.motion_direction`

允许值:

- `forward`
- `backward`
- `up`
- `down`
- `target`
- `self`
- `front_up`
- `front_down`

### 4.3 `motion.origin`

允许值:

- `self`
- `back`
- `front_up`
- `front_down`

### 4.4 `motion.target`

允许值:

- `self`
- `enemy`
- `none`

## 6. `expression.powerness`

第一版不再输出多条风格轴.
只保留一个连续单值:

- `powerness`

它表示一句话带来的综合威力增益倾向.

它不再是礼貌或书面程度.
而是更接近:

- 这句说法有多“像有效施法”
- 说法是否更凝练, 更强势, 更有压迫感
- 同义施法下, 这句是否更有威力感

建议范围:

- `0.0 - 1.0`

解释:

- 越接近 `0.0`
  - 越弱
  - 越散
  - 越像随口一说
- 越接近 `1.0`
  - 越像强力施法
  - 越有压迫感
  - 越有“这一句会更强”的感觉

## 7. 为什么第一版要收缩

原因很简单:

- 细粒度连续属性头太多, 第一版难学稳
- reaction 真正的执行含义比旧版标签复杂
- 先选模板, 再展开细节, 更符合第一版工程目标

因此第一版模型的任务是:

- 从语言里选对 `material_template`
- 选对 `reaction_template`
- 选对 `release_template`
- 选对 `motion_template`
- 选对 `motion_direction/origin/target`
- 给一个合理的 `powerness`

剩下的完整字段由模板系统补足.
