# Magic Socket

本文定义游戏运行时真正消费的完整魔法结构.

它不是第一版模型直接输出的内容.
第一版模型先输出 [model_socket.md](/mnt/d/project/oracle_translator/prompts/model/model_socket.md), 再由模板展开器补成这里的完整结构.

## 1. 总体结构

```json
{
  "subject_kind": "summon_material",
  "subject": {},
  "reaction": {},
  "release": {},
  "motion": {},
  "expression": {}
}
```

## 2. 第一版与完整结构的关系

第一版模型只直接输出:

- `material_template`
- `reaction_template`
- `release_template`
- `motion_template`
- `motion_direction`
- `origin`
- `target`
- `powerness`

然后模板展开器把它们补成完整运行时字段.

## 3. Subject

`subject` 表示被召出来并进入模拟的主体物质.

### 3.1 模板字段

- `material_template`
  - 例如:
    - `granite`
    - `earth`
    - `obsidian`
    - `sand`
    - `water`
    - `ice`
    - `steam`
    - `fire`
    - `light`
    - `wind`
    - `lightning`
    - `acid`
    - `poison_slurry`
    - `tar`
    - `explosive_slurry`
    - `grass`
    - `wood`
    - `glass`
    - `iron`
    - `quicksilver`

### 3.2 展开后的完整字段

- `color`
- `state`
- `density`
- `temperature`
- `amount`
- `hardness`
- `friction`
- `viscosity`

这些字段不由第一版模型直接预测, 而由 `material_template` 展开得到.

## 4. Reaction

`reaction` 表示主体接触到别的对象后, 如何把它们转化.

### 4.1 模板字段

- `reaction_template`
  - 允许值:
    - `none`
    - `burn`
    - `corrode`
    - `freeze`
    - `poison`
    - `grow`

### 4.2 反应的真实含义

reaction 的本质不是文学标签, 而是:

- 接触后把目标转化成什么
- 转化速度多快
- 向哪个方向扩散
- 感染几代

因此完整字段应包括:

- `convert_mode`
  - `self`
  - `empty`
  - `none`
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

### 4.3 方向示例

- 草向上长:
  - `reaction_template = grow`
  - `reaction_direction = up`
- 生根向下:
  - `reaction_template = grow`
  - `reaction_direction = down`
- 火往前蔓:
  - `reaction_template = burn`
  - `reaction_direction = forward`

## 5. Release

第一版模型直接预测:

- `release_template`

再由模板默认值展开:

完整字段保留:

- `release_template`
- `release_profile`
- `release_speed`
- `release_spread`

### 5.1 模板字段

- `release_template`
  - `spray`
  - `appear`

含义:

- `spray`
  - 快速逐渐释放
- `appear`
  - 瞬间全部出现

### 5.2 关于 `duration`

完整 runtime 不再把 `release_duration` 作为独立字段强调.
至少在第一版设计里:

- `duration` 更像一个由 `amount / speed` 推导出来的次级量
- 不应成为模型单独预测字段

## 6. Motion

第一版把旧 `targeting` 合并进 `motion`.

### 6.1 模板字段

- `motion_template`
  - `none`
  - `fixed`
  - `flow`
  - `vortex`
  - `rotation`
  - `vibration`

### 6.2 方向字段

- `motion_direction`
  - `forward`
  - `backward`
  - `up`
  - `down`
  - `target`
  - `self`
  - `front_up`
  - `front_down`

### 6.3 起点与目标

- `origin`
  - `self`
  - `back`
  - `front_up`
  - `front_down`

- `target`
  - `self`
  - `enemy`
  - `none`

### 6.4 完整运动字段

展开后完整运行时还可包括:

- `force_strength`
- `carrier_velocity`

这些也不要求第一版模型直接输出.

## 7. Expression

第一版只保留一个连续值:

- `powerness`

它是综合威力倾向值, 统一承载:

- 这句说法的威力感
- 说法的压迫感
- 同义施法下的强势程度

建议范围:

- `0.0 - 1.0`

## 8. 第一版展开示意

第一版模型输出:

```json
{
  "subject_kind": "summon_material",
  "subject": {
    "material_template": "fire"
  },
  "reaction": {
    "reaction_template": "burn"
  },
  "release": {
    "release_template": "spray"
  },
  "motion": {
    "motion_template": "flow",
    "motion_direction": "forward",
    "origin": "self",
    "target": "enemy"
  },
  "expression": {
    "powerness": 0.72
  }
}
```

展开后可变成:

```json
{
  "subject_kind": "summon_material",
  "subject": {
    "material_template": "fire",
    "color": "orange",
    "state": "gas",
    "density": "low",
    "temperature": "high",
    "amount": "mid_high"
  },
  "reaction": {
    "reaction_template": "burn",
    "convert_mode": "self",
    "reaction_speed": "high",
    "reaction_mask": ["living", "terrain"],
    "reaction_direction": "forward",
    "generation_limit": 2
  },
  "release": {
    "release_template": "spray",
    "release_profile": "stream",
    "release_speed": "high",
    "release_spread": "mid_low"
  },
  "motion": {
    "motion_template": "flow",
    "motion_direction": "forward",
    "origin": "self",
    "target": "enemy",
    "force_strength": "mid_high",
    "carrier_velocity": "high"
  },
  "expression": {
    "powerness": 0.72
  }
}
```
