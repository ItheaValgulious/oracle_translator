# Sockets — Model Socket & Magic Socket

本文档定义从玩家语言到游戏魔法的两层数据接口。

## 总览

```
玩家文本 → [SLM 模型] → Model Socket → [模板展开器] → Magic Socket → [体素引擎]
                          (第一层)                       (第二层)
```

- **Model Socket**: 模型直接从自然语言预测的模板化结构, 只做分类不做细粒度数值预测
- **Magic Socket**: 模板展开后的完整运行时结构, 包含所有体素引擎消费的物理参数

---

# Model Socket (第一层: 模型输出)

第一版模型 (Qwen3-0.6B) 直接从玩家咒语文本预测的闭集分类结构。

## 顶层结构

```json
{
  "subject_kind": "summon_material",
  "subject":    { "material_template": "<string>" },
  "reaction":   { "reaction_template": "<string>" },
  "release":    { "release_template": "<string>" },
  "motion":     {
    "motion_template":  "<string>",
    "motion_direction": "<string>",
    "origin":           "<string>",
    "target":           "<string>"
  },
  "expression": { "politeness": <0 或 1, 推理时 [0,1] 连续值> }
}
```

共 8 个分类头: 7 个模板分类 + 1 个 politeness。

---

## 字段详解

### `subject_kind`

固定值, 当前只支持一种。

| 值 | 含义 |
|---|------|
| `summon_material` | 召唤物质: 在游戏世界中创造/召唤出实体物质 |

---

### `subject.material_template`

表示被召唤的主体属于哪种物质模板。**不是所有物理属性的总和, 而是最核心的主体类别。**

| 值 | 含义 | 典型咒语示例 |
|---|------|------------|
| `granite` | 花岗岩 / 石锋 / 岩刺 / 石枪 | "地底给我顶出几根花岗岩枪" |
| `obsidian` | 黑曜石, 比花岗岩更硬更脆, 颜色更深 | "黑曜石壁从地底升起" |
| `earth` | 泥土 / 土块 / 土地 | "前方地面翻起, 土块垒成矮墙" |
| `sand` | 沙砾 / 流沙 / 沙暴 | "沙从脚前卷起, 成一片沙浪" |
| `water` | 水 / 清水 / 水流 | "水自空中凝出, 向前冲刷" |
| `ice` | 冰 / 冰晶 / 冰棱 / 冰刺 | "冰刺从地底突起, 一排排戳向前面" |
| `steam` | 蒸汽 / 水雾 / 白汽 / 寒息(低温蒸汽) | "白雾贴地漫去, 一路把地面冻得发白" |
| `fire` | 火 / 烈焰 / 火潮 / 圣火 / 野火 | "炽焰出掌, 替我开前路" |
| `light` | 光 / 光束 / 辉光 / 激光般白线 / 亮尘 | "一束耀眼的白光从掌心射出" |
| `wind` | 风 / 气流 / 旋风 | "一阵狂风从身后掀起" |
| `lightning` | 闪电 / 电弧 / 雷光 / 电芒 | "电弧从指尖弹出, 直劈前方" |
| `acid` | 酸液 / 强酸 / 腐蚀液 | "那股刺鼻的腐蚀液贴地漫去" |
| `poison_slurry` | 毒烟 / 毒雾 / 瘴气 / 灰烬云 | "毒雾从身前漫开, 裹住那一排敌影" |
| `tar` | 黑沥青 / 粘稠黑液 / 蜜胶 / 焦油 | "黑沥漫地, 缚其双足" |
| `explosive_slurry` | 熔岩 / 岩浆 / 滚烫灼液 | "滚烫的岩浆从地缝翻出, 一路向前涌" |
| `grass` | 草 / 青草 / 草叶 | "青草从脚下蔓延, 盖住前方的地面" |
| `wood` | 木头 / 荆棘 / 藤蔓 / 树根 | "藤蔓从地底窜出, 缠向前方" |
| `glass` | 玻璃 / 晶屑 / 玻璃碎片 / 盐晶 | "玻璃碎片像雨一样从上方落下" |
| `iron` | 铁 / 铁砂 / 金属砂流 | "铁砂如黑浪从掌心喷出" |
| `quicksilver` | 水银 / 液态金属 / 银亮流体 | "银流从掌中淌出, 像水银一样滑过石面" |
| `unknown` | 无法归入以上模板的极端或不常见物质 | "纸页成山, 试卷洪流" → 无稳定模板, 用 unknown |

**原则**: 现实参照 (如 "像激光切割一样") 不改变主体分类, 仍选最核心的物质模板 (`light` 或 `fire`), 不发明新模板。

---

### `reaction.reaction_template`

表示主体接触其他对象后的**主反应模板**, 不是所有控制效果的总称。

| 值 | 含义 | 判定规则 |
|---|------|---------|
| `none` | 无转化反应, 主体靠自身物理性质 (黏, 重, 覆盖, 堆积) 起作用 | "黑沥漫地, 缚其双足" — 粘住是物理效果, 不是反应 |
| `burn` | 接触后燃烧/点燃/灼烧目标 | "给我一股只烧装备不烧人的火" |
| `corrode` | 接触后蚀空/腐蚀/咬穿目标, 目标转化为空 | "来点F, 谁挡在前面就先给它咬掉" |
| `freeze` | 接触后冻结/结霜/封冻目标 | "寒息过处, 水面结成了冰" |
| `poison` | 接触后毒化/侵体/染毒目标 | "毒雾渗进盔甲, 把那人的皮肤染成青紫" |
| `grow` | 接触后长出/蔓生/生根/增殖 | "藤蔓攀上石壁, 在缝隙里扎下根" |

**关键规则**: 不要把 "困住, 黏住, 压住, 挡住" 写成 reaction。如果一句话主要通过物质本身的黏/重/覆盖/堆积起作用, `reaction_template = none`。

---

### `release.release_template`

表示主体是怎样被释放到世界中的。

| 值 | 含义 | 典型描述 |
|---|------|---------|
| `spray` | 喷射: 快速逐渐释放, 喷/扫/泼/涌, 连续送出 | 火流从掌心持续喷出, 一波接一波 |
| `appear` | 显现: 瞬间全部出现, 突然长出/顿时立起/整团现身 | 石锋从地底突刺而出, 一排排立在身前 |

判断原则:
- "把那片火往前推" → `spray` (连续推进)
- "石锋起地" → `appear` (瞬间立起)
- "黑沥漫地" → 如果强调整团铺开, 更像 `appear`
- "银流出掌" → `spray` (连续淌出)

---

### `motion.motion_template`

表示主体被释放后的整体运动方式。

| 值 | 含义 |
|---|------|
| `none` | 无主动运动, 释放后由物理引擎接管 (重力下坠, 液体铺开等) |
| `fixed` | 悬停/钉住/定在原地, 有空间位置但不随重力移动 |
| `flow` | 流动: 整体朝某方向流去/冲去/压去, 像火焰喷射或水流冲击 |
| `vortex` | 涡旋: 围绕粒子群中心回旋, 像小型旋风或毒雾盘旋 |
| `rotation` | 旋转: 围绕发射点或某固定轴转动, 像环绕自身的护盾 |
| `vibration` | 振动: 在原地或小范围内快速往复, 像共鸣或震颤 |

---

### `motion.motion_direction`

表示主体的主运动方向。不允许 `none` — 必须有明确方向。

| 值 | 含义 | 典型咒语 |
|---|------|---------|
| `forward` | 向前方/远处/施法者面朝的方向 | "直直压过去" |
| `backward` | 向后方/反方向 | "反卷身后" |
| `up` | 向上/升空 | "火柱冲天" |
| `down` | 向下/下沉/砸落 | "玻璃碎片像雨一样往下落" |
| `target` | 指向目标 (当前锁定敌人或指定对象) | "直取那个人" |
| `self` | 指向施法者自身 | "绕回我身边" |
| `front_up` | 前上方, 抛射/升起后向前 | "从前方升起, 越过屏障" |
| `front_down` | 前下方, 向地面/低处推进 | "贴地向前翻滚" |

---

### `motion.origin`

表示主体从哪里出现。

| 值 | 含义 | 典型咒语 |
|---|------|---------|
| `self` | 从施法者自身/掌心/身体出发 | "自掌心起" |
| `back` | 从施法者身后出现 | "自背后升起一圈火焰" |
| `front_up` | 从施法者前上方出现 | "从前上方落下" |
| `front_down` | 从施法者前下方/地面/脚前出现 | "脚前翻开" |

---

### `motion.target`

表示主体主要朝向谁 / 以谁为参考对象。

| 值 | 含义 | 典型咒语 |
|---|------|---------|
| `self` | 以施法者自身为目标/环绕自身 | "环归我身" |
| `enemy` | 以敌人为目标 | "直取前敌" |
| `none` | 无特定目标, 纯方向性运动 | "向前铺开" |

---

### `expression.politeness`

综合文体值。**不是字面礼貌, 而是咒语的文学性/仪式性/书面程度 vs 口语/临场程度**。

- 训练时: `0` (口语/生活化/临场喊话) 或 `1` (文学/仪式/书面/正式咒辞)
- 推理时: 取模型输出概率, 得到 `[0, 1]` 连续值
- 运行时映射: `powerness = politeness` (游戏的施法威力系数)

| 值 | 含义 | 典型咒语 |
|---|------|---------|
| `≈0` | 口语, 生活化, 临场喊话, 不像正式咒辞 | "给我来点火, 往前烧." |
| `≈0.5` | 介于口语和书面之间 | "来一束像激光切割的灼热线, 直直划过去." |
| `≈1` | 文学, 仪式, 书面, 像正式咒辞/祈请 | "愿明焰垂临此手, 为我焚开前路." |

---

# Magic Socket (第二层: 模板展开后的运行时结构)

Magic Socket 是 Model Socket 经模板展开器展开后的完整结构, 直接消费于体素引擎。

**当前状态: 设计已定, 展开器代码尚未实现。**

模板展开规则: 每个 `_template` 字段查表得到一组默认物理参数, 再叠加 `motion` 各字段的方向/位置信息写入运行时字段。

## 顶层结构

```json
{
  "subject_kind": "summon_material",
  "subject":    { ... 展开后的完整物质属性 ... },
  "reaction":   { ... 展开后的完整反应参数 ... },
  "release":    { ... 展开后的完整释放参数 ... },
  "motion":     { ... 展开后的完整运动参数 ... },
  "expression": { "powerness": <float> }
}
```

## 展开规则

### Subject 展开

输入: `material_template` (21 选 1)
输出: 以下运行时字段由模板查表填充, 不由模型直接预测:

| 展开字段 | 类型 | 含义 | 示例 (fire) |
|---------|------|------|-------------|
| `material_template` | string | 保留原始模板名, 用于引擎索引材质族 | `"fire"` |
| `color` | string | 物质外观颜色名 | `"orange"` |
| `state` | string | 物态: `solid` / `liquid` / `gas` | `"gas"` |
| `density` | string | 相对密度等级: `very_low` / `low` / `mid` / `high` / `very_high` | `"low"` |
| `temperature` | string | 初始温度等级: `very_low` / `low` / `room` / `high` / `very_high` | `"high"` |
| `amount` | string | 召唤量等级: `very_small` / `small` / `mid` / `large` / `very_large` | `"mid_high"` |
| `hardness` | string | 硬度等级 (仅固体有效) | — |
| `friction` | string | 表面摩擦等级 | — |
| `viscosity` | string | 粘度等级 (仅液体有效) | — |

### Reaction 展开

输入: `reaction_template` (6 选 1)
输出: 展开为具体转化行为参数

| 展开字段 | 类型 | 含义 |
|---------|------|------|
| `reaction_template` | string | 保留原始模板名 |
| `convert_mode` | string | 转化目标: `self` (将目标转化为反应物本身) / `empty` (将目标转化为空格) / `none` (无转化) |
| `reaction_speed` | string | 转化速度等级: `very_slow` / `slow` / `mid` / `high` / `very_high` |
| `reaction_mask` | [string] | 反应可作用的对象类别列表, 如 `["living", "terrain", "item"]` |
| `reaction_direction` | string | 反应扩散方向, 可与 `motion_direction` 不同。例如: 火往前冲 (motion=forward) 但火苗向上蔓 (reaction=up) |
| `generation_limit` | int | 反应最多传播代数, 防无限连锁。默认 2 |

**模板映射表**:

| reaction_template | convert_mode | 典型 speed | 说明 |
|-------------------|-------------|-----------|------|
| `none` | `none` | — | 不转化任何目标 |
| `burn` | `self` | `high` | 目标逐渐变成火/燃烧态 |
| `corrode` | `empty` | `mid` | 目标被蚀空, 转化为空格 |
| `freeze` | `self` | `mid` | 目标逐渐冷冻为冰/霜 |
| `poison` | `self` | `slow` | 目标逐渐被毒化 |
| `grow` | `self` | `mid` | 接触到的地方长出同类物质 |

### Release 展开

输入: `release_template` (2 选 1)
输出: 展开为具体释放行为参数

| 展开字段 | 类型 | 含义 |
|---------|------|------|
| `release_template` | string | 保留原始模板名 |
| `release_profile` | string | 释放形态: `stream` (流束) / `cone` (锥形) / `burst` (爆散) / `pool` (铺开) |
| `release_speed` | string | 释放速率等级: `very_slow` / `slow` / `mid` / `high` / `very_high` |
| `release_spread` | string | 释放扩散度等级: `none` / `very_low` / `low` / `mid` / `high` |

| release_template | 默认 profile | 典型 speed | 说明 |
|-----------------|-------------|-----------|------|
| `spray` | `stream` | `high` | 快速喷出, 连续释放 |
| `appear` | `burst` | `very_high` | 瞬间全量出现 |

注意: `release_duration` 不作为独立字段, 由 `amount / speed` 推导。

### Motion 展开

输入: `motion_template` + `motion_direction` + `origin` + `target`
输出: 方向/位置字段直接保留, 额外补强度参数:

| 展开字段 | 类型 | 含义 |
|---------|------|------|
| `motion_template` | string | 保留原始模板名 |
| `motion_direction` | string | 保留原始方向 |
| `origin` | string | 保留原始起点 |
| `target` | string | 保留原始目标 |
| `force_strength` | string | 运动力度等级: `very_weak` / `weak` / `mid` / `high` / `very_high` |
| `carrier_velocity` | string | 载体速度等级 (物质的初速度): `none` / `low` / `mid` / `high` / `very_high` |

### Expression 展开

| Model Socket | Magic Socket | 规则 |
|-------------|-------------|------|
| `politeness` (模型输出概率, [0,1]) | `powerness` (威力系数, [0,1]) | `powerness = politeness` (当前阶段直接映射) |

`powerness` 在运行时消费为:
- 同义施法下的威力倍率
- 影响 `amount` 缩放, `force_strength` 缩放, `reaction_speed` 缩放等
- 实际效果: 越正式的咒辞 → 越高的 powerness → 越强的魔法效果

---

## 完整展开示例

Model Socket:
```json
{
  "subject_kind": "summon_material",
  "subject":    { "material_template": "fire" },
  "reaction":   { "reaction_template": "burn" },
  "release":    { "release_template": "spray" },
  "motion":     {
    "motion_template":  "flow",
    "motion_direction": "forward",
    "origin":           "self",
    "target":           "enemy"
  },
  "expression": { "politeness": 0.72 }
}
```

展开为 Magic Socket:
```json
{
  "subject_kind": "summon_material",
  "subject": {
    "material_template": "fire",
    "color": "orange",
    "state": "gas",
    "density": "low",
    "temperature": "high",
    "amount": "mid_high",
    "hardness": null,
    "friction": null,
    "viscosity": null
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

## 数据流总结

```
玩家输入 "愿明焰垂临此手, 为我焚开前路"
    │
    ▼
[SLM 模型推理]
    │
    ▼
Model Socket:
  material_template = fire
  reaction_template = burn
  release_template   = spray
  motion_template    = flow
  motion_direction   = forward
  origin             = self
  target             = enemy
  politeness         = 0.91  (连续概率 → 非常正式)
    │
    ▼
[模板展开器] (每个 template 查表, 补物理参数)
    │
    ▼
Magic Socket:
  subject:  {fire, gas, low density, high temp, ...}
  reaction: {burn → self, speed=high, mask=[living,terrain], ...}
  release:  {spray → stream, speed=high, spread=mid_low}
  motion:   {flow forward from self toward enemy, force=mid_high}
  powerness = 0.91
    │
    ▼
[体素引擎] 在格子世界里创造火, 施加流动力, 开启燃烧反应...
```