这份文档描述《言灵炼金师》当前的高层方向。

当前项目不再把第一版运行时建立在 Godot 上。
新的第一阶段目标是：

- 保留现有模型与 `model_socket -> magic_socket` 的整体思路
- 先实现一个独立的体素引擎原型
- 先把物理模拟骨架写对
- 已经接上一个最小 `pyglet + ModernGL` demo 壳
- 继续为真正的 GPU 后端预留空间

---

### 一、 核心玩法

**1. 语义驱动施法**

玩家通过语言描述魔法,模型先输出模板化结构,运行时再把它展开成真正进入世界的物质与运动。

**2. 涌现式体素博弈**

游戏不依赖预设的“火克水”数值表。
战斗的核心来自体素物质之间的真实互动：

- 结构会失撑、脆化、粉化
- 液体会流动、改道、积压
- 高温会传导、熔化、蒸发
- 腐蚀、点燃、毒化会改变地形和战场

**3. 世界可持续改写**

魔法不仅打敌人,也持续改写环境：

- 岩石被加热后熔成熔岩
- 沙被烧成玻璃
- 冰墙可临时搭桥或封路
- 酸液能慢慢蚀穿支撑结构

---

### 二、 运行时方向

当前原型采用：

- Python
- CPU 模拟核心
- `pyglet + ModernGL` 作为当前桌面 demo 壳

当前这一轮已经实现：

- 材质表与格子状态
- `fixpoint/platform` 支撑网络
- `powder/liquid/gas/fire/molten` 运动
- `vel + blocked_impulse` 驱动
- 热传导
- 家族内相变
- `fire/acid/poison/tar` 反应
- 一个可交互桌面 demo

当前仍不做：

- Godot 集成
- 真正的 GPU 模拟后端
- 整块刚体坍塌
- `fill_ratio`

---

### 三、 体素模拟核心

运行时的基本单位是 cell。

每个 cell 保存动态信息：

- 当前材质家族与变体
- 当前速度向量
- 当前未实现运动残差 `blocked_impulse`
- 当前温度
- 当前支撑值 `support_value`
- 当前结构完整度 `integrity`
- 当前传播代数 `generation`
- 当前形态年龄 `age`
- 结构标志位,例如 `fixpoint`

材质的静态信息则统一放在材质表中：

- 密度
- 硬度
- 摩擦
- 黏度
- 导热率
- 热容量
- 是否承重
- 是否传播支撑
- 相变阈值
- 反应类型与强度

---

### 四、 首批材质族

第一版核心玩法集固定为：

- `stone`
- `sand`
- `glass`
- `iron`
- `water`
- `acid`
- `poison`
- `tar`
- `fire`

关键闭环包括：

- `stone -> magma -> stone`
- `sand -> molten_glass -> glass`
- `glass <-> molten_glass`
- `iron <-> molten_iron`
- `water <-> ice <-> steam`
- `acid_liquid -> acid_gas`
- `poison_liquid -> poison_gas`
- `tar_liquid -> tar_smoke/fire`

---

### 五、 当前原则

**1. 先做 CPU 原型**

先验证规则和手感,不要在第一轮就被渲染壳和 GPU 工程细节拖住。

**2. 先做渐进粉化坍塌**

结构失撑后不是整块刚体掉落,而是 `integrity` 逐步下降并转成同家族粉体。

**3. 先把动态状态与静态材质分层**

不要把 `magic_socket.subject` 里的静态物理参数直接复制到每个格子。
每格只存动态状态,静态材质参数统一查材质表。
