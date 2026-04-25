# Engine Prototype Basic

本文定义新的独立体素引擎原型方向。

当前阶段不再使用 Godot 作为第一版运行时壳。
第一版技术栈改为：

- Python
- pyglet
- ModernGL

当前仓库已经落下：

- `engine core`
- 一个最小 `pyglet + ModernGL` 可交互 demo 壳

也就是：

- 核心模拟已经可以独立步进
- demo 可以开窗,实时渲染网格并用笔刷注入材质
- demo 优先使用 compute shader GPU backend 步进和着色
- CPU 路径保留为 reference 和 fallback

## 1. 当前目标

第一版原型要先验证：

- `8 邻域` 的体素运动
- `fixpoint/platform` 支撑网络
- `platform` 失撑后逐步粉化，而不是整块刚体坍塌
- 每格速度向量与 `blocked_impulse`
- 热传导
- 热空气、气体与液体的温度相关运动
- 家族内相变
- `fire/acid/poison/tar` 的基础反应

## 2. CellState

`CellState` 只存每个格子随时间变化的动态状态，不重复存静态材质参数。

固定字段：

- `family_id`
  - 当前材质家族。
- `variant_id`
  - 当前家族内的具体变体。
- `vel_x`
- `vel_y`
  - 当前主运动倾向向量。
- `blocked_x`
- `blocked_y`
  - 未实现运动残差向量。
  - 不是反弹，不是反向冲量。
  - 下一步会和 `vel` 共同决定新的主方向。
- `temperature`
  - 当前温度,是跟随 cell 移动和交换的动态状态。
- `support_value`
  - 当前收到的支撑强度。
- `integrity`
  - 当前结构完整度。
- `generation`
  - 当前格子是第几代传播产物。
- `age`
  - 当前形态已持续多久。
  - 当前主要给 `fire` 使用。
- `flags`
  - 至少包含 `is_fixpoint`。

## 3. MaterialFamily / VariantDef

静态材质参数不进 `CellState`，而放到材质表中。

### 3.1 MaterialFamily

- `family_id`
- `name`
- `default_variant`
- `collapse_target`
- `variants`
- `phase_map`
- `reaction_profile`
- `render_profile`

其中：

- `collapse_target`
  - 当承重结构失撑或完整度归零时，转成的同家族粉体或碎屑变体。
- `phase_map`
  - 家族内或跨家族的相变规则。

### 3.2 VariantDef

`VariantDef` 定义一个具体变体的静态模拟参数。

- `variant_id`
- `matter_state`
  - 只分:
    - `solid`
    - `liquid`
    - `gas`
- `motion_mode`
  - 例如:
    - `static`
    - `powder`
    - `fluid`
- `density`
- `hardness`
- `friction`
- `viscosity`
- `thermal_conductivity`
- `heat_capacity`
- `support_bearing`
- `support_transmission`
- `base_temperature`
  - 只表示创建 cell 时的默认初始温度,不会把已有 cell 拉回这个温度。
- 相变阈值：
  - `ignite_temperature`
  - `melt_temperature`
  - `freeze_temperature`
  - `boil_temperature`
  - `decompose_temperature`
- `integrity_decay_from_heat`
- `reaction_kind`
- `reaction_strength`
- `reaction_energy`
- `lifetime_mode`
- `render_color` 或 `palette_id`

## 4. 支撑系统

### 4.1 fixpoint 不是材质

`fixpoint` 不是独立材质，而是一个结构角色标志位。

- 主材质仍然可能是 `stone/glass/iron/...`
- `fixpoint` 通过 `flags` 表示
- 正式游戏里可以做到与普通 `platform` 视觉上不可区分

### 4.2 support_value

支撑信号按离散波次传播,但只在承重网络内传播：

- `fixpoint` 每个模拟步发出一个新的支撑波次
- 只有 `fixpoint/platform` 网络传播支撑
- 每个模拟步只向相邻格传播一格,靠多帧传到远处
- 连通的 `platform` 网络不按距离削弱支撑波次
- `generation` 记录已收到的最新波次 id,只有更新的波次才能继续刷新平台
- `support_value` 表示距离失撑还剩多少秒,当前超时时间先设为 10 秒
- `powder/liquid/gas/fire/molten` 不传播支撑

当 `platform` 超过 10 秒没有收到新支撑波次时：

- 结构不会立刻塌
- 而是开始持续降低 `integrity`
- 当 `integrity` 降到阈值以下时，再转为同家族 `powder`

## 5. blocked_impulse

`blocked_impulse` 表示未实现运动残差。

它的语义是：

- 这格本来想沿某个方向流动
- 但这一轮没有完全释放出去
- 剩下的那部分方向性需求会积累到 `blocked_impulse`

它不是：

- 反向回弹
- 法向反作用力的直接存储

当前运动改为:

- 先计算一个统一压力标量场 `pressure`
- 再由压力差生成速度变化
- 最后按 `8 邻域` 里与当前速度夹角最小的方向选择交换目标

当前压力语义是:

- 气体格:
  - 给本格提供 `air_pressure + gas_density` 的局部压力
- 非气体格:
  - 压力逐步回归空气基准压力
- 液体格:
  - 自上而下累积
  - 当前格压力 = 上一格压力 + 当前液体密度

当前受力语义是:

- 先由压强梯度得到本地 `source force`
- 再维护一个跨帧持久化的 `force wave`
- 当前帧注入的源项是:
  - `delta = source - lambda * prev_source`
- `force wave` 会沿连通液体跨帧向外传播
- 当前运动最终使用:
  - `source force + propagated wave`
- 这样远处液体不需要在同一帧内多次 `motion` 迭代,也能逐步吃到来自高压液柱的横向传力

当前速度语义是:

- `platform/fixpoint`
  - 当前仍不移动
  - 但会先更新受压后的 `vec`,为后续扩展留接口
- `powder`
  - 受重力
- `liquid`
  - 在 `powder` 基础上增加极小的温度相关布朗运动
  - 液面整平当前主要靠横向压差,而不是靠放大布朗运动
  - 当下方不可交换时,会显著加强横向压力驱动,加快大尺度铺开
  - 深水区会额外按压头增强横向推力,避免大水堆长时间保持高弧顶
  - 对自由表面和外侧壁,当前还会对“纯水平向外”的离散候选方向做额外偏置,减少边缘格总是优先往斜下掉而形成锯齿
- `gas`
  - 在 `liquid` 基础上按温度修正后的等效密度决定上浮或下沉
  - 当前会使用比液体更强的温度相关布朗运动,避免只剩单向浮沉
- `empty`
  - 当前作为空气格参与对流
  - 只有在温度明显偏离环境或已有残余速度时才会主动移动
  - 垂直方向的空气/气体交换不能只看“目标为空”,而要按移动方向比较温度修正后的等效密度:
    - 向上时允许更轻的当前气体换入更重的目标气体/空气
    - 向下时允许更重的当前气体换入更轻的目标气体/空气

目标格是否可交换:

- 不是只看空不空
- 而是看目标是否更轻
- 比较顺序是:
- 先比物态: `solid > liquid > gas`
- 同物态再比密度
- 空背景仍视作可交换目标
- 但同为气体/空气的垂直交换使用方向感知密度规则,避免冷空气被随机或冲突求解错误地顶上去、热空气错误地下沉

当前交换落地语义还包括:

- source 成功交换到目标格后
- 被顶开的目标格不会只是“原样塞回 source 位置”
- 而是会先应用它自身的受力、随机扰动和剩余意图
- 再作为 displaced cell 落到 source 位置
- 这样可以避免下落固体长期把液体一路抬高成细柱

`blocked_impulse` 当前更接近“剩余移动意图 / 未释放需求”,用于把连续速度拆成多步离散移动。

当前阶段为了让大水体在有限步数内更明显铺平,液体额外 relaxation 以小固定次数开启:

- 每个外部 `step` 当前先跑一次完整 `motion`
- 之后追加 2 次 `liquids_only` 的轻量 relaxation pass
- 液体内部更远处的传力仍主要依赖跨帧的 `force wave`,额外 pass 只用于消除大水堆表面的高尖峰和局部阻塞

## 6. 首批材质族闭环

第一版核心玩法集固定为：

- `stone_family`
  - `stone_platform`
  - `stone_powder`
  - `magma`
- `sand_family`
  - `sand_powder`
  - 高温转 `glass_family.molten_glass`
- `glass_family`
  - `glass_platform`
  - `glass_shards`
  - `molten_glass`
- `iron_family`
  - `iron_platform`
  - `iron_grit`
  - `molten_iron`
- `water_family`
  - `ice`
  - `water`
  - `steam`
- `acid_family`
  - `acid_liquid`
  - `acid_gas`
- `poison_family`
  - `poison_liquid`
  - `poison_gas`
- `tar_family`
  - `tar_liquid`
  - `tar_smoke`
- `fire_family`
  - `fire`

固定相变闭环：

- `stone -> magma -> stone`
- `sand -> molten_glass -> glass`
- `glass <-> molten_glass`
- `iron <-> molten_iron`
- `water <-> ice <-> steam`
- `acid_liquid -> acid_gas`
- `poison_liquid -> poison_gas`
- `tar_liquid -> tar_smoke/fire`

## 7. 反应语义

- `fire`
  - 使用 `age`
  - 在反应阶段按 `reaction_energy` 对自身放热
  - 对周围的升温统一通过后续 `thermal` 的邻格温差传导完成
- `acid`
  - 主打腐蚀
  - 可让结构掉 `integrity`
- `poison`
  - 主打毒液与毒雾扩散
  - 受热后可分解成更危险的毒雾与短火团
- `tar`
  - 高黏液体
  - 易燃

结构材质当前遵循：

- `stone/glass/iron/ice` 不持续燃烧
- 它们只会受热损伤或相变
- 也就是“先烧后塌”，不是着火瞬间失撑

## 8. 当前 demo 交互

当前 demo 已具备:

- 打开桌面窗口
- 实时步进模拟
- 优先使用 GPU compute shader,不可用时回退 CPU reference
- overlay 显示当前 backend、刷新频率和网格/窗口尺寸
- overlay 显示当前模拟子步进数与调试视图
- 可运行时开关液体布朗运动,便于观察其对液体铺开和性能的影响
- 可运行时开关 `blocked_impulse`,便于观察残余意图是否导致液体边缘异常运动
- 鼠标绘制/擦除
- 切换 `stone/fixpoint/sand/water/fire/acid/tar/poison/ice`
- 可切换材质视图 / 温度视图 / 压力视图
- 初始网格分辨率和窗口大小可由启动参数配置
- 窗口可直接拖拽调整大小
- 暂停、单步、重置和清空场景

## 9. 第一版不做什么

- 不做 `fill_ratio`
- 不做整块刚体坍塌
- 不做 Godot 集成
- 不做实时模型 socket 接入

当前阶段是：

- compute shader GPU backend 负责 `support / motion / thermal / phases / reactions / collapse` 和调试着色输出
- CPU 路径保留为同规则 reference 实现
- `pyglet + ModernGL` 继续作为桌面 demo 壳
- demo 当前允许“一次渲染对应多次模拟子步进”
- `support` 当前按“fixpoint 持续发信号, platform 跨多帧逐格保持传播”的方式工作
- 压力场当前已进入 CPU/GPU 两条路径,并参与速度更新
- `solid` 当前已经会更新受压后的 `vec`,但仍不移动
- 重液下沉和液体/气体分层当前都通过统一的交换式 `motion` 完成
- `fire` 和 `steam` 当前按“比空气轻”上浮, `poison_gas` 当前按“高于空气密度”下沉
- 气体当前的布朗运动和浮力都受当前温度影响; 温度越高,等效密度越低
- `liquid/gas/fire` 的扩散当前按质量守恒处理,是移动或交换,不是凭空复制
- 液体当前每个外部 `step` 会追加 2 次轻量 `liquids_only` relaxation,用于让大水堆更快降低峰值
- `acid` 当前在成功腐蚀承重目标后会自耗
- `poison` 当前主要是受热挥发成更易扩散的毒气,本轮不主打地形腐蚀
- 温度当前可以通过空气、`platform/fixpoint` 等固体和流体一起传导
- 液体和可凝结物质气体相邻时,热传导会有相界面加速项,用于让水包裹的蒸汽团能在可感知时间内冷却和凝结; `fire` 不使用这个倍率
- 温度是 cell 的动态状态,运动或交换时随 cell 一起移动
- 物质离开某格时,留下的 `empty` 空气仍保留被加热后的温度,尤其 `fire` 移走或熄灭不会把原位置重置成默认空气温度
- 热空气当前除了导热,还会通过空气格交换向上漂移
