这份文档描述《言灵炼金师》当前的高层方向。

当前项目不再把第一版运行时建立在 Godot 上。
新的第一阶段目标是：

- 保留现有模型与 `model_socket -> magic_socket` 的整体思路
- 先实现一个独立的体素引擎原型
- 先把物理模拟骨架写对
- 已经接上一个最小 `pyglet + ModernGL` demo 壳
- 已经接上第一版 compute shader GPU backend

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
- CPU reference 模拟核心
- compute shader GPU backend
- `pyglet + ModernGL` 作为当前桌面 demo 壳

当前这一轮已经实现：

- 材质表与格子状态
- `fixpoint/platform` 支撑网络
- `solid/liquid/gas` 运动,同一物态内部差异由材质参数表驱动
- `vel + blocked_impulse` 驱动
- 热传导
- 家族内相变
- `fire/acid/poison/tar` 反应
- compute shader 版 support / motion / thermal / phases / reactions / collapse
- 一个可交互桌面 demo
- demo 可显示材质视图与温度视图
- 温度视图现在在室温附近使用更高对比的调色带,弱环境梯度也能直接看出来
- demo 支持每帧多次模拟子步进
- demo 相机现在支持按住 `W/A/S/D` 或方向键连续移动,不再是按一下跳一段
- 支撑信号可沿连通承重网络逐帧一格、无距离衰减地向远处传播
- 热量可通过空气和固体一起扩散
- 热空气现在除了导热,还会作为空气格在空域内发生更强的温度相关布朗运动与上浮
- 空环境现在带有弱的垂直背景温度梯度: 越靠近地面越暖,越靠近高处越冷
- 默认新建出来的 `empty` 空空气现在就会按所在高度初始化到这条背景梯度,不会先以全图 `20°C` 触发一轮假风
- `empty` 空气会弱地向这条背景梯度回归,即使没有火焰也会逐步形成更像环境温度层结的空域
- 热空气的局部 pressure 现在也会真正进入空气 motion,让热羽流能卷动周围空气形成更复杂的局部对流风
- 非承重固体变体当前可通过材质参数读取邻近空气/气体包已经形成的真实速度,作为横向更明显、竖向更弱的 wind drag,让空气流动能直接带出可见的推沙效果
- 气体/空气垂直交换按移动方向比较等效密度: 更轻的热气体向上换位,更重的冷气体向下换位
- 液体-气体相界面会额外加强热交换,避免被水包裹的水蒸气团长时间像绝热团块一样保温
- 水蒸气团内部也会加强热交换,让成团蒸汽的中心热量能传到接触水的边界
- 水蒸气在普通空气中有明显热容量和凝结滞后,不会离开火焰后立刻变水
- 冰有较大的等效潜热和融化滞后,常温空气中不会几帧内全部融化
- 液体支持侧向铺开、重液下沉和相邻交换
- 大水体当前每步会追加轻量液体 relaxation,用于更快降低局部高水峰
- 开启 `downward_blocked_diagonal_fallback` 的非承重固体变体,在主方向是向下且正下方受阻时,当前会只在 `<= 45°` 的左下和右下候选里继续释放意图,不用额外粉体 pass 也能形成坡面
- 方向并列和多格抢占同一目标时,CPU/GPU 都会按 `step_id` 交替镜像去偏置,减少固定 `45°` 左上到右下空线
- “首选方向受阻时退而求其次找最近可移动方向”的机制现在带运行时开关,且只会在和当前期望方向夹角小于等于 `directional_fallback_angle_limit_degrees` 的候选里找; 当前默认上限为 `45°`
- 主 `motion` 当前会先让非气态物质完成交换,再让空气/气体完成交换,减少空气抢先占掉本该让沙子落入的空格
- 气体支持逐步随机漂移扩散
- 酸液可在成功腐蚀后自耗
- 大世界活动窗当前会更早、更小步地分页平移,以减少单次切页卡顿
- GPU 路径下被逐出活动窗的区域当前会先留在 staging 队列里,只有相机空闲一小段时间后才逐步回写 world store
- GPU staging 区写回 `WorldChunkStore` 时,当前直接把 region 内仍有支撑的承重格作为冻结区锚定快照一并写回,不再在 flush 当帧对整张世界做全图 `recompute_anchored_support`
- `WorldChunkStore.rect_has_stored_cells()` 当前只会在矩形内真实命中已存储 cell 时才返回 true,不会因为同 chunk 其他位置有内容就误把整条 incoming strip 判成非空
- 新暴露条带如果对应的是空世界默认区,当前会直接在 GPU 上按 world row 环境温度填成默认 `empty`
- 新暴露条带如果确实命中已存储 cell,当前会直接把 world store 的 chunk 稀疏内容打包成 GPU state bytes 写入 incoming strip,不再先走完整 `read_rect + GridSlice(CellState) + pack + write_region` 慢链
- 冻结区外部支撑锚点当前只重建活动窗四条边,并压成 compact edge buffer 供 support shader 读取,切页时不再上传整张 anchor 纹理
- incoming 条带的 transient 清理当前改成 GPU region clear,只清下一步真正会被读取的 transient 纹理,不再对整组六张 transient texture 做 CPU 侧分块上传
- GPU staging region 当前除了普通尺寸分桶复用,还会优先把常见横向/纵向窄条带放进固定 atlas 槽位,减少切页时的 staging 分配和释放尖刺
- 停止移动后的 GPU staging writeback 当前会按小 slice 分多次 readback + write_rect,而不是一口气 flush 整条 strip
- demo 相机当前会限制单帧用于移动的有效 `dt`,避免掉帧后单帧位移暴涨,进一步触发连续多次分页
- demo overlay 当前会额外显示分页次数、最近/最大切页耗时、pending writeback 压力和分页分阶段耗时,便于直接观察分块流送性能

当前仍不做：

- Godot 集成
- 整块刚体坍塌
- `fill_ratio`

---

### 三、 体素模拟核心

运行时的基本单位是 cell。

每个 cell 保存动态信息：

- 当前材质家族与变体
- 当前速度向量
- 当前未实现运动残差 `blocked_impulse`
- 当前温度,并在运动或交换时跟随 cell
- 当前支撑新鲜度剩余秒数 `support_value`
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
- 创建时默认温度,只用于新建 cell
- 是否承重
- 是否传播支撑
- 相变阈值
- 反应类型、反应强度与反应能量
- 其中 `reaction_energy` 只在反应阶段改变当前格自身温度,后续向周围扩散统一交给热传导
- 固体没有热运动
- 液体只有极微小的温度相关布朗运动
- 气体和热空气有明显温度相关布朗运动,并因高温降密而上浮; 当前高温会额外放大气体布朗运动的温度项

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

**1. 先做 CPU 原型,再把规则搬到 GPU**

CPU 路径仍然是 reference 实现,用于验证规则和做 fallback。
桌面 demo 默认优先走 compute shader backend,避免逐格 Python 扫描的帧耗时。

后续细化方向包括:

- 运动应把一次离散移动视作“消耗一部分轴向意图”,而不是把整格位移直接反扣成反向冲量
- demo 层允许模拟子步进数与调试视图切换
- 长期上应把模拟与渲染彻底解耦,不要求一帧渲染只对应一次模拟

**2. 先做渐进粉化坍塌**

结构失撑后不是整块刚体掉落,而是 `integrity` 逐步下降并转成同家族粉体。

**3. 先把动态状态与静态材质分层**

不要把 `magic_socket.subject` 里的静态物理参数直接复制到每个格子。
每格只存动态状态,静态材质参数统一查材质表。
