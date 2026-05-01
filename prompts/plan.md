# Runtime Parser Fine Plan

当前细架构不再把“风格”压成少数几个离散标签。
新的 parser 应拆成两层：

1. 语义解析层
   - 负责把玩家文本稳定映射到 `subject/release/motion/targeting` 等运行时槽位。
   - 目标是“尽量能放出来”, 而不是只奖励某一种文风。

2. 文风评估层
   - 负责评估表达的语体混合, 世界观贴合度, 咒语感, 以及由此派生的魔力增益。
   - 不要求 one-hot 分类。
   - 必须允许同一句同时带有：
     - 半文半白
     - 文学描写
     - 生活化
     - 口语化
     - 仪式祈请
     - 现代物象引用

推荐的新 head 体系：

- `status_head`
- semantic categorical heads
- semantic binned heads
- `reaction_mask_head`
- `style_family_head`
  - 多标签或软分布。
  - 预测一条输入同时落在哪些语体族。
- `style_axis_heads`
  - 用更细的 ordinal bins 或连续值表示程度。
  - 至少覆盖：
    - `classicalness`
    - `literaryness`
    - `rituality`
    - `colloquiality`
    - `everydayness`
    - `modernity`
    - `directness`
- `worldview_fit_head`
- `incantation_fit_head`
- `magic_gain_head` 或由上述评分派生的 `magic_gain`

其中：

- `style_family` 解决“是什么风格在混”。
- `style_axes` 解决“每种风格有多强”。
- `worldview_fit` 解决“这句话像不像这个世界会说的话”。
- `incantation_fit` 解决“这句话像不像咒语”。
- `magic_gain` 解决“同义施法时, 为什么有的更强”。

---

## Runtime Engine Note

体素 runtime 的后续细化方向额外记录:

- 模拟层与渲染层最终应解耦。
- 不要求一次渲染只绑定一次模拟。
- 更合理的结构是:
  - 模拟按自己的固定时间步持续推进
  - 渲染只采样最近一次完成的稳定状态
  - demo 可临时允许每帧执行多次模拟子步进来验证手感
- 大世界方向当前补充为:
  - 世界存储与活动模拟窗解耦
  - `WorldChunkStore` 负责有限大地图的 chunk 化持久层
  - `ActiveWorldWindow` 负责一个连续的滑动模拟域和 viewport / camera 映射
  - 相机移动不再走整窗 `load_grid()`,而是:
    - 保留重叠区域 cell state
    - 新暴露区域按矩形增量装入
    - `WorldChunkStore.rect_has_stored_cells()` 只按条带矩形内真实命中的存储 cell 判定慢路径,不会因为同 chunk 其他位置有内容就误判
    - 如果新暴露条带在 `WorldChunkStore` 里没有真实存储内容,则直接在 GPU 上按 world row 环境温度填成默认 `empty`
    - 如果条带确实命中已存储 cell,则直接把 world store 的稀疏 chunk 内容打包成 GPU state bytes 写入 incoming strip,避免 `read_rect + GridSlice(CellState) + pack + write_region` 的整链 Python 开销
    - 被逐出活动窗的区域先进入 staging / 待写回队列
    - 只有相机空闲一小段时间后,再逐步回写 CPU world store
    - staged region flush 当前会按小 slice 分片 readback + write_rect,避免停止后单次整条 strip 写回卡顿
    - staging region flush 时直接把 region 当前仍有支撑的承重格写成冻结区 anchored snapshot,避免 flush 当帧做全图 `recompute_anchored_support`
    - incoming strip 的 transient 清理当前使用 GPU region clear,而不是多张 transient texture 的 CPU viewport write
    - external anchor 当前不再走整张 anchor texture upload,而是把四条边压成 compact edge buffer 供 support shader 直接读取
    - staging texture 当前除了普通尺寸复用,还会优先把窄条带写进固定 atlas 槽位,减少频繁分页时的 GPU 纹理分配尖刺
    - 分页策略优先“小步高频”而不是“大步低频”,以降低单次切页开销
  - demo 相机输入层当前还会限制用于移动的单帧有效 `dt`,避免掉帧后一次移动跨过过多 cells,把单帧分页压力进一步放大
  - `pressure / source_force / force_wave` 当前只属于活动模拟窗,不写入 chunk store
  - support 在大世界里增加“冻结区外部锚定”语义:
    - chunk store 记录冻结承重网络是否仍连到真实 `fixpoint`
    - 活动窗边界可以把这些冻结外部连接视为虚拟支撑源
    - 不把整条承重链永久改写成真正的 `FIXPOINT`
    - 当前切页只重建和上传活动窗四条边的 external anchor,并保留分页分阶段耗时统计用于后续继续压热点
