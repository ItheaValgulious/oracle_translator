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
