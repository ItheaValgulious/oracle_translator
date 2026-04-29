# motion.py

这个文件实现第一版 `8 邻域` 交换式运动。

## 对外接口

- `apply_motion(grid, registry, dt)`
  - 基于速度、剩余意图和“更轻目标交换”规则更新移动结果。

## 依赖的对外接口

- `engine.atmosphere.ambient_air_temperature_for_row`
- `engine.grid.Grid`
- `engine.types.MaterialRegistry`
- `engine.types.MatterState`

## 主要功能

- 先更新统一压力标量场。
- 由压力梯度生成当前帧的局部 `source force` 向量场。
- 维护 `prev_source_force` 与 `force_wave`,并用 `delta = source - lambda * prev_source` 的方式把受力跨帧往连通液体内部传播。
- 求解器只按 `MatterState` 分成 `solid/liquid/gas` 三条大分支,不再对 `powder`、`water`、`steam` 这类具体材质名做运动特判。
- 同一物态内部的差异全部从 `VariantDef` 的通用参数读取,包括:
  - `mobility`
  - `pressure_response`
  - `gravity_scale`
  - `buoyancy_scale`
  - `thermal_motion_scale`
  - `wind_coupling`
  - `wind_vertical_factor`
  - `downward_blocked_diagonal_fallback`
  - `velocity_decay`
- 承重静态固体当前通过 `mobility=0`、`pressure_response=0` 等参数保持不动,而不是走独立的求解器类型。
- 对这类“当前没有任何平移能力”的变体,CPU `motion` 会先做一次通用参数判定,直接跳过昂贵的方向候选与抢占搜索,只保留速度衰减和 blocked impulse 结算。
- 非承重固体可通过 `wind_coupling` 和 `downward_blocked_diagonal_fallback` 获得“受风拖拽”和“正下受阻时只试左右下对角”的行为,从主 `motion` 本身形成坡面。
- 液体随机扰动受 `grid.liquid_brownian_enabled` 运行时开关控制,主要铺开仍依赖横向压差而不是放大随机项。
- `blocked_impulse` 受 `grid.blocked_impulse_enabled` 运行时开关控制; 关闭后,速度求解不再读取旧的残余意图,并会把已有 `blocked_x/y` 清零。
- “首选离散方向受阻时继续尝试最近夹角方向”的 fallback 受 `grid.directional_fallback_enabled` 控制,并只会考虑与当前 `desired` 方向夹角小于等于 `grid.directional_fallback_angle_limit_degrees` 的候选。
- 当某个变体开启 `downward_blocked_diagonal_fallback` 且主方向仍向下时,求解器会把 fallback 收窄成“左下 / 右下”两个 `<= 45°` 候选。
- 当液体下方不可交换时,会额外放大横向压力响应; 深水区还会按局部压头进一步增强横向推力。
- 气体会根据当前温度修正后的等效密度决定上浮或下沉,并使用更强的温度相关布朗运动。
- `empty` 空空气格按所在高度的背景环境温度判断“偏离环境”的程度,并能被局部 pressure 卷入流场,用于表达热空气对流。
- 同为气体/空气的垂直交换按移动方向比较等效密度: 向上必须更轻,向下必须更重。
- 环境 `empty` 空气包不会主动替换显式气体包,显式气体自己负责穿过环境空气传播,避免空气背景反向吞掉具体气体团。
- 离散移动仍按与当前速度夹角最小的 `8 邻域` 目标选择。
- 对液体自由表面或外侧壁,会给“纯水平向外”的候选方向额外偏置,减少边缘锯齿滴落。
- 当多个方向分数接近时,会用轻量哈希扰动和按 `step_id + x + y` 交替的左右平局打破规则去偏置。
- 逐格扫描顺序按 `step_seed` 交替镜像,减少固定左上到右下的处理偏置。
- 主 `motion` 拆成“dense first / gas second” 两段,先让非气态物质交换,再让空气/气体交换,减少空气抢格阻挡下落。
- occupied-target 交换时,被顶开的目标格也会先应用自身受力和剩余意图,再回填到 source 位置。
- 当物质移入 `empty` 时,原位置留下的空气会保留源物质和目标空气二者中较高的温度。
- 当前每个外部 `step` 先跑一次完整 `motion`,再跑 2 次 `liquids_only` relaxation; 远程液体传力仍主要依赖跨帧 `force_wave`。
