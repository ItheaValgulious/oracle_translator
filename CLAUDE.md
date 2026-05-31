# 言灵炼金师 (Oracle Translator)

中文奇幻咒语解析 + 体素物理模拟项目，分两大模块。

## 技术栈

- Python, pyglet + ModernGL (渲染), compute shader (GPU 模拟)
- 模型: Qwen3-0.6B-Base, 外部标注用 qwen3.5-27b API
- API: `https://models.sjtu.edu.cn/api/v1` (key 在 api.txt)

## 项目结构

```
src/engine/          # 体素物理引擎 (CPU reference + GPU backend)
  types.py           # CellState, MaterialFamily, VariantDef, MaterialRegistry
  materials.py       # 9 族材质定义 (stone/sand/glass/iron/water/acid/poison/tar/fire)
  grid.py            # 二维格子容器
  motion.py          # 固/液/气运动求解
  thermal.py         # 热传导
  phases.py          # 家族内相变 (固<->液<->气)
  reactions.py       # 化学反应 (burn/corrode/poison...)
  support.py         # 结构支撑网络 (fixpoint 源, 沿连通承重网络传播)
  world.py           # 大世界分页 (WorldChunkStore + ActiveWorldWindow)
  gpu_backend.py     # compute shader GPU backend (~145K)
  atmosphere.py      # 环境温度垂直梯度
  sim.py             # 模拟主步进: support -> reactions -> thermal -> phases -> motion -> collapse
  render.py          # 渲染 (材质/温度双视图)
  demo_app.py        # pyglet 桌面 demo (相机移动, 子步进, 分页统计 overlay)
  scenarios.py       # 默认场景注入

src/slm/             # SLM 数据与 schema
  model_socket_schema.py  # model socket schema 定义、校验、归一化 (7 个模板分类 + politeness)
  data_generation.py      # 训练数据生成管线 (调用外部 LLM API, 含风格配方和 motif 覆盖)
  io_utils.py             # JSON/JSONL 读写工具

src/prompts/         # LLM 标注用的实际 prompt 模板 (代码运行时读取这些文件)
  spell_to_json_prompt.md       # 咒语 -> model socket 标注
  spell_generation_prompt.md    # 从 model socket 生成咒语
  json_to_spell_prompt.md       # model socket -> 咒语 反向生成

scripts/             # 入口脚本 (均为对 src/slm/data_generation.py 或 src/engine/ 的薄封装)
  run_engine_demo.py           # 启动引擎 demo
  benchmark_world_paging.py    # 大世界分页性能测试
  build_seed_spells.py         # 生成种子咒语 -> data/source/manual_spell_seeds.jsonl
  generate_spells.py           # 从种子生成更多咒语 -> data/source/generated_spells.jsonl
  translate_spells_to_json.py  # 咒语 -> model socket -> data/source/spell_to_model_socket.jsonl
  run_random_spell_to_json.py  # 随机咒语生成 + model socket 标注全流程
  run_random_json_to_spell.py  # 随机 model socket -> 咒语全流程
  json_to_spell.py             # 通用: JSON 转咒语

configs/parser_v1.yaml  # 模型训练配置
tests/
  test_engine_core.py         # 引擎核心测试 (~95K, 覆盖各子系统)
  test_model_socket_schema.py # model socket schema 测试
data/
  source/                     # 生成的数据产物 (JSONL)
  logs/                       # API 调用日志
```

## 关键设计决策

### Model Socket (第一版, text -> structured JSON)
- **只做模板分类**, 不做细粒度物理属性预测。模型输出后由模板系统展开为完整运行时结构。
- 8 个输出头: `material_template` (21类), `reaction_template` (6类), `release_template` (2类), `motion_template` (6类), `motion_direction` (8类), `origin` (4类), `target` (3类), `politeness` (二值训练, 连续推理)
- `politeness` 在训练数据中为 0/1, 推理时取概率作为连续值, 运行时映射为 `powerness`
- 数据生成**必须调用外部模型**, 不允许本地模板/规则回填。失败直接报错并留日志。

### 体素引擎
- 每格存动态状态 (family_id, variant_id, velocity, temperature, support_value, integrity, age, generation, flags), 静态参数统一查 `MaterialRegistry`
- 运动: `vel + blocked_impulse` 驱动, 固/液/气分层交换, 非气态先于气体
- 支撑: fixpoint 为源, 沿连通承重网络逐帧一格传播, 无距离衰减。失撑后渐进粉化为同族粉体, 不做整块刚体坍塌。
- CPU 先做 reference 验证规则, 再搬到 GPU compute shader
- 大世界: `WorldChunkStore` (有限大 chunk 持久层) + `ActiveWorldWindow` (滑动模拟窗), 分页"小步高频", staging 区延迟写回

### 数据格式
- 当前主链路: `text <-> model socket`, 旧 `runtime_b` 格式已废弃
- 旧数据 (`data/raw/curated_v1.jsonl`, `data/processed/*`) 不能直接当训练主数据用

## 开发原则
- 模拟层与渲染层最终解耦, 不要求一帧渲染只对应一次模拟
- 追求涌现式物理互动, 不依赖预设"火克水"数值表
- 魔法可持续改写世界环境
- **网格尺寸不可缩减**: active window 672x412 是设计基准, 不能通过缩小网格来提升帧率。性能优化必须从 shader 本身、调度策略、异步化等方向入手。

## 测试方式

- 测试游戏功能时, **必须通过 debug server (HTTP, port 9123) 发指令来运行真实评测**, 不能只跑短时间 timeout 观察.
- debug server 支持的端点: `/status`, `/teleport?x=&y=`, `/spawn?type=A|B|C`, `/heal?amount=`, `/fps` 等.
- 先 `python -m src.game` 启动游戏, 再用 `curl http://localhost:9123/...` 发送指令验证行为.