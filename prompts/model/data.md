# Model Data

本文描述当前模型训练数据如何生成.

## 1. 当前阶段训练什么

当前阶段训练目标不再是完整 runtime 细字段.
而是训练 `text -> model socket`.

也就是:

- `material_template`
- `reaction_template`
- `release_template`
- `motion_template`
- `motion_direction`
- `origin`
- `target`
- `powerness`

## 2. 当前阶段不训练什么

暂不训练:

- `backfire`
- `unstable`
- prefix 切片
- 细粒度物理属性头
- 细粒度 reaction 属性头

## 3. 数据来源

当前阶段训练数据应主要来自 4 部分:

### 3.1 Assistant 手写 seed

由 assistant 手写高覆盖 seed 咒语.
当前仓库已有:

- `data/source/manual_spell_seeds_v2.jsonl`

它的作用:

- 作为 few-shot 例句
- 作为人工可审阅样本
- 作为风格分布锚点

### 3.2 spell -> model socket

把完整咒语翻译成第一版 `model socket`.

### 3.3 model socket -> spell

从模板化结构反推咒语.

### 3.4 random spell -> model socket

先按 recipe 生成随机咒语, 再独立翻译成 `model socket`.

## 4. 覆盖分布

当前种子和生成数据必须覆盖:

- 半文半白
- 文学描写
- 仪式祈请
- 生活化
- 口语化
- 现实参照
- 人类巧思/钻规则空子

特别要包含:

- 化学物质尝试
  - `U235`
  - `F`
  - `Cl2`
- 规则试探
  - “能摧毁一切事物的气体”

## 5. 数据标注格式

当前阶段每条训练样本至少应包含:

```json
{
  "id": "sample_xxx",
  "text": "完整咒语",
  "model_socket": {
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
  },
  "meta": {}
}
```

## 6. 当前最重要的质量标准

不是“词多华丽”, 而是:

1. 文本和模板结构是否真的对应
2. 口语/生活化是否足够多
3. 文学描写是否足够具体
4. 人类巧思和试探性表达是否被覆盖

## 7. 代码使用的 prompt

代码侧真正运行的 prompt 不在 `prompts/`.
而在:

- `src/prompts/spell_generation_prompt_v2.md`
- `src/prompts/spell_to_json_prompt_v2.md`
- `src/prompts/json_to_spell_prompt_v2.md`

`prompts/` 目录下的文档主要是给 Codex 和项目设计讨论使用的.
