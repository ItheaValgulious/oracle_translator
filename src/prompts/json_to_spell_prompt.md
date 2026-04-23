你在为中文奇幻施法解析器生成反向训练语料.

任务:
根据给定的 `model socket`, 写出一句完整中文咒语.

只输出合法 JSON:

```json
{
  "text": "一句完整咒语"
}
```

要求:

1. 回答的第一个字符必须是 `{`.
2. 不要输出解释, 不要输出代码块, 不要输出 `<think>`.
3. 只生成一句完整句子.
4. 优先使用英文逗号和英文句号.
5. 必须像玩家真的可能说出来的话.
6. 不要写成字段名拼接句.
7. `politeness` 在输入中只会是 `0` 或 `1`.
8. `politeness = 1` 时, 越允许更书面, 更文学, 更像正式咒辞.
9. `politeness = 0` 时, 越允许更口语, 更临场, 更像玩家直接喊出来的话.
10. 仍然必须体现:
   - `material_template`
   - `reaction_template`
   - `release_template`
   - `motion_template`
   - `motion_direction`
   - `origin`
   - `target`

输入会提供:

- `model_socket`
- `recipe_name`
- `recipe_guidance`
- `reference_examples`

请根据这些输入生成一句新的中文咒语.
