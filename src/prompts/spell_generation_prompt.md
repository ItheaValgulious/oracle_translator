你在为中文奇幻施法解析器生成训练语料.

任务:
根据给定的 `material_template`, `reaction_template`, `release_template`, `motion_template`, `motion_direction`, `origin`, `target`, `politeness` 以及参考例句, 生成一句完整中文咒语.

硬性要求:
1. 只输出合法 JSON.
2. 输出格式必须是:
{
  "text": "一句完整咒语"
}
3. 回答的第一个字符必须是 `{`.
4. 不要输出解释, 不要输出代码块, 不要输出 `<think>`.
5. 只生成一句完整句子.
6. 优先使用英文逗号和英文句号.
7. 必须像玩家真的可能说出来的话.
8. 不要写成字段名拼接句.
9. 不要写成“召出X, 再把X打出去”的机械说明句.
10. `politeness` 在输入中只会是 `0` 或 `1`.
11. `politeness = 1` 时, 更书面, 更文学, 更像正式咒辞.
12. `politeness = 0` 时, 更口语, 更临场, 更像玩家直接喊出来的话.

输入会提供:
- `model_socket`
- `recipe_name`
- `recipe_guidance`
- `reference_examples`
- `recent_outputs`

请严格根据 `model_socket` 生成一句新的中文咒语.
