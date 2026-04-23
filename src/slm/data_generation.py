from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .io_utils import append_jsonl, read_jsonl
from .model_socket_schema import (
    MATERIAL_TEMPLATE_LABELS,
    MOTION_DIRECTION_LABELS,
    MOTION_TEMPLATE_LABELS,
    ORIGIN_LABELS,
    RELEASE_TEMPLATE_LABELS,
    REACTION_TEMPLATE_LABELS,
    TARGET_LABELS,
    normalize_model_socket,
    validate_model_socket,
)


ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "src" / "prompts"


@dataclass(frozen=True)
class StyleRecipe:
    recipe_id: str
    name: str
    politeness: int
    guidance: str
    avoid: str


@dataclass(frozen=True)
class ContentMotif:
    motif_id: str
    prompt_hint: str


STYLE_RECIPES = [
    StyleRecipe("semi_classical_balanced", "半文半白平衡", 1, "像认真施法的玩家, 略带古意, 但不过典.", "不要变成纯文言."),
    StyleRecipe("semi_classical_literary", "半文半白偏文学", 1, "半文半白, 有修辞, 但仍可理解.", "不要堆生造词."),
    StyleRecipe("ritual_prayer", "仪式祈请", 1, "像誓辞或祈请, 仪式感强.", "不要太生活化."),
    StyleRecipe("literary_soft", "文学描写偏柔", 1, "文学性高, 有画面, 更柔和.", "不要写成旁白."),
    StyleRecipe("literary_sharp", "文学描写偏锋利", 1, "文学性高, 但更有推进感.", "不要拖太长."),
    StyleRecipe("detail_literary", "细节描写文学", 1, "大量细节和修辞, 颜色, 质地, 动势都可以写.", "不要写成长说明文."),
    StyleRecipe("everyday_clean", "生活化克制", 0, "像玩家自然说出来的话, 清楚, 完整.", "不要彻底失去施法感."),
    StyleRecipe("battle_colloquial", "口语战斗", 0, "像战斗里急促喊出来的话.", "不要写成残句."),
    StyleRecipe("colloquial_with_lift", "口语里带一点文气", 0, "主体口语, 但更凝练.", "不要突然切回古风."),
    StyleRecipe("modern_reference_light", "轻度现实参照", 0, "允许熟悉现实物象进入句子.", "不要变成科技说明."),
    StyleRecipe("modern_reference_heavy", "强现实参照", 0, "明显借用现实世界物象.", "不要写成长解释句."),
    StyleRecipe("hacky_clever", "人类钻空子", 0, "像玩家在试探系统边界, 用化学物质, 漏洞思维或极端规则描述.", "不要写成参数串."),
    StyleRecipe("classical_direct", "古意但直接", 1, "有古意, 但说法直接.", "不要写得太浮艳."),
    StyleRecipe("high_ritual_high_literary", "高仪式高文学", 1, "非常正式, 庄重, 文学, 仪式化.", "不要变成纯谜语."),
]


CONTENT_MOTIFS = [
    ContentMotif("holy_fire", "明亮圣火, 炽焰, 火潮"),
    ContentMotif("wild_fire", "野火, 烈焰, 火浪"),
    ContentMotif("acid_liquid", "酸液, 腐蚀液, 刺鼻酸潮"),
    ContentMotif("poison_smoke", "毒烟, 毒雾, 瘴气"),
    ContentMotif("frost_breath", "寒气, 霜息, 冷白雾"),
    ContentMotif("ice_shards", "冰棱, 冰刺, 寒晶"),
    ContentMotif("stone_spear", "石枪, 岩刺, 地上突起的石锋"),
    ContentMotif("iron_sand", "铁砂, 金属砂流, 磨蚀性颗粒"),
    ContentMotif("lightning_arc", "电弧, 雷光, 白亮电芒"),
    ContentMotif("sticky_tar", "黑色黏液, 沥青, 焦黑黏流"),
    ContentMotif("radiant_dust", "发亮粉尘, 微光颗粒, 星屑般的亮尘"),
    ContentMotif("lava_pool", "熔流, 岩浆, 滚烫灼液"),
    ContentMotif("thorn_vines", "荆棘, 藤蔓, 刺藤"),
    ContentMotif("liquid_metal", "液态金属, 水银般银流, 明亮银液"),
    ContentMotif("ash_cloud", "灰雾, 烬云, 热灰"),
    ContentMotif("honey_glue", "粘稠蜂蜜, 胶状甜液, 拉丝黏流"),
    ContentMotif("glass_rain", "玻璃碎片, 晶屑, 透明锋片"),
    ContentMotif("salt_crystal", "盐粒, 盐霜, 白色结晶"),
    ContentMotif("paper_flood", "试卷, 纸页, 成堆纸张"),
    ContentMotif("laser_like_light", "灼热白光, 激光般光束, 切割白线"),
]


MODEL_SOCKET_BLUEPRINTS = {
    "holy_fire": {"material_template": "fire", "reaction_template": "burn", "release_template": "spray", "motion_template": "flow", "motion_direction": "forward", "origin": "self", "target": "enemy"},
    "wild_fire": {"material_template": "fire", "reaction_template": "burn", "release_template": "spray", "motion_template": "flow", "motion_direction": "forward", "origin": "front_up", "target": "enemy"},
    "acid_liquid": {"material_template": "acid", "reaction_template": "corrode", "release_template": "spray", "motion_template": "flow", "motion_direction": "forward", "origin": "self", "target": "enemy"},
    "poison_smoke": {"material_template": "poison_slurry", "reaction_template": "poison", "release_template": "spray", "motion_template": "vortex", "motion_direction": "target", "origin": "front_down", "target": "enemy"},
    "frost_breath": {"material_template": "steam", "reaction_template": "freeze", "release_template": "spray", "motion_template": "flow", "motion_direction": "forward", "origin": "self", "target": "enemy"},
    "ice_shards": {"material_template": "ice", "reaction_template": "freeze", "release_template": "appear", "motion_template": "fixed", "motion_direction": "target", "origin": "front_up", "target": "enemy"},
    "stone_spear": {"material_template": "granite", "reaction_template": "none", "release_template": "appear", "motion_template": "fixed", "motion_direction": "forward", "origin": "front_down", "target": "enemy"},
    "iron_sand": {"material_template": "iron", "reaction_template": "none", "release_template": "spray", "motion_template": "vortex", "motion_direction": "target", "origin": "self", "target": "enemy"},
    "lightning_arc": {"material_template": "lightning", "reaction_template": "burn", "release_template": "appear", "motion_template": "fixed", "motion_direction": "target", "origin": "self", "target": "enemy"},
    "sticky_tar": {"material_template": "tar", "reaction_template": "none", "release_template": "appear", "motion_template": "flow", "motion_direction": "forward", "origin": "front_down", "target": "enemy"},
    "radiant_dust": {"material_template": "light", "reaction_template": "grow", "release_template": "spray", "motion_template": "flow", "motion_direction": "forward", "origin": "self", "target": "none"},
    "lava_pool": {"material_template": "explosive_slurry", "reaction_template": "burn", "release_template": "appear", "motion_template": "flow", "motion_direction": "forward", "origin": "front_down", "target": "none"},
    "thorn_vines": {"material_template": "wood", "reaction_template": "grow", "release_template": "appear", "motion_template": "rotation", "motion_direction": "target", "origin": "front_down", "target": "enemy"},
    "liquid_metal": {"material_template": "quicksilver", "reaction_template": "none", "release_template": "spray", "motion_template": "flow", "motion_direction": "target", "origin": "self", "target": "enemy"},
    "ash_cloud": {"material_template": "poison_slurry", "reaction_template": "poison", "release_template": "appear", "motion_template": "vortex", "motion_direction": "target", "origin": "back", "target": "enemy"},
    "honey_glue": {"material_template": "tar", "reaction_template": "none", "release_template": "appear", "motion_template": "flow", "motion_direction": "forward", "origin": "front_down", "target": "enemy"},
    "glass_rain": {"material_template": "glass", "reaction_template": "none", "release_template": "spray", "motion_template": "flow", "motion_direction": "down", "origin": "front_up", "target": "enemy"},
    "salt_crystal": {"material_template": "glass", "reaction_template": "freeze", "release_template": "spray", "motion_template": "flow", "motion_direction": "forward", "origin": "self", "target": "none"},
    "paper_flood": {"material_template": "unknown", "reaction_template": "none", "release_template": "appear", "motion_template": "flow", "motion_direction": "forward", "origin": "self", "target": "none"},
    "laser_like_light": {"material_template": "light", "reaction_template": "burn", "release_template": "appear", "motion_template": "fixed", "motion_direction": "forward", "origin": "self", "target": "enemy"},
}


def load_api_credentials(path: str | Path = ROOT / "api.txt") -> tuple[str, str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    base_url, api_key = raw.split(" ", 1)
    return base_url.rstrip("/"), api_key.strip()


def load_prompt_template(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> Any:
    text = _strip_think_blocks(text).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [idx for idx in (object_start, array_start) if idx != -1]
    if not starts:
        raise ValueError(f"cannot locate json object in response: {text[:200]}")
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(text[min(starts):])
    return parsed


def _normalized_text(text: str) -> str:
    text = str(text).replace("\u3000", " ")
    text = text.replace("，", ",").replace("。", ".").replace("：", ":").replace("；", ";").replace("！", "!").replace("？", "?")
    return " ".join(text.split()).strip()


def _parse_text_payload(parsed: Any) -> str:
    if not isinstance(parsed, dict) or "text" not in parsed:
        raise ValueError("response must be an object with text")
    text = _normalized_text(parsed["text"])
    if not text:
        raise ValueError("generated text is empty")
    if "\n" in text:
        raise ValueError("generated text must be a single line")
    if len(text) < 4 or len(text) > 100:
        raise ValueError(f"generated text length out of range: {len(text)}")
    banned_fragments = ["输出:", "解释:", "JSON", "```", "<think>", "这句咒语"]
    if any(fragment in text for fragment in banned_fragments):
        raise ValueError(f"generated text contains banned fragment: {text}")
    return text


def _parse_text_completion(raw: str) -> str:
    try:
        return _parse_text_payload(_extract_json(raw))
    except Exception:
        cleaned = _strip_think_blocks(raw)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            raise
        candidate = lines[-1].removeprefix("-").strip()
        if candidate.startswith("{") or "\"text\"" in candidate or ":" in candidate:
            raise
        return _parse_text_payload({"text": candidate})


def _build_client(timeout_sec: float = 300.0) -> httpx.Client:
    base_url, api_key = load_api_credentials()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout_sec)


def _chat_completion(
    client: httpx.Client,
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    response = client.post(
        "/chat/completions",
        json={
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _log(log_path: str | Path, payload: dict[str, Any]) -> None:
    append_jsonl(log_path, payload)


def _pick_examples(seed_rows: list[dict[str, Any]], *, recipe: StyleRecipe, motif: ContentMotif, rng: random.Random, limit: int = 4) -> list[dict[str, Any]]:
    same_motif = [row for row in seed_rows if row.get("meta", {}).get("motif_id") == motif.motif_id]
    same_recipe = [row for row in seed_rows if row.get("meta", {}).get("recipe_id") == recipe.recipe_id]
    pool = same_motif + same_recipe + seed_rows
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in pool:
        if row["id"] not in seen:
            seen.add(row["id"])
            deduped.append(row)
    rng.shuffle(deduped)
    return deduped[:limit]


def _build_spell_generation_user_prompt(
    *,
    template: str,
    recipe: StyleRecipe,
    motif: ContentMotif,
    examples: list[dict[str, Any]],
    recent_outputs: list[str],
) -> str:
    example_block = "\n".join(f'- {row["text"]}' for row in examples) if examples else "- 无"
    recent_block = "\n".join(f'- {text}' for text in recent_outputs) if recent_outputs else "- 无"
    model_socket = _model_socket_from_blueprint(motif.motif_id, recipe.politeness)
    return (
        f"{template}\n\n"
        f"recipe_name:\n{recipe.name}\n\n"
        f"politeness_target:\n{json.dumps(recipe.politeness, ensure_ascii=False)}\n\n"
        f"recipe_guidance:\n{recipe.guidance}\n\n"
        f"content_motif:\n{motif.prompt_hint}\n\n"
        f"avoid_patterns:\n{recipe.avoid}\n\n"
        f"model_socket:\n{json.dumps(model_socket, ensure_ascii=False, indent=2)}\n\n"
        f"reference_examples:\n{example_block}\n\n"
        f"recent_outputs:\n{recent_block}\n"
    )


def _build_spell_to_json_user_prompt(*, template: str, text: str) -> str:
    return f"{template}\n\n待解析咒语:\n{text}\n"


def _build_json_to_spell_user_prompt(*, template: str, model_socket: dict[str, Any], recipe: StyleRecipe) -> str:
    return (
        f"{template}\n\n"
        f"politeness_target:\n{json.dumps(recipe.politeness, ensure_ascii=False)}\n\n"
        f"model_socket:\n{json.dumps(model_socket, ensure_ascii=False, indent=2)}\n"
    )


def _model_socket_from_blueprint(blueprint_id: str, politeness: int) -> dict[str, Any]:
    blueprint = MODEL_SOCKET_BLUEPRINTS[blueprint_id]
    model_socket = {
        "subject_kind": "summon_material",
        "subject": {"material_template": blueprint["material_template"]},
        "reaction": {"reaction_template": blueprint["reaction_template"]},
        "release": {"release_template": blueprint["release_template"]},
        "motion": {
            "motion_template": blueprint["motion_template"],
            "motion_direction": blueprint["motion_direction"],
            "origin": blueprint["origin"],
            "target": blueprint["target"],
        },
        "expression": {"politeness": politeness},
    }
    validate_model_socket(model_socket)
    return model_socket


def _fallback_model_socket_from_source(source: dict[str, Any]) -> dict[str, Any]:
    meta = source.get("meta", {})
    motif_id = meta.get("motif_id")
    politeness = meta.get("politeness_target")
    if motif_id is None or politeness is None:
        raise ValueError("source row does not have motif_id/politeness_target for fallback")
    if motif_id not in MODEL_SOCKET_BLUEPRINTS:
        if motif_id == "wild_fire":
            motif_id = "holy_fire"
        else:
            raise KeyError(motif_id)
    return _model_socket_from_blueprint(motif_id, int(politeness))


def _fallback_spell_from_model_socket(model_socket: dict[str, Any]) -> str:
    material = model_socket["subject"]["material_template"]
    reaction = model_socket["reaction"]["reaction_template"]
    release = model_socket["release"]["release_template"]
    motion = model_socket["motion"]
    politeness = int(model_socket["expression"]["politeness"])

    noun_map = {
        "fire": "火潮",
        "acid": "酸液",
        "poison_slurry": "毒雾",
        "tar": "黑沥",
        "quicksilver": "银流",
        "granite": "石锋",
        "obsidian": "黑岩",
        "earth": "土潮",
        "sand": "砂流",
        "water": "水流",
        "ice": "冰棱",
        "steam": "寒气",
        "light": "白光",
        "wind": "风压",
        "lightning": "雷弧",
        "explosive_slurry": "灼液",
        "grass": "草蔓",
        "wood": "荆藤",
        "glass": "晶锋",
        "iron": "铁砂",
    }
    noun = noun_map.get(material, "异质")

    release_phrase = {
        "spray": "散出一片",
        "appear": "骤然现身",
    }[release]
    direction_phrase = {
        "forward": "向前而行",
        "backward": "反卷身后",
        "up": "向上而起",
        "down": "向下而落",
        "target": "直取前敌",
        "self": "环归我身",
        "front_up": "自前上方压下",
        "front_down": "自前下方翻起",
    }[motion["motion_direction"]]

    if politeness >= 1:
        return f"愿{noun}{release_phrase}, {direction_phrase}."
    return f"给我来点{noun}, {direction_phrase}."


def write_model_socket_seed_samples(*, output_path: str | Path, target_count: int, rng_seed: int = 23) -> list[dict[str, Any]]:
    rows = read_jsonl(output_path) if Path(output_path).exists() else []
    rng = random.Random(rng_seed)
    motif_ids = list(MODEL_SOCKET_BLUEPRINTS)
    while len(rows) < target_count:
        recipe = rng.choice(STYLE_RECIPES)
        motif_id = rng.choice(motif_ids)
        row = {
            "id": _stable_id("model_socket_seed", motif_id + f"::{recipe.recipe_id}::{len(rows)}"),
            "model_socket": _model_socket_from_blueprint(motif_id, recipe.politeness),
            "meta": {
                "source": "model_socket_seed",
                "recipe_id": recipe.recipe_id,
                "recipe_name": recipe.name,
                "politeness_target": recipe.politeness,
                "motif_id": motif_id,
            },
        }
        append_jsonl(output_path, row)
        rows.append(row)
    return rows


def _write_success_row(output_path: str | Path, rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    append_jsonl(output_path, row)
    rows.append(row)


def generate_spells(
    *,
    seed_examples_path: str | Path,
    output_path: str | Path,
    log_path: str | Path,
    target_count: int,
    model_name: str = "minimax-m2.5",
    max_retries: int = 3,
    rng_seed: int = 23,
) -> list[dict[str, Any]]:
    rows = read_jsonl(output_path) if Path(output_path).exists() else []
    existing_texts = {_normalized_text(row["text"]) for row in rows}
    seed_rows = read_jsonl(seed_examples_path)
    template = load_prompt_template("spell_generation_prompt.md")
    system_prompt = "你是中文奇幻咒语语料生成器. 只返回合法 JSON. 回答的第一个字符必须是 {. 不要解释, 不要分析, 不要代码块, 不要 <think>."
    rng = random.Random(rng_seed)
    client = _build_client()
    try:
        while len(rows) < target_count:
            recipe = rng.choice(STYLE_RECIPES)
            motif = rng.choice(CONTENT_MOTIFS)
            examples = _pick_examples(seed_rows, recipe=recipe, motif=motif, rng=rng)
            recent_outputs = [row["text"] for row in rows[-8:]]
            user_prompt = _build_spell_generation_user_prompt(template=template, recipe=recipe, motif=motif, examples=examples, recent_outputs=recent_outputs)
            last_error: str | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    raw = _chat_completion(client, model_name=model_name, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.9, max_tokens=1800)
                    text = _parse_text_completion(raw)
                    if text in existing_texts:
                        raise ValueError(f"duplicate text: {text}")
                    row = {
                        "id": _stable_id("spell", text),
                        "text": text,
                        "meta": {
                            "source": "spell_generation",
                            "recipe_id": recipe.recipe_id,
                            "recipe_name": recipe.name,
                            "politeness_target": recipe.politeness,
                            "motif_id": motif.motif_id,
                            "motif_hint": motif.prompt_hint,
                            "model_name": model_name,
                        },
                    }
                    _write_success_row(output_path, rows, row)
                    existing_texts.add(text)
                    _log(log_path, {"event": "spell_generation_success", "row_id": row["id"], "recipe_id": recipe.recipe_id, "motif_id": motif.motif_id, "attempt": attempt, "response_preview": raw[:500]})
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)
                    _log(log_path, {"event": "spell_generation_retry", "recipe_id": recipe.recipe_id, "motif_id": motif.motif_id, "attempt": attempt, "error": last_error})
                    time.sleep(min(10, attempt * 2))
            else:
                _log(log_path, {"event": "spell_generation_failed", "recipe_id": recipe.recipe_id, "motif_id": motif.motif_id, "error": last_error})
    finally:
        client.close()
    return rows


def translate_spells_to_json(
    *,
    input_path: str | Path,
    output_path: str | Path,
    log_path: str | Path,
    model_name: str = "minimax-m2.5",
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    source_rows = read_jsonl(input_path)
    rows = read_jsonl(output_path) if Path(output_path).exists() else []
    translated_source_ids = {row["meta"]["source_spell_id"] for row in rows}
    template = load_prompt_template("spell_to_json_prompt.md")
    system_prompt = "你是中文奇幻咒语结构化标注器. 只返回合法 JSON. 回答的第一个字符必须是 {. 不要解释, 不要分析, 不要代码块, 不要 <think>."
    client = _build_client()
    try:
        for source in source_rows:
            if source["id"] in translated_source_ids:
                continue
            user_prompt = _build_spell_to_json_user_prompt(template=template, text=source["text"])
            last_error: str | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    raw = _chat_completion(client, model_name=model_name, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.1, max_tokens=1200)
                    model_socket = normalize_model_socket(_extract_json(raw))
                    validate_model_socket(model_socket)
                    row = {
                        "id": _stable_id("spell_to_model_socket", source["id"] + "::" + source["text"]),
                        "text": source["text"],
                        "model_socket": model_socket,
                        "meta": {
                            "source": "spell_to_model_socket",
                            "source_spell_id": source["id"],
                            "source_meta": source.get("meta", {}),
                            "model_name": model_name,
                        },
                    }
                    _write_success_row(output_path, rows, row)
                    translated_source_ids.add(source["id"])
                    _log(log_path, {"event": "spell_to_json_success", "source_spell_id": source["id"], "attempt": attempt, "response_preview": raw[:500]})
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)
                    _log(log_path, {"event": "spell_to_json_retry", "source_spell_id": source["id"], "attempt": attempt, "error": last_error})
                    time.sleep(min(10, attempt * 2))
            else:
                try:
                    model_socket = _fallback_model_socket_from_source(source)
                    row = {
                        "id": _stable_id("spell_to_model_socket", source["id"] + "::" + source["text"]),
                        "text": source["text"],
                        "model_socket": model_socket,
                        "meta": {
                            "source": "spell_to_model_socket_fallback",
                            "source_spell_id": source["id"],
                            "source_meta": source.get("meta", {}),
                            "model_name": model_name,
                        },
                    }
                    _write_success_row(output_path, rows, row)
                    translated_source_ids.add(source["id"])
                    _log(log_path, {"event": "spell_to_json_fallback_success", "source_spell_id": source["id"], "error": last_error})
                except Exception as fallback_exc:  # noqa: BLE001
                    _log(log_path, {"event": "spell_to_json_failed", "source_spell_id": source["id"], "error": last_error, "fallback_error": repr(fallback_exc)})
    finally:
        client.close()
    return rows


def json_rows_to_spells(
    *,
    input_path: str | Path,
    output_path: str | Path,
    log_path: str | Path,
    model_name: str = "minimax-m2.5",
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    source_rows = read_jsonl(input_path)
    rows = read_jsonl(output_path) if Path(output_path).exists() else []
    translated_source_ids = {row["meta"]["source_model_socket_id"] for row in rows}
    existing_texts = {_normalized_text(row["text"]) for row in rows}
    template = load_prompt_template("json_to_spell_prompt.md")
    system_prompt = "你是中文奇幻咒语反推生成器. 只返回合法 JSON. 回答的第一个字符必须是 {. 不要解释, 不要分析, 不要代码块, 不要 <think>."
    client = _build_client()
    try:
        for source in source_rows:
            if source["id"] in translated_source_ids:
                continue
            model_socket = source["model_socket"]
            validate_model_socket(model_socket)
            recipe_politeness = source.get("meta", {}).get("politeness_target")
            recipe = StyleRecipe(
                source.get("meta", {}).get("recipe_id", "external"),
                source.get("meta", {}).get("recipe_name", "external"),
                int(recipe_politeness if recipe_politeness is not None else model_socket["expression"]["politeness"]),
                "按给定 politeness 生成.",
                "不要重复原句.",
            )
            user_prompt = _build_json_to_spell_user_prompt(template=template, model_socket=model_socket, recipe=recipe)
            last_error: str | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    raw = _chat_completion(client, model_name=model_name, system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.8, max_tokens=1800)
                    text = _parse_text_completion(raw)
                    if text in existing_texts:
                        raise ValueError(f"duplicate text: {text}")
                    row = {
                        "id": _stable_id("json_to_spell", source["id"] + "::" + text),
                        "text": text,
                        "model_socket": model_socket,
                        "meta": {
                            "source": "json_to_spell",
                            "source_model_socket_id": source["id"],
                            "source_meta": source.get("meta", {}),
                            "model_name": model_name,
                        },
                    }
                    _write_success_row(output_path, rows, row)
                    translated_source_ids.add(source["id"])
                    existing_texts.add(text)
                    _log(log_path, {"event": "json_rows_to_spell_success", "source_model_socket_id": source["id"], "attempt": attempt, "response_preview": raw[:500]})
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = repr(exc)
                    _log(log_path, {"event": "json_rows_to_spell_retry", "source_model_socket_id": source["id"], "attempt": attempt, "error": last_error})
                    time.sleep(min(10, attempt * 2))
            else:
                try:
                    text = _normalized_text(_fallback_spell_from_model_socket(model_socket))
                    if text not in existing_texts:
                        row = {
                            "id": _stable_id("json_to_spell", source["id"] + "::" + text),
                            "text": text,
                            "model_socket": model_socket,
                            "meta": {
                                "source": "json_to_spell_fallback",
                                "source_model_socket_id": source["id"],
                                "source_meta": source.get("meta", {}),
                                "model_name": model_name,
                            },
                        }
                        _write_success_row(output_path, rows, row)
                        translated_source_ids.add(source["id"])
                        existing_texts.add(text)
                        _log(log_path, {"event": "json_rows_to_spell_fallback_success", "source_model_socket_id": source["id"], "error": last_error})
                except Exception as fallback_exc:  # noqa: BLE001
                    _log(log_path, {"event": "json_rows_to_spell_failed", "source_model_socket_id": source["id"], "error": last_error, "fallback_error": repr(fallback_exc)})
    finally:
        client.close()
    return rows


def generate_json_to_spell_dataset(
    *,
    output_path: str | Path,
    log_path: str | Path,
    target_count: int,
    model_name: str = "minimax-m2.5",
    max_retries: int = 3,
    rng_seed: int = 23,
) -> list[dict[str, Any]]:
    seed_path = Path(output_path).with_name(Path(output_path).stem + "_seeds.jsonl")
    write_model_socket_seed_samples(output_path=seed_path, target_count=target_count, rng_seed=rng_seed)
    return json_rows_to_spells(input_path=seed_path, output_path=output_path, log_path=log_path, model_name=model_name, max_retries=max_retries)


def build_random_spell_to_json_dataset(
    *,
    seed_examples_path: str | Path,
    generated_spell_output_path: str | Path,
    generated_spell_log_path: str | Path,
    translated_output_path: str | Path,
    translated_log_path: str | Path,
    target_count: int,
    spell_model_name: str = "minimax-m2.5",
    translation_model_name: str = "minimax-m2.5",
    max_retries: int = 3,
    rng_seed: int = 23,
) -> list[dict[str, Any]]:
    generate_spells(
        seed_examples_path=seed_examples_path,
        output_path=generated_spell_output_path,
        log_path=generated_spell_log_path,
        target_count=target_count,
        model_name=spell_model_name,
        max_retries=max_retries,
        rng_seed=rng_seed,
    )
    return translate_spells_to_json(
        input_path=generated_spell_output_path,
        output_path=translated_output_path,
        log_path=translated_log_path,
        model_name=translation_model_name,
        max_retries=max_retries,
    )
