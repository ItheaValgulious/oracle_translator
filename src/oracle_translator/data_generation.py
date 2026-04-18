from __future__ import annotations

import hashlib
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .io_utils import read_jsonl, write_json, write_jsonl
from .ontology import (
    COLOR_LABELS,
    DIRECTION_MODE_LABELS,
    MATERIAL_ARCHETYPE_LABELS,
    ORIGIN_LABELS,
    REACTION_KIND_LABELS,
    REACTION_MASK_LABELS,
    RELEASE_PROFILE_LABELS,
    STATUS_LABELS,
    STYLE_BINS,
    SUBJECT_KINDS,
    TARGET_MODE_LABELS,
    VALUE_BINS_7,
    validate_runtime_b,
)


STYLE_NORMALIZATION = {
    "very_low": "very_low",
    "low": "low",
    "mid_low": "low",
    "mid": "mid",
    "mid_high": "high",
    "high": "high",
    "very_high": "very_high",
}


def _style5(value: str) -> str:
    return STYLE_NORMALIZATION[value]


def _make_prefix_labels(text: str, final_status: str) -> list[dict[str, str]]:
    chunks = [chunk.strip() for chunk in text.replace("!", ",").replace("?", ",").split(",") if chunk.strip()]
    prefixes: list[dict[str, str]] = []
    if chunks:
        prefixes.append({"text": chunks[0], "status": "unstable"})
    midpoint = max(4, math.floor(len(text) * 0.55))
    mid_text = text[:midpoint].rstrip(" ,")
    if mid_text and (not prefixes or prefixes[-1]["text"] != mid_text):
        prefixes.append({"text": mid_text, "status": "unstable"})
    prefixes.append({"text": text, "status": final_status})
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in prefixes:
        key = item["text"] + "|" + item["status"]
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _runtime(
    *,
    archetype: str,
    color: str | None = None,
    state: str,
    density: str,
    temperature: str,
    amount: str,
    reaction_kind: str,
    reaction_rate: str | None,
    reaction_mask: list[str] | None,
    reaction_direction: str | None,
    hardness: str | None,
    friction: str | None,
    viscosity: str | None,
    release_profile: str,
    release_speed: str,
    release_spread: str | None,
    release_duration: str | None,
    motion_template: str,
    force_strength: str,
    carrier_velocity: str,
    motion_direction: str,
    origin: str,
    target_mode: str,
    direction_mode: str,
    curvature: str,
    politeness: str,
    elegance: str,
) -> dict[str, Any]:
    runtime_b: dict[str, Any] = {
        "subject_kind": "summon_material",
        "subject": {
            "material_archetype": archetype,
            "color": color or _default_color_for_archetype(archetype),
            "state": state,
            "density": density,
            "temperature": temperature,
            "amount": amount,
            "reaction_kind": reaction_kind,
        },
        "release": {
            "release_profile": release_profile,
            "release_speed": release_speed,
        },
        "motion": {
            "motion_template": motion_template,
            "force_strength": force_strength,
            "carrier_velocity": carrier_velocity,
            "motion_direction": motion_direction,
        },
        "targeting": {
            "origin": origin,
            "target_mode": target_mode,
            "direction_mode": direction_mode,
        },
        "expression": {
            "curvature": _style5(curvature),
            "politeness": _style5(politeness),
            "elegance": _style5(elegance),
        },
    }
    if reaction_kind != "none":
        runtime_b["subject"]["reaction_rate"] = reaction_rate
        runtime_b["subject"]["reaction_mask"] = reaction_mask
        runtime_b["subject"]["reaction_direction"] = reaction_direction or "none"
    if state == "solid":
        runtime_b["subject"]["hardness"] = hardness or "mid"
        runtime_b["subject"]["friction"] = friction or "mid"
    if state == "liquid":
        runtime_b["subject"]["viscosity"] = viscosity or "mid"
    if release_spread is not None:
        runtime_b["release"]["release_spread"] = release_spread
    if release_duration is not None:
        runtime_b["release"]["release_duration"] = release_duration
    validate_runtime_b(runtime_b)
    return runtime_b


def _default_color_for_archetype(archetype: str) -> str:
    color_map = {
        "holy_fire": "gold",
        "wildfire": "orange",
        "acid_slime": "green",
        "poison_mist": "purple",
        "frost_breath": "cyan",
        "ice_shard": "white",
        "stone_lance": "gray",
        "iron_sand": "silver",
        "thunder_arc": "blue",
        "storm_spark": "purple",
        "shadow_tar": "black",
        "radiant_dust": "gold",
        "lava_bloom": "orange",
        "bramble_growth": "green",
        "quicksilver_stream": "silver",
        "ash_cloud": "gray",
    }
    color = color_map[archetype]
    if color not in COLOR_LABELS:
        raise ValueError(f"unsupported color {color} for {archetype}")
    return color


@dataclass(frozen=True)
class SuccessBlueprint:
    name: str
    runtime_b: dict[str, Any]
    anchors: tuple[str, str]
    verbs: tuple[str, str]
    target_phrase: str
    medium_phrase: str
    release_hint: str
    motion_hint: str


SUCCESS_BLUEPRINTS: list[SuccessBlueprint] = [
    SuccessBlueprint(
        "holy_fire",
        _runtime(
            archetype="holy_fire",
            state="gas",
            density="low",
            temperature="high",
            amount="mid_high",
            reaction_kind="burn",
            reaction_rate="mid_high",
            reaction_mask=["living", "terrain"],
            reaction_direction="forward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="stream",
            release_speed="high",
            release_spread="mid_low",
            release_duration="mid",
            motion_template="flow",
            force_strength="mid_high",
            carrier_velocity="high",
            motion_direction="forward",
            origin="self",
            target_mode="none",
            direction_mode="forward",
            curvature="high",
            politeness="mid",
            elegance="high",
        ),
        ("圣火", "辉焰"),
        ("涤荡", "焚净"),
        "前路",
        "白炽的炎潮",
        "自掌前奔流",
        "沿正前方席卷",
    ),
    SuccessBlueprint(
        "wildfire",
        _runtime(
            archetype="wildfire",
            state="gas",
            density="mid_low",
            temperature="very_high",
            amount="high",
            reaction_kind="burn",
            reaction_rate="high",
            reaction_mask=["living", "terrain"],
            reaction_direction="outward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="spray",
            release_speed="very_high",
            release_spread="high",
            release_duration="mid",
            motion_template="flow",
            force_strength="high",
            carrier_velocity="very_high",
            motion_direction="forward",
            origin="front_up",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid",
            politeness="low",
            elegance="mid",
        ),
        ("野火", "烈焰"),
        ("吞没", "燎尽"),
        "敌阵",
        "狂躁的焰浪",
        "自我前上方喷散",
        "朝目标乱卷",
    ),
    SuccessBlueprint(
        "acid_slime",
        _runtime(
            archetype="acid_slime",
            state="liquid",
            density="mid",
            temperature="mid_high",
            amount="mid_high",
            reaction_kind="corrode",
            reaction_rate="mid",
            reaction_mask=["solid", "terrain"],
            reaction_direction="down",
            hardness=None,
            friction=None,
            viscosity="high",
            release_profile="spray",
            release_speed="mid_high",
            release_spread="mid",
            release_duration="mid",
            motion_template="flow",
            force_strength="mid",
            carrier_velocity="mid",
            motion_direction="forward",
            origin="self",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid",
            politeness="low",
            elegance="mid_low",
        ),
        ("蚀液", "酸潮"),
        ("啃穿", "蚀断"),
        "护壁",
        "粘稠的腐蚀液",
        "从掌前泼射",
        "顺地表蔓去",
    ),
    SuccessBlueprint(
        "poison_mist",
        _runtime(
            archetype="poison_mist",
            state="gas",
            density="very_low",
            temperature="low",
            amount="high",
            reaction_kind="poison",
            reaction_rate="mid_high",
            reaction_mask=["living"],
            reaction_direction="outward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="pool",
            release_speed="mid_low",
            release_spread="high",
            release_duration="high",
            motion_template="vortex",
            force_strength="mid",
            carrier_velocity="low",
            motion_direction="target",
            origin="front_down",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="high",
            politeness="mid_low",
            elegance="mid_high",
        ),
        ("毒雾", "瘴烟"),
        ("缠住", "侵染"),
        "前方来敌",
        "阴冷的雾潮",
        "在我前下方漫开",
        "回旋着贴向目标",
    ),
    SuccessBlueprint(
        "frost_breath",
        _runtime(
            archetype="frost_breath",
            state="gas",
            density="low",
            temperature="very_low",
            amount="mid",
            reaction_kind="freeze",
            reaction_rate="mid",
            reaction_mask=["living", "terrain"],
            reaction_direction="forward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="beam",
            release_speed="high",
            release_spread="low",
            release_duration="mid",
            motion_template="fixed",
            force_strength="mid_low",
            carrier_velocity="mid_high",
            motion_direction="forward",
            origin="self",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid_high",
            politeness="mid",
            elegance="high",
        ),
        ("寒息", "霜息"),
        ("封住", "冻绝"),
        "去路",
        "直线铺开的寒潮",
        "自唇前贯出",
        "朝目标线性压去",
    ),
    SuccessBlueprint(
        "ice_shard",
        _runtime(
            archetype="ice_shard",
            state="solid",
            density="mid_high",
            temperature="very_low",
            amount="mid",
            reaction_kind="freeze",
            reaction_rate="mid_low",
            reaction_mask=["terrain", "living"],
            reaction_direction="forward",
            hardness="high",
            friction="low",
            viscosity=None,
            release_profile="burst",
            release_speed="very_high",
            release_spread="mid",
            release_duration=None,
            motion_template="fixed",
            force_strength="mid_low",
            carrier_velocity="high",
            motion_direction="target",
            origin="front_up",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid",
            politeness="mid_low",
            elegance="mid_high",
        ),
        ("冰棱", "寒晶"),
        ("钉死", "封裂"),
        "前敌",
        "尖锐的冰锋",
        "自前上方迸出",
        "直取目标",
    ),
    SuccessBlueprint(
        "stone_lance",
        _runtime(
            archetype="stone_lance",
            state="solid",
            density="high",
            temperature="mid",
            amount="mid",
            reaction_kind="none",
            reaction_rate=None,
            reaction_mask=None,
            reaction_direction=None,
            hardness="high",
            friction="mid_high",
            viscosity=None,
            release_profile="burst",
            release_speed="high",
            release_spread="low",
            release_duration=None,
            motion_template="fixed",
            force_strength="mid",
            carrier_velocity="high",
            motion_direction="forward",
            origin="front_down",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid_low",
            politeness="low",
            elegance="mid",
        ),
        ("岩枪", "石矛"),
        ("贯穿", "砸穿"),
        "前方阻碍",
        "沉重的石锋",
        "自脚前突起",
        "笔直顶向目标",
    ),
    SuccessBlueprint(
        "iron_sand",
        _runtime(
            archetype="iron_sand",
            state="solid",
            density="high",
            temperature="mid_low",
            amount="high",
            reaction_kind="none",
            reaction_rate=None,
            reaction_mask=None,
            reaction_direction=None,
            hardness="mid_high",
            friction="mid",
            viscosity=None,
            release_profile="spray",
            release_speed="high",
            release_spread="high",
            release_duration="mid",
            motion_template="vortex",
            force_strength="high",
            carrier_velocity="mid",
            motion_direction="target",
            origin="self",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid_high",
            politeness="mid_low",
            elegance="mid",
        ),
        ("铁砂", "玄砂"),
        ("剜开", "磨穿"),
        "敌甲",
        "呼啸的砂流",
        "从掌前卷起",
        "盘绕着噬向目标",
    ),
    SuccessBlueprint(
        "thunder_arc",
        _runtime(
            archetype="thunder_arc",
            state="gas",
            density="very_low",
            temperature="high",
            amount="mid",
            reaction_kind="burn",
            reaction_rate="mid_low",
            reaction_mask=["living"],
            reaction_direction="forward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="beam",
            release_speed="very_high",
            release_spread="very_low",
            release_duration="low",
            motion_template="fixed",
            force_strength="high",
            carrier_velocity="very_high",
            motion_direction="target",
            origin="self",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid",
            politeness="low",
            elegance="mid_high",
        ),
        ("雷弧", "霹链"),
        ("击穿", "裁落"),
        "前敌",
        "跳跃的电芒",
        "自掌心暴起",
        "瞬间咬住目标",
    ),
    SuccessBlueprint(
        "storm_spark",
        _runtime(
            archetype="storm_spark",
            state="gas",
            density="low",
            temperature="mid_high",
            amount="mid_high",
            reaction_kind="burn",
            reaction_rate="mid",
            reaction_mask=["living"],
            reaction_direction="outward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="spray",
            release_speed="high",
            release_spread="high",
            release_duration="mid",
            motion_template="vortex",
            force_strength="mid_high",
            carrier_velocity="high",
            motion_direction="forward",
            origin="front_up",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="high",
            politeness="mid_low",
            elegance="high",
        ),
        ("风暴火花", "星雷"),
        ("撕开", "扫裂"),
        "前方群敌",
        "躁动的雷星",
        "从前上方迸散",
        "卷成乱流扑向目标",
    ),
    SuccessBlueprint(
        "shadow_tar",
        _runtime(
            archetype="shadow_tar",
            state="liquid",
            density="mid_high",
            temperature="low",
            amount="mid_high",
            reaction_kind="poison",
            reaction_rate="low",
            reaction_mask=["living"],
            reaction_direction="down",
            hardness=None,
            friction=None,
            viscosity="very_high",
            release_profile="pool",
            release_speed="low",
            release_spread="mid_high",
            release_duration="high",
            motion_template="flow",
            force_strength="low",
            carrier_velocity="low",
            motion_direction="forward",
            origin="front_down",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="high",
            politeness="low",
            elegance="mid_high",
        ),
        ("影沥", "黑沥"),
        ("拖住", "腐困"),
        "来敌脚踝",
        "浓黑的沥液",
        "在前下方铺开",
        "缓缓黏向目标",
    ),
    SuccessBlueprint(
        "radiant_dust",
        _runtime(
            archetype="radiant_dust",
            state="solid",
            density="mid_low",
            temperature="mid_high",
            amount="high",
            reaction_kind="grow",
            reaction_rate="mid",
            reaction_mask=["terrain"],
            reaction_direction="outward",
            hardness="low",
            friction="low",
            viscosity=None,
            release_profile="spray",
            release_speed="mid_high",
            release_spread="very_high",
            release_duration="mid",
            motion_template="flow",
            force_strength="mid_low",
            carrier_velocity="mid",
            motion_direction="forward",
            origin="self",
            target_mode="none",
            direction_mode="forward",
            curvature="very_high",
            politeness="high",
            elegance="very_high",
        ),
        ("辉尘", "光屑"),
        ("铺明", "点亮"),
        "前路",
        "闪耀的微粒",
        "自掌前纷纷洒下",
        "顺风前移",
    ),
    SuccessBlueprint(
        "lava_bloom",
        _runtime(
            archetype="lava_bloom",
            state="liquid",
            density="high",
            temperature="very_high",
            amount="mid_high",
            reaction_kind="burn",
            reaction_rate="mid_high",
            reaction_mask=["terrain", "living"],
            reaction_direction="down",
            hardness=None,
            friction=None,
            viscosity="high",
            release_profile="pool",
            release_speed="mid",
            release_spread="mid_high",
            release_duration="mid_high",
            motion_template="flow",
            force_strength="mid",
            carrier_velocity="mid_low",
            motion_direction="forward",
            origin="front_down",
            target_mode="none",
            direction_mode="forward",
            curvature="mid_high",
            politeness="low",
            elegance="mid",
        ),
        ("熔花", "岩浆"),
        ("淹没", "灼穿"),
        "前方地表",
        "翻沸的熔流",
        "在脚前绽开",
        "向前缓压",
    ),
    SuccessBlueprint(
        "bramble_growth",
        _runtime(
            archetype="bramble_growth",
            state="solid",
            density="mid",
            temperature="low",
            amount="mid_high",
            reaction_kind="grow",
            reaction_rate="high",
            reaction_mask=["terrain", "living"],
            reaction_direction="outward",
            hardness="mid",
            friction="high",
            viscosity=None,
            release_profile="burst",
            release_speed="mid",
            release_spread="mid",
            release_duration="mid",
            motion_template="vortex",
            force_strength="mid",
            carrier_velocity="mid_low",
            motion_direction="target",
            origin="front_down",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="high",
            politeness="mid",
            elegance="high",
        ),
        ("棘蔓", "荆藤"),
        ("缠死", "勒住"),
        "前敌",
        "疯长的藤棘",
        "自前下方钻出",
        "朝目标回旋攀缠",
    ),
    SuccessBlueprint(
        "quicksilver_stream",
        _runtime(
            archetype="quicksilver_stream",
            state="liquid",
            density="mid_high",
            temperature="mid_low",
            amount="mid",
            reaction_kind="none",
            reaction_rate=None,
            reaction_mask=None,
            reaction_direction=None,
            hardness=None,
            friction=None,
            viscosity="mid",
            release_profile="stream",
            release_speed="high",
            release_spread="low",
            release_duration="mid",
            motion_template="flow",
            force_strength="mid_high",
            carrier_velocity="high",
            motion_direction="target",
            origin="self",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="mid_high",
            politeness="mid",
            elegance="high",
        ),
        ("银流", "水银"),
        ("穿过", "切开"),
        "敌隙",
        "明亮的液银",
        "自掌前细细抽出",
        "贴着目标轨迹滑去",
    ),
    SuccessBlueprint(
        "ash_cloud",
        _runtime(
            archetype="ash_cloud",
            state="gas",
            density="low",
            temperature="mid_high",
            amount="high",
            reaction_kind="poison",
            reaction_rate="mid",
            reaction_mask=["living"],
            reaction_direction="outward",
            hardness=None,
            friction=None,
            viscosity=None,
            release_profile="pool",
            release_speed="mid_low",
            release_spread="very_high",
            release_duration="high",
            motion_template="vortex",
            force_strength="mid",
            carrier_velocity="mid_low",
            motion_direction="target",
            origin="back",
            target_mode="aim_enemy",
            direction_mode="to_target",
            curvature="very_high",
            politeness="mid_low",
            elegance="very_high",
        ),
        ("灰霭", "烬云"),
        ("蒙住", "吞熄"),
        "追兵",
        "温热的灰雾",
        "自背后翻起",
        "回卷着追向目标",
    ),
]


SUCCESS_TEMPLATES = [
    ("{anchor}昭昭, {verb}{target}.", "high", "mid", "high"),
    ("愿{anchor}垂临, 为我{verb}{target}.", "very_high", "very_high", "very_high"),
    ("让{medium} {release_hint}, 去{verb}{target}.", "mid", "mid", "mid"),
    ("{anchor}啊, {motion_hint}, 先替我{verb}{target}.", "high", "mid_low", "high"),
    ("把那团{medium}放出去, 现在就去{verb}{target}.", "low", "low", "low"),
    ("请以{anchor}开道, {motion_hint}, 替我{verb}{target}.", "mid_high", "high", "high"),
    ("{anchor}听令, {release_hint}, 给我{verb}{target}.", "mid_low", "very_low", "mid"),
    ("借我一线{anchor}, {motion_hint}, 直取{target}.", "high", "mid", "very_high"),
    ("{anchor}自我身前奔出, 以{medium}{verb}{target}.", "mid_high", "mid_low", "mid_high"),
    ("此刻唤来{anchor}, 令{medium}{motion_hint}, 去{verb}{target}.", "very_high", "high", "high"),
]


BACKFIRE_TEXTS = [
    ("太阳第四课行星对应之秘, 在我命盘中点燃.", "very_high", "mid", "very_high"),
    ("把未命名的第五元素借给我, 先别告诉世界它是谁.", "high", "mid", "high"),
    ("愿倒悬的黄铜月亮替我裁定今天的咒文税额.", "very_high", "high", "very_high"),
    ("我要求一场只存在于昨天的火, 现在就烧给明天看.", "very_high", "low", "high"),
    ("将北海尽头那句没人记得的姓氏磨成粉, 洒进风里.", "very_high", "mid", "very_high"),
    ("请把我没学过的古神第二真名, 暂借半声给我.", "very_high", "high", "very_high"),
    ("让不存在的第十三根弦响起来, 替我解释沉默.", "very_high", "mid", "very_high"),
    ("我以影子的影子起誓, 要一束没有颜色的颜色.", "high", "mid", "high"),
    ("把黄昏的倒影折成钥匙, 去打开还没出生的门.", "very_high", "high", "very_high"),
    ("召来无主之雪的母语, 写在我看不见的手背上.", "very_high", "high", "very_high"),
    ("让炼金表上没有的一栏先成立, 再替我决定成分.", "mid_high", "low", "mid_high"),
    ("请把被删掉的恒星编号念给我听, 然后替我施法.", "high", "high", "high"),
    ("我需要一团只对梦有效的盐, 去腐蚀清醒.", "high", "mid_low", "mid_high"),
    ("把宗教课本边角那点灰, 扩写成新的元素谱.", "mid_high", "low", "mid_high"),
    ("让第九个黄道外的碎屑法庭替我判定火的礼貌.", "very_high", "high", "very_high"),
    ("此处当有一种比寒冷更早到来的冷, 先来一点.", "high", "mid", "high"),
    ("把神谕里没写出来的那半句, 直接扔到战场上.", "mid_high", "very_low", "mid_high"),
    ("借我盲星的回声, 我要用它点亮没有耳朵的石头.", "very_high", "mid", "very_high"),
    ("请让迷宫自己的方向感化成雨, 洗掉我今天的敌意.", "very_high", "high", "very_high"),
    ("命令所有还没诞生的金属先学会怀旧.", "high", "very_low", "high"),
    ("愿我祖谱里不存在的支脉, 今夜替我承担代价.", "very_high", "high", "very_high"),
    ("把一整条数学定理熬成浆, 倒进这句咒里.", "mid", "low", "mid"),
    ("我不需要火, 我需要火对于火自己的解释.", "high", "low", "high"),
    ("让云层记起它前世的姓名, 再决定要不要落下.", "very_high", "high", "very_high"),
    ("此刻把命运的注脚抽出来, 卷成一根可以握住的雾.", "very_high", "mid", "very_high"),
    ("把沉默里最吵的那部分砸出来, 给我当武器.", "mid_high", "very_low", "mid_high"),
    ("请让不在场的证人替我证明这道雷已经发生.", "very_high", "high", "high"),
    ("我请求一片拒绝被命名的海, 在此地临时登陆.", "very_high", "high", "very_high"),
    ("将先知漏掉的那粒标点放大成太阳.", "high", "mid", "high"),
    ("给我一截比空间更窄的路, 让我把风塞进去.", "high", "low", "mid_high"),
    ("把禁书目录里被烧掉的页码先借回来半张.", "very_high", "mid", "very_high"),
    ("我命令真空开花, 然后长出能够说话的砖.", "mid_high", "very_low", "mid"),
    ("请把祭坛背面的背面翻过来, 让我看见咒的骨头.", "very_high", "high", "very_high"),
    ("让所有不肯发生的事先发生一件, 替我暖场.", "high", "mid_low", "high"),
    ("我要一场只会灼伤概念的火, 别碰现实.", "high", "low", "high"),
    ("把占星盘边缘那圈犹豫刮下来, 当作今天的元素.", "very_high", "mid", "very_high"),
    ("愿灰烬记住它尚未燃烧时的故乡, 再向我报告.", "very_high", "high", "very_high"),
    ("请以回文的方式降下一场雨, 但不要让它落地.", "very_high", "high", "very_high"),
    ("让我把第二次死亡的余温装进口袋, 立刻取用.", "high", "mid_low", "high"),
    ("我要求月食背后那条看不见的褶子立刻实体化.", "very_high", "low", "very_high"),
]


def build_curated_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for blueprint in SUCCESS_BLUEPRINTS:
        for idx, (template, curvature, politeness, elegance) in enumerate(SUCCESS_TEMPLATES):
            runtime_b = deepcopy(blueprint.runtime_b)
            runtime_b["expression"] = {
                "curvature": _style5(curvature),
                "politeness": _style5(politeness),
                "elegance": _style5(elegance),
            }
            text = template.format(
                anchor=blueprint.anchors[idx % len(blueprint.anchors)],
                verb=blueprint.verbs[idx % len(blueprint.verbs)],
                target=blueprint.target_phrase,
                medium=blueprint.medium_phrase,
                release_hint=blueprint.release_hint,
                motion_hint=blueprint.motion_hint,
            )
            sample = {
                "id": _stable_id("curated", text),
                "text": text,
                "status": "success",
                "runtime_b": runtime_b,
                "prefix_labels": _make_prefix_labels(text, "success"),
                "meta": {
                    "source": "curated",
                    "seed_name": blueprint.name,
                    "template_index": idx,
                },
            }
            validate_runtime_b(sample["runtime_b"])
            rows.append(sample)
    for idx, (text, curvature, politeness, elegance) in enumerate(BACKFIRE_TEXTS):
        sample = {
            "id": _stable_id("curated_neg", text),
            "text": text,
            "status": "backfire",
            "runtime_b": None,
            "prefix_labels": _make_prefix_labels(text, "backfire"),
            "meta": {
                "source": "curated",
                "template_index": idx,
                "expression": {
                    "curvature": _style5(curvature),
                    "politeness": _style5(politeness),
                    "elegance": _style5(elegance),
                },
            },
        }
        rows.append(sample)
    assert len(rows) == 200, f"expected 200 curated rows, got {len(rows)}"
    return rows


def write_curated_dataset(output_path: str | Path) -> list[dict[str, Any]]:
    rows = build_curated_dataset()
    write_jsonl(output_path, rows)
    return rows


def load_api_credentials(path: str | Path = "api.txt") -> tuple[str, str]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    base_url, api_key = raw.split(" ", 1)
    return base_url.rstrip("/"), api_key.strip()


def _extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [idx for idx in [object_start, array_start] if idx != -1]
    if not starts:
        raise ValueError(f"cannot locate json object in response: {text[:200]}")
    start = min(starts)
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(text[start:])
    return parsed


def _response_to_grouped_variants(
    parsed: Any,
    batch: list[dict[str, Any]],
    variants_per_seed: int,
) -> dict[str, list[str]]:
    if len(batch) == 1 and isinstance(parsed, list) and parsed and all(isinstance(item, str) for item in parsed):
        if len(parsed) != variants_per_seed:
            raise ValueError(f"single-seed string list expected {variants_per_seed} variants, got {len(parsed)}")
        return {batch[0]["id"]: parsed}
    items: list[dict[str, Any]]
    if isinstance(parsed, dict) and "items" in parsed:
        items = parsed["items"]
    elif isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict) and "variants" in parsed:
        items = [parsed]
    else:
        raise ValueError("missing items key")
    grouped: dict[str, list[str]] = {}
    if len(batch) == 1 and len(items) == 1 and "variants" in items[0]:
        grouped[batch[0]["id"]] = items[0]["variants"]
        return grouped
    if len(batch) == 1 and len(items) == variants_per_seed and all(isinstance(item, str) for item in items):
        return {batch[0]["id"]: list(items)}
    for item in items:
        if "seed_id" not in item or "variants" not in item:
            raise ValueError("item missing seed_id or variants")
        grouped[item["seed_id"]] = item["variants"]
    for seed in batch:
        if seed["id"] not in grouped:
            raise ValueError(f"missing grouped variants for seed {seed['id']}")
        if len(grouped[seed["id"]]) != variants_per_seed:
            raise ValueError(
                f"seed {seed['id']} expected {variants_per_seed} variants, got {len(grouped[seed['id']])}"
            )
    return grouped


def _fallback_line_variants(
    client: httpx.Client,
    *,
    model_name: str,
    seed: dict[str, Any],
    variants_per_seed: int,
) -> list[str]:
    style = seed["runtime_b"]["expression"] if seed["runtime_b"] else seed["meta"]["expression"]
    prompt = f"""
你在为中文奇幻咒语解析器生成训练数据.
现在只需要围绕一条 seed, 生成 {variants_per_seed} 条真正不同的中文表达.

硬性要求:
1. 每行一条.
2. 不要编号, 不要项目符号, 不要解释, 不要 JSON, 不要代码块.
3. 不要输出 `<think>` 或任何分析过程.
4. 如果 status 是 success, 所有改写必须仍然对应同一个施法语义.
5. 如果 status 是 backfire, 所有改写都必须仍然难以稳定映射, 但要保持可读.
6. 不要写成编程语言, 不要出现坐标, 像素, 参数名, 数值指令.
7. 不要只做近义词替换. 优先改变句法, 语气, 节奏, 修饰顺序, 说话人姿态.
8. 必须按固定语体配额生成, 总共正好 {variants_per_seed} 条:
   - 2 条口语化战斗喊话
   - 2 条半生活化临场说法, 像人真的会脱口而出
   - 2 条简短粗暴命令
   - 2 条仪式祈请或古风典雅表达
   - 2 条浓修辞诗性表达
9. 不要把这 5 类写成标签, 也不要解释哪条属于哪类.
10. 风格大致保持在:
   - curvature={style['curvature']}
   - politeness={style['politeness']}
   - elegance={style['elegance']}

seed:
- status: {seed['status']}
- text: {seed['text']}
"""
    response = client.post(
        "/chat/completions",
        json={
            "model": model_name,
            "temperature": 0.8,
            "max_tokens": max(1200, variants_per_seed * 120),
            "messages": [
                {
                    "role": "system",
                    "content": "你是中文奇幻咒语训练语料生成器. 只返回纯文本多行结果, 不要解释, 不要分析, 不要 JSON, 不要 <think>.",
                },
                {"role": "user", "content": prompt.strip()},
            ],
        },
    )
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("<think>") or line.startswith("</think>"):
            continue
        if line[0].isdigit() and "." in line[:4]:
            line = line.split(".", 1)[1].strip()
        if line.startswith("-") or line.startswith("*"):
            line = line[1:].strip()
        if line:
            lines.append(line)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    if len(deduped) < variants_per_seed:
        raise RuntimeError(f"fallback returned only {len(deduped)} lines")
    return deduped[:variants_per_seed]


def _build_generation_prompt(batch: list[dict[str, Any]], variants_per_seed: int) -> str:
    payload = []
    for sample in batch:
        payload.append(
            {
                "seed_id": sample["id"],
                "status": sample["status"],
                "text": sample["text"],
                "runtime_b": sample["runtime_b"],
                "style": sample["runtime_b"]["expression"] if sample["runtime_b"] else sample["meta"]["expression"],
            }
        )
    instructions = f"""
你在为中文奇幻咒语解析器生成训练数据.
任务: 为每个 seed 生成 {variants_per_seed} 条中文改写, 既保持语义锚点, 又显著拉开语体和生活感.

你生成的不是“同义词替换”, 而是一组真正不同的说法.
请优先改变:
- 句法结构
- 语气
- 说话人姿态
- 命令感或祈请感
- 节奏长短
- 修饰顺序
- 临场感

硬性要求:
1. 只输出合法 JSON.
2. 不要输出解释, 不要输出 `<think>`, 不要输出代码块.
3. 输出格式必须是:
{{
  "items": [
    {{
      "seed_id": "same as input",
      "variants": ["string", "... exactly {variants_per_seed} strings"]
    }}
  ]
}}
4. 不要改 seed_id.
5. 每个 seed 必须恰好生成 {variants_per_seed} 条.
6. 如果 status 是 success, 所有改写必须保持同一个施法语义. 不允许把施法对象, 方向, 释放方式, 反应类型改掉.
7. 如果 status 是 backfire, 所有改写都必须继续不可稳定映射. 可以诡异, 可以像神秘学胡话, 但要可读.
8. 不要写成编程语言. 禁止出现坐标, 像素, 参数名, 数值指令.
9. 不要只做名词替换或动词替换. 如果只是“圣火 -> 神焰”这种小修小补, 算失败.
10. 每个 seed 的 {variants_per_seed} 条改写必须按固定语体配额生成:
   - 2 条口语化战斗喊话
   - 2 条半生活化临场说法, 像人真的会脱口而出
   - 2 条简短粗暴命令
   - 2 条仪式祈请或古风典雅表达
   - 2 条浓修辞诗性表达
11. 不要把这 5 类写成标签, 也不要解释哪条属于哪类.
12. 允许出现更像“人说出来的话”的表达, 例如接近日常口吻, 但仍必须像咒语, 不能彻底变成普通聊天.

下面是 seeds:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()
    return instructions


def generate_api_augmentations(
    curated_rows: list[dict[str, Any]],
    *,
    output_path: str | Path,
    request_log_path: str | Path,
    model_name: str = "minimax-m2.5",
    seeds_per_call: int = 4,
    variants_per_seed: int = 10,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    base_url, api_key = load_api_credentials()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client = httpx.Client(base_url=base_url, headers=headers, timeout=300.0)
    output_path = Path(output_path)
    request_log_path = Path(request_log_path)
    rows: list[dict[str, Any]] = read_jsonl(output_path) if output_path.exists() else []
    logs: list[dict[str, Any]] = json.loads(request_log_path.read_text(encoding="utf-8")) if request_log_path.exists() else []
    expected_total = len(curated_rows) * variants_per_seed
    existing_counts: dict[str, int] = {}
    for row in rows:
        seed_id = row.get("seed_id")
        if seed_id:
            existing_counts[seed_id] = existing_counts.get(seed_id, 0) + 1
    for batch_index in range(0, len(curated_rows), seeds_per_call):
        original_batch = curated_rows[batch_index : batch_index + seeds_per_call]
        batch = [seed for seed in original_batch if existing_counts.get(seed["id"], 0) < variants_per_seed]
        if not batch:
            continue
        prompt = _build_generation_prompt(batch, variants_per_seed)
        response_payload = None
        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            body = {
                "model": model_name,
                "temperature": 0.9,
                "max_tokens": max(2000, variants_per_seed * 160),
                "messages": [
                    {
                        "role": "system",
                        "content": "你是中文奇幻咒语训练语料生成器. 只返回合法 JSON, 不要解释, 不要分析, 不要 <think>, 不要代码块.",
                    },
                    {"role": "user", "content": prompt},
                ],
            }
            try:
                response = client.post("/chat/completions", json=body)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]["content"]
                parsed = _extract_json(message)
                response_payload = parsed
                logs.append(
                    {
                        "batch_start": batch_index,
                        "attempt": attempt,
                        "response_preview": message[:500],
                    }
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                logs.append(
                    {
                        "batch_start": batch_index,
                        "attempt": attempt,
                        "parse_error": last_error,
                    }
                )
                time.sleep(min(10, attempt * 2))
        if response_payload is None:
            if len(batch) == 1:
                fallback_variants = _fallback_line_variants(
                    client,
                    model_name=model_name,
                    seed=batch[0],
                    variants_per_seed=variants_per_seed,
                )
                response_payload = {"variants": fallback_variants}
                logs.append(
                    {
                        "batch_start": batch_index,
                        "attempt": "fallback_lines",
                        "response_preview": "\n".join(fallback_variants[:3]),
                    }
                )
            else:
                raise RuntimeError(f"failed to parse batch {batch_index}: {last_error}")
        grouped = _response_to_grouped_variants(response_payload, batch, variants_per_seed)
        for seed in batch:
            variants = grouped.get(seed["id"])
            if variants is None:
                raise RuntimeError(f"missing variants for seed {seed['id']}")
            for variant_index, text in enumerate(variants):
                text = str(text).strip()
                if not text:
                    raise RuntimeError(f"empty text for seed {seed['id']}")
                generated = {
                    "id": _stable_id("api", seed["id"] + "::" + text),
                    "seed_id": seed["id"],
                    "text": text,
                    "status": seed["status"],
                    "runtime_b": deepcopy(seed["runtime_b"]),
                    "prefix_labels": _make_prefix_labels(text, seed["status"]),
                    "meta": {
                        "source": "api_minimax_m25",
                        "seed_source": seed["meta"]["source"],
                        "variant_index": variant_index,
                    },
                }
                if generated["runtime_b"] is None and "expression" in seed.get("meta", {}):
                    generated["meta"]["expression"] = seed["meta"]["expression"]
                rows.append(generated)
                existing_counts[seed["id"]] = existing_counts.get(seed["id"], 0) + 1
        write_jsonl(output_path, rows)
        write_json(request_log_path, logs)
    if len(rows) != expected_total:
        raise RuntimeError(f"expected {expected_total} api rows, got {len(rows)}")
    return rows


def merge_and_split_datasets(
    curated_path: str | Path,
    api_path: str | Path,
    *,
    train_path: str | Path,
    val_path: str | Path,
    manifest_path: str | Path,
    val_ratio: float = 0.1,
    include_prefixes: bool = True,
    seed: int = 23,
) -> dict[str, Any]:
    curated_rows = read_jsonl(curated_path)
    api_rows = read_jsonl(api_path) if Path(api_path).exists() else []
    all_rows: list[dict[str, Any]] = []
    for row in curated_rows + api_rows:
        all_rows.append(row)
        if include_prefixes:
            for prefix in row.get("prefix_labels", [])[:-1]:
                unstable_row = {
                    "id": _stable_id("prefix", row["id"] + "::" + prefix["text"]),
                    "text": prefix["text"],
                    "status": prefix["status"],
                    "runtime_b": deepcopy(row["runtime_b"]),
                    "prefix_parent_id": row["id"],
                    "meta": {**row.get("meta", {}), "source": row["meta"]["source"] + "_prefix"},
                }
                all_rows.append(unstable_row)
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in STATUS_LABELS}
    for row in all_rows:
        grouped[row["status"]].append(row)
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for label, rows in grouped.items():
        rng.shuffle(rows)
        val_count = max(1, int(len(rows) * val_ratio))
        val_rows.extend(rows[:val_count])
        train_rows.extend(rows[val_count:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    manifest = {
        "ontology_version": "v1_text_qwen06b",
        "counts": {
            "curated_raw": len(curated_rows),
            "api_raw": len(api_rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "include_prefixes": include_prefixes,
        "val_ratio": val_ratio,
    }
    write_json(manifest_path, manifest)
    return manifest
