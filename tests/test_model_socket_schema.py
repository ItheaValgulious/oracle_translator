from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm.model_socket_schema import normalize_model_socket, validate_model_socket


def build_model_socket(politeness: float) -> dict:
    return {
        "subject_kind": "summon_material",
        "subject": {"material_template": "fire"},
        "reaction": {"reaction_template": "burn"},
        "release": {"release_template": "spray"},
        "motion": {
            "motion_template": "flow",
            "motion_direction": "forward",
            "origin": "self",
            "target": "enemy",
        },
        "expression": {"politeness": politeness},
    }


class ModelSocketSchemaTests(unittest.TestCase):
    def test_normalize_politeness_string_to_float(self) -> None:
        normalized = normalize_model_socket(build_model_socket("0.72"))  # type: ignore[arg-type]
        self.assertAlmostEqual(normalized["expression"]["politeness"], 0.72)

    def test_validate_allows_continuous_runtime_politeness(self) -> None:
        validate_model_socket(build_model_socket(0.72))

    def test_validate_rejects_out_of_range_politeness(self) -> None:
        with self.assertRaises(ValueError):
            validate_model_socket(build_model_socket(1.2))

    def test_validate_can_require_binary_training_politeness(self) -> None:
        validate_model_socket(build_model_socket(1.0), require_binary_politeness=True)
        with self.assertRaises(ValueError):
            validate_model_socket(build_model_socket(0.72), require_binary_politeness=True)


if __name__ == "__main__":
    unittest.main()
