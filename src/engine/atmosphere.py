from __future__ import annotations


DEFAULT_EMPTY_AIR_BASE_TEMPERATURE = 20.0
AMBIENT_AIR_STRATIFICATION_DELTA = 6.0
AMBIENT_AIR_RESTORE_RATE = 0.35


def ambient_air_temperature_for_row(height: int, y: int, base_temperature: float) -> float:
    if height <= 1:
        return base_temperature
    clamped_y = max(0, min(height - 1, y))
    normalized_height = clamped_y / (height - 1)
    return base_temperature + (normalized_height - 0.5) * AMBIENT_AIR_STRATIFICATION_DELTA


def default_ambient_air_temperature_for_row(height: int, y: int) -> float:
    return ambient_air_temperature_for_row(height, y, DEFAULT_EMPTY_AIR_BASE_TEMPERATURE)
