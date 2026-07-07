import os


DEFAULT_SEASON = "2025_26"
SEASON_ENV_VAR = "FPL_SEASON"


def get_current_season() -> str:
    return os.environ.get(SEASON_ENV_VAR, DEFAULT_SEASON)
