# region_of_interest/__init__.py
from __future__ import annotations
from importlib.resources import files
import yaml

# configs.yaml is at: region_of_interest/externals/configs.yaml
_CFG_TEXT = (files("region_of_interest.externals") / "configs.yaml").read_text(encoding="utf-8")
CONFIG = yaml.safe_load(_CFG_TEXT)  # dict

def get_config() -> dict:
    """Public accessor (avoids importing yaml in callers)."""
    return CONFIG
