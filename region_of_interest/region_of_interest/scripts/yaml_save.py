from pathlib import Path, PosixPath
from dataclasses import dataclass, asdict
import yaml

def save_yaml(path: PosixPath, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

def load_yaml(path: PosixPath) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))