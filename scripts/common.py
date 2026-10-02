# -*- coding: utf-8 -*-
"""共通ユーティリティ: パス解決 / 設定読み込み / ログ。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # lora-train/
CONFIG_PATH = ROOT / "train_config.json"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
SD_SCRIPTS = ROOT / "sd-scripts"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        die(f"設定ファイルが見つかりません: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    # コメント用キー(_始まり)は無視できるが、そのまま残してOK
    return cfg


def project_paths(cfg: dict) -> dict:
    name = cfg["project_name"]
    ds = ROOT / "datasets" / name
    return {
        "name": name,
        "dataset": ds,
        "raw": ds / "raw",                              # 元画像を置く場所
        "img": ds / "img",                              # kohya形式 (NN_trigger)
        "img_sub": ds / "img" / f"{cfg['num_repeats']}_{cfg['trigger_word']}",
        "output": ROOT / "output" / name,
        "config_toml": ROOT / "configs" / f"{name}.toml",
        "samples": ROOT / "output" / name / "samples",
        "logs": ROOT / "logs",
    }


def info(msg):  print(f"[INFO] {msg}")
def warn(msg):  print(f"[WARN] {msg}")
def ok(msg):    print(f"[ OK ] {msg}")


def die(msg, code=1):
    print(f"[ERROR] {msg}")
    sys.exit(code)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
