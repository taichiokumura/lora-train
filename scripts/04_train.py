# -*- coding: utf-8 -*-
"""学習を実行する。accelerate 経由で sdxl_train_network.py を起動。"""
import subprocess
import sys

from common import load_config, project_paths, ROOT, SD_SCRIPTS, VENV_PY, info, ok, die


def main():
    cfg = load_config()
    p = project_paths(cfg)

    accel_yaml = ROOT / "configs" / "accelerate_config.yaml"
    train_py = SD_SCRIPTS / "sdxl_train_network.py"

    if not p["config_toml"].exists():
        die(f"設定TOMLがありません。先に 03_make_config を実行してください: {p['config_toml']}")
    if not any(p["img_sub"].glob("*.txt")):
        die("タグ(.txt)がありません。先に 02_tag を実行してください。")

    cmd = [
        str(VENV_PY), "-m", "accelerate.commands.launch",
        "--config_file", str(accel_yaml),
        "--num_cpu_threads_per_process", "2",
        str(train_py),
        "--config_file", str(p["config_toml"]),
    ]
    info("学習開始 (8GB VRAM・SDXLは時間がかかります。GPU使用率/VRAMは別窓の nvidia-smi で確認可)")
    info(" ".join(cmd))
    r = subprocess.run(cmd, cwd=str(SD_SCRIPTS))
    if r.returncode != 0:
        die(f"学習が異常終了しました (code {r.returncode})。OOM時は train_config.json の fp8_base=true / resolution=896 を試してください。")
    ok(f"学習完了! LoRA出力先: {p['output']}")
    info("次は 05_test.py でテスト生成、または reForge で <lora:%s:0.8> を使用" % cfg["project_name"])


if __name__ == "__main__":
    main()
