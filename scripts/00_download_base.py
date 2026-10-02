# -*- coding: utf-8 -*-
"""Illustrious 公式ベースモデルを Hugging Face からダウンロード。

学習ベース = Illustrious-XL-v0.1 (Illustrious系LoRAの定番ベース)。
既に models/ にあればスキップ。
"""
import shutil
from pathlib import Path

from common import ROOT, info, ok, warn, die

# (repo_id, filename) の候補。上から順に試す。
CANDIDATES = [
    ("OnomaAIResearch/Illustrious-xl-early-release-v0", "Illustrious-XL-v0.1.safetensors"),
]
DEST = ROOT / "models" / "Illustrious-XL-v0.1.safetensors"


def main():
    if DEST.exists() and DEST.stat().st_size > 1_000_000_000:
        ok(f"既に存在: {DEST} ({DEST.stat().st_size/1e9:.1f} GB) — スキップ")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        die("huggingface_hub が未インストールです。先に依存インストール(setup)を完了してください。")

    last_err = None
    for repo_id, filename in CANDIDATES:
        try:
            info(f"ダウンロード試行: {repo_id} / {filename}")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(ROOT / "models"),
            )
            src = Path(path)
            if src.resolve() != DEST.resolve():
                shutil.move(str(src), str(DEST))
            ok(f"完了: {DEST} ({DEST.stat().st_size/1e9:.1f} GB)")
            return
        except Exception as e:  # noqa
            warn(f"失敗: {repo_id} -> {e}")
            last_err = e

    die(f"全候補で失敗しました。手動でベースモデルを models/ に置いてください。最後のエラー: {last_err}")


if __name__ == "__main__":
    main()
