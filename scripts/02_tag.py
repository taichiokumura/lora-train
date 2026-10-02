# -*- coding: utf-8 -*-
"""自動タグ付け: WD14 tagger で各画像にbooruタグを付与し、
トリガーワードを先頭に付ける。

画風LoRAの考え方:
  「内容(content)」はタグで説明し、説明しきれない一貫した「画風(style)」を
  トリガーワードに吸わせる。よって全captionの先頭に trigger_word を入れる。
"""
import subprocess
import sys
from pathlib import Path

from common import load_config, project_paths, info, ok, warn, die, SD_SCRIPTS, VENV_PY


def run_tagger(cfg, img_dir: Path):
    script = SD_SCRIPTS / "finetune" / "tag_images_by_wd14_tagger.py"
    cmd = [
        str(VENV_PY), str(script), str(img_dir),
        "--onnx",
        "--repo_id", cfg["tagger_repo"],
        "--thresh", str(cfg["tagger_threshold"]),
        "--batch_size", "1",
        "--caption_extension", ".txt",
        "--remove_underscore",
        "--max_data_loader_n_workers", "2",
    ]
    remove = (cfg.get("remove_tags") or "").strip()
    if remove:
        cmd += ["--undesired_tags", remove]

    info("WD14 tagger 実行中 (初回はモデルDLで数分)...")
    r = subprocess.run(cmd, cwd=str(SD_SCRIPTS))
    if r.returncode != 0:
        die("WD14 tagger が失敗しました。onnxruntime のインストールを確認してください。")


def postprocess(cfg, img_dir: Path):
    trigger = cfg["trigger_word"].strip()
    remove = {t.strip() for t in (cfg.get("remove_tags") or "").split(",") if t.strip()}
    txts = list(img_dir.glob("*.txt"))
    if not txts:
        die("タグファイル(.txt)が生成されていません。tagger の出力を確認してください。")

    for txt in txts:
        tags = [t.strip() for t in txt.read_text(encoding="utf-8").split(",") if t.strip()]
        tags = [t for t in tags if t not in remove and t != trigger]
        tags = [trigger] + tags          # トリガーを必ず先頭に
        txt.write_text(", ".join(tags), encoding="utf-8")

    ok(f"タグ付け完了: {len(txts)}件。先頭トリガー = '{trigger}'")
    info(f"確認/手直し可: {img_dir}\\*.txt (不要タグは train_config.json の remove_tags へ)")


def main():
    cfg = load_config()
    p = project_paths(cfg)
    img_dir = p["img_sub"]
    if not img_dir.exists() or not any(img_dir.glob("*.png")):
        die(f"先に 01_prepare を実行してください (画像なし: {img_dir})")
    run_tagger(cfg, img_dir)
    postprocess(cfg, img_dir)


if __name__ == "__main__":
    main()
