# -*- coding: utf-8 -*-
"""データ準備: datasets/<project>/raw/ の元画像を学習用に整える。

- 壊れた画像/非画像をスキップ
- RGBに統一 (透過は白背景に合成)
- 巨大画像は長辺 MAX_SIDE に縮小 (アスペクト比維持)
- kohya形式フォルダ datasets/<project>/img/<repeats>_<trigger>/ に連番PNGで配置
"""
from pathlib import Path

from common import load_config, project_paths, info, ok, warn, die, IMAGE_EXTS

MAX_SIDE = 2048   # これより大きい長辺は縮小 (学習は1024なので十分)
MIN_SIDE = 512    # これ未満は警告 (低解像度は画風学習に不利)


def main():
    cfg = load_config()
    p = project_paths(cfg)

    if not p["raw"].exists():
        p["raw"].mkdir(parents=True, exist_ok=True)
        die(f"元画像フォルダを作成しました。ここに参考イラストを入れてから再実行してください:\n        {p['raw']}")

    try:
        from PIL import Image
    except ImportError:
        die("Pillow が未インストールです。依存インストール(setup)を完了してください。")

    raws = [f for f in sorted(p["raw"].iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    if not raws:
        die(f"画像が見つかりません: {p['raw']} に参考イラストを入れてください。")

    # 出力先をクリーンに作り直す
    sub = p["img_sub"]
    if sub.exists():
        for old in sub.iterdir():
            old.unlink()
    sub.mkdir(parents=True, exist_ok=True)

    saved, skipped, small = 0, 0, 0
    for f in raws:
        try:
            img = Image.open(f)
            img.load()
        except Exception as e:  # noqa
            warn(f"壊れた画像をスキップ: {f.name} ({e})")
            skipped += 1
            continue

        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")

        w, h = img.size
        if min(w, h) < MIN_SIDE:
            small += 1
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)

        saved += 1
        img.save(sub / f"img_{saved:04d}.png")

    ok(f"準備完了: {saved}枚 -> {sub}")
    if skipped:
        warn(f"{skipped}枚は壊れていたためスキップ")
    if small:
        warn(f"{small}枚は長辺{MIN_SIDE}px未満。画風学習には{MIN_SIDE}px以上推奨")

    # ステップ数の見積もり
    repeats = cfg["num_repeats"]
    epochs = cfg["epochs"]
    bs = cfg["train_batch_size"]
    ga = cfg.get("gradient_accumulation_steps", 1)
    total = saved * repeats * epochs // (bs * ga)
    info(f"見積り総ステップ数 = {saved} x repeats{repeats} x epochs{epochs} / (bs{bs} x ga{ga}) = 約 {total} steps")
    if total < 1000:
        warn("総ステップが少なめです。num_repeats か epochs を増やすと安定します (目安1500-2500)")
    elif total > 4000:
        warn("総ステップが多めです。過学習に注意 (目安1500-2500)")


if __name__ == "__main__":
    main()
