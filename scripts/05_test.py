# -*- coding: utf-8 -*-
"""テスト生成: 学習したLoRAを reForge にコピーし、API経由で確認画像を生成。

- reForge を --api 付きで起動しておくこと (sd-auto と同じ前提)。
- LoRA強度を 0.6 / 0.8 / 1.0 で振って比較画像を出力。
"""
import base64
import io
import shutil
import sys
from pathlib import Path

from common import load_config, project_paths, info, ok, warn, die

WEIGHTS = [0.6, 0.8, 1.0]


def pick_lora(out_dir: Path, name: str) -> Path:
    final = out_dir / f"{name}.safetensors"
    if final.exists():
        return final
    cps = sorted(out_dir.glob(f"{name}-*.safetensors"))
    if cps:
        return cps[-1]
    die(f"LoRAが見つかりません: {out_dir} (先に学習を完了してください)")


def main():
    cfg = load_config()
    p = project_paths(cfg)
    name = cfg["project_name"]

    try:
        import requests
    except ImportError:
        die("requests が未インストールです。依存インストールを確認してください。")

    lora = pick_lora(p["output"], name)
    lora_dir = Path(cfg["reforge_lora_dir"])
    if not lora_dir.exists():
        die(f"reForgeのLoraフォルダが見つかりません: {lora_dir} (train_config.json で修正)")
    dest = lora_dir / lora.name
    shutil.copy2(lora, dest)
    ok(f"LoRAをreForgeへコピー: {dest.name}")

    lora_tag_name = lora.stem
    api = cfg["reforge_api_url"].rstrip("/")
    samples = p["samples"]
    samples.mkdir(parents=True, exist_ok=True)
    trigger = cfg["trigger_word"]

    try:
        requests.get(f"{api}/sdapi/v1/sd-models", timeout=5)
    except Exception:  # noqa
        die(f"reForge APIに接続できません: {api}\n        reForgeを --api 付きで起動してください。")

    n = 0
    for pi, base_prompt in enumerate(cfg["test_prompts"], 1):
        for w in WEIGHTS:
            prompt = f"{trigger}, {base_prompt}, <lora:{lora_tag_name}:{w}>"
            payload = {
                "prompt": prompt,
                "negative_prompt": cfg.get("test_negative", ""),
                "steps": 26,
                "width": 832,
                "height": 1216,
                "cfg_scale": 4.0,
                "sampler_name": "Euler a",
                "seed": 1234,
            }
            info(f"生成中 prompt{pi} weight{w} ...")
            r = requests.post(f"{api}/sdapi/v1/txt2img", json=payload, timeout=600)
            r.raise_for_status()
            for img_b64 in r.json().get("images", []):
                img = base64.b64decode(img_b64.split(",", 1)[-1])
                out = samples / f"p{pi}_w{w}.png"
                out.write_bytes(img)
                n += 1

    ok(f"テスト生成完了: {n}枚 -> {samples}")
    info("強度(weight)で画風の効き具合を比較してください。0.7-0.9が常用域です。")


if __name__ == "__main__":
    main()
