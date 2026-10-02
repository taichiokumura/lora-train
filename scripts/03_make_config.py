# -*- coding: utf-8 -*-
"""train_config.json から sd-scripts 用の学習設定 TOML を生成する。

8GB VRAM / Illustrious SDXL 向けの省メモリ設定をまとめて出力。
"""
from pathlib import Path

from common import load_config, project_paths, ROOT, info, ok, warn, die


def fwd(path: Path) -> str:
    return str(path).replace("\\", "/")


def main():
    cfg = load_config()
    p = project_paths(cfg)

    base = (ROOT / cfg["base_model"]).resolve()
    if not base.exists():
        warn(f"ベースモデルが未配置: {base}  (00_download_base.py を実行)")

    img_dir = p["img_sub"]
    n_img = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    if n_img == 0:
        warn("学習画像が0枚です。01_prepare を先に実行してください。")

    repeats = cfg["num_repeats"]
    epochs = cfg["epochs"]
    bs = cfg["train_batch_size"]
    ga = cfg.get("gradient_accumulation_steps", 1)
    total = max(1, n_img * repeats * epochs // (bs * ga))
    warmup = round(total * cfg.get("lr_warmup_ratio", 0.05))

    res = int(cfg["resolution"])
    lr = float(cfg["unet_lr"])

    p["output"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)

    L = []
    a = L.append
    a("# === 自動生成: 03_make_config.py が train_config.json から生成 ===")
    a(f"# 画像{n_img}枚 x repeats{repeats} x epochs{epochs} / (bs{bs} x ga{ga}) = 約{total} steps")
    a("")
    a("# --- モデル / データ ---")
    a(f'pretrained_model_name_or_path = "{fwd(base)}"')
    a(f'train_data_dir = "{fwd(p["img"])}"')
    a(f'output_dir = "{fwd(p["output"])}"')
    a(f'output_name = "{cfg["project_name"]}"')
    a(f'logging_dir = "{fwd(p["logs"])}"')
    a('caption_extension = ".txt"')
    a("keep_tokens = 1")
    a("")
    a("# --- 解像度 / アスペクト比バケット ---")
    a(f'resolution = "{res},{res}"')
    a("enable_bucket = true")
    a("min_bucket_reso = 512")
    a("max_bucket_reso = 1536")
    a("bucket_reso_steps = 64")
    a("bucket_no_upscale = true")
    a("")
    a("# --- LoRA ネットワーク ---")
    a('network_module = "networks.lora"')
    a(f'network_dim = {cfg["network_dim"]}')
    a(f'network_alpha = {cfg["network_alpha"]}')
    if cfg.get("conv_dim", 0) and cfg["conv_dim"] > 0:
        a(f'network_args = [ "conv_dim={cfg["conv_dim"]}", "conv_alpha={cfg.get("conv_alpha", cfg["conv_dim"])}" ]')
    if cfg.get("train_unet_only", True):
        a("network_train_unet_only = true")
    a("")
    a("# --- 学習スケジュール ---")
    a(f"max_train_epochs = {epochs}")
    a(f"train_batch_size = {bs}")
    a(f"gradient_accumulation_steps = {ga}")
    a(f"save_every_n_epochs = {cfg['save_every_n_epochs']}")
    a('save_model_as = "safetensors"')
    a('save_precision = "fp16"')
    a(f"seed = {cfg.get('seed', 42)}")
    a("")
    a("# --- 最適化 ---")
    a(f'optimizer_type = "{cfg["optimizer"]}"')
    a(f"learning_rate = {lr}")
    a(f"unet_lr = {lr}")
    a(f'lr_scheduler = "{cfg["lr_scheduler"]}"')
    a(f"lr_warmup_steps = {warmup}")
    a("max_grad_norm = 1.0")
    a(f"min_snr_gamma = {cfg.get('min_snr_gamma', 5)}")
    a(f"noise_offset = {cfg.get('noise_offset', 0.0)}")
    a("prior_loss_weight = 1.0")
    a("")
    a("# --- 8GB VRAM 省メモリ ---")
    a(f'mixed_precision = "{cfg.get("mixed_precision", "bf16")}"')
    if cfg.get("full_bf16", True):
        a("full_bf16 = true")
    if cfg.get("fp8_base", False):
        a("fp8_base = true")
    a("gradient_checkpointing = true")
    a("cache_latents = true")
    a("cache_latents_to_disk = true")
    if cfg.get("cache_text_encoder_outputs", True):
        a("cache_text_encoder_outputs = true")
        a("cache_text_encoder_outputs_to_disk = true")
    a("sdpa = true")
    a("no_half_vae = true")
    a("persistent_data_loader_workers = true")
    a("max_data_loader_n_workers = 2")
    a("")

    p["config_toml"].write_text("\n".join(L) + "\n", encoding="utf-8")
    ok(f"設定TOML生成: {p['config_toml']}")
    info(f"総ステップ約{total} / warmup {warmup} / 解像度{res} / dim{cfg['network_dim']} alpha{cfg['network_alpha']}")
    if cfg.get("conv_dim", 0) > 0:
        info(f"conv有効: conv_dim{cfg['conv_dim']} (画風再現↑ / VRAM↑)")


if __name__ == "__main__":
    main()
