# LoRA 画風学習ツール (Illustrious SDXL / RTX 4060 Ti 8GB 向け)

参考イラストの**画風(画風・塗り・線)を学習**し、Stable Diffusion (reForge) で
その画風のイラストを生成するための LoRA を**本格的に**学習するツールです。
学習エンジンは業界標準の **kohya sd-scripts**、それを「データ準備→タグ付け→
設定生成→学習→テスト生成」まで一気通貫で回せるよう自動化しています。

```
lora-train/
├─ run.bat              ← これをダブルクリック（メニュー式）
├─ setup.bat           ← 環境の再構築/修復（通常不要・構築済み）
├─ train_config.json   ← ★設定はここを編集するだけ
├─ venv/               ← Python環境 (構築済み)
├─ sd-scripts/         ← 学習エンジン (kohya)
├─ models/            ← ベースモデル (Illustrious-XL-v0.1)
├─ datasets/<project>/raw/   ← ★参考イラストをここに入れる
│              /img/         ← 自動生成 (学習用画像+タグ)
├─ output/<project>/  ← 完成LoRA (.safetensors) とテスト画像
├─ configs/           ← 自動生成の学習設定 (.toml) / accelerate設定
└─ scripts/           ← 自動化スクリプト 00〜05
```

---

## クイックスタート (最短手順)

1. **画像を入れる**
   `datasets\mystyle\raw\` を作り、学習させたい画風のイラストを **20〜50枚** 入れる。
   （`run.bat` の `1` を一度実行すると raw フォルダが自動で作られます）

2. **`run.bat` をダブルクリック** → メニューが出ます。

3. メニューで順に実行：
   - `0` … ベースモデルDL（初回のみ・構築時にDL済みなら不要）
   - `1` … データ準備（raw の画像を学習用に整形）
   - `2` … 自動タグ付け（WD14。初回はタガーDLで数分）
   - `3` … 学習設定(toml)を生成
   - `4` … **学習**（8GBではSDXLは時間がかかります。目安1〜3時間）
   - `5` … テスト生成（reForge を `--api` で起動しておく）

   `A` を選べば `1→2→3→4` を自動で連続実行します。

4. 完成した LoRA は `output\mystyle\mystyle.safetensors`。
   reForge の `models\Lora` に置き、プロンプトに `<lora:mystyle:0.8>` と
   トリガーワード `mystyle` を入れて生成。

---

## 設定 (train_config.json) の要点

| 項目 | 意味 | 既定 |
|---|---|---|
| `project_name` | プロジェクト名（フォルダ名にも使用） | mystyle |
| `trigger_word` | 画風を呼び出す合言葉。生成時にプロンプト先頭へ | mystyle |
| `num_repeats` | 1枚あたりの繰り返し回数 | 4 |
| `epochs` | エポック数 | 10 |
| `network_dim` / `alpha` | LoRA容量。画風は大きめ有利 | 32 / 16 |
| `conv_dim` / `conv_alpha` | >0でConv層も学習＝画風再現↑/VRAM↑ | 0 (無効) |
| `unet_lr` | 学習率 | 1e-4 |
| `resolution` | 学習解像度 | 1024 |
| `tagger_threshold` | タグ採用のしきい値（低いほどタグ増） | 0.35 |
| `remove_tags` | 付けたくないタグ（カンマ区切り） | "" |

**総ステップ数の目安は 1500〜2500**。`画像枚数 × num_repeats × epochs` で決まります。
`1`（準備）実行時に見積りステップ数が表示されるので、少なすぎ/多すぎなら
`num_repeats` か `epochs` を調整してください。

### 画風LoRAのコツ
- **タグ付けの考え方**：内容（髪色・服・構図など）はタグで説明し、説明しきれない
  一貫した「画風」をトリガーワードに吸わせます。本ツールは全captionの先頭に
  自動で `trigger_word` を入れます。
- **画風そのものを表すタグは消す**：例えば全画像が同じ絵柄なら、絵柄に紐づく
  特徴タグ（例 `flat color` 等）を `remove_tags` に入れると画風がトリガーに集約されます。
- **画像は構図・ポーズ・背景がバラけている方が良い**（特定の内容に過学習しにくい）。
- **強さ調整**：生成時 `<lora:名前:0.6〜0.9>`。効きすぎなら下げる。

---

## VRAM 8GB のための省メモリ設定（自動適用済み）

`03_make_config.py` が以下を自動で設定します：
- `mixed_precision=bf16` + `full_bf16`（Ada世代対応）
- `gradient_checkpointing`（メモリ大幅節約）
- `cache_latents` + `cache_text_encoder_outputs`（潜在表現とTE出力をディスクへ）
- `network_train_unet_only`（UNetのみ学習＝省メモリ・画風向き）
- `optimizer=AdamW8bit`（bitsandbytes）
- `sdpa`（xformers不要の省メモリAttention）/ `no_half_vae`

### それでも OOM（メモリ不足）になったら
`train_config.json` を次の順で調整：
1. `"fp8_base": true` にする（UNetをfp8化）
2. `"resolution": 896`（または 768）に下げる
3. `"network_dim": 16`, `"network_alpha": 8` に下げる
4. `"conv_dim": 0`（Conv層を使っていれば無効化）

---

## トラブルシューティング

- **タガーが動かない**：`onnxruntime-gpu` が入っているか確認（構築済み）。
- **AdamW8bit でエラー**：`optimizer` を `Adafactor` に変更（bitsandbytes不要）。
- **テスト生成が繋がらない**：reForge を `webui-user.bat` の
  `set COMMANDLINE_ARGS=--api` 付きで起動。
- **学習の進捗/VRAM確認**：別のPowerShellで `nvidia-smi -l 2`。
- **環境を作り直したい**：`setup.bat` を実行。

---

## コマンドで直接実行したい場合

```powershell
cd C:\Users\okuta\lora-train
venv\Scripts\python.exe scripts\01_prepare.py
venv\Scripts\python.exe scripts\02_tag.py
venv\Scripts\python.exe scripts\03_make_config.py
venv\Scripts\python.exe scripts\04_train.py
venv\Scripts\python.exe scripts\05_test.py
```
