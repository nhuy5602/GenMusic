# Kaggle GPU workflow

Local machine chỉ tạo `prompt_pack.json`. Kaggle chạy inference hoặc fine-tune khi cần GPU.

## MusicGen inference

```bash
pip install -U audiocraft
python musicgen_generate.py \
  --prompt-pack /kaggle/input/genmusic-vn-prompt-pack/prompt_pack.json \
  --model facebook/musicgen-small \
  --out /kaggle/working/genmusic_vn
```

Gợi ý model:

- `facebook/musicgen-small`: nhẹ nhất, hợp thử nhanh.
- `facebook/musicgen-medium`: cân bằng chất lượng/compute.
- `facebook/musicgen-melody`: dùng khi muốn conditioning bằng melody/chroma ở bước mở rộng.

## Stable Audio Open inference

```bash
pip install -U diffusers transformers accelerate soundfile
python stable_audio_open_generate.py \
  --prompt-pack /kaggle/input/genmusic-vn-prompt-pack/prompt_pack.json \
  --out /kaggle/working/genmusic_vn
```

## Stable Audio 3 CLI

Nếu notebook đã cài repo/tooling Stable Audio 3 và có lệnh `stable-audio`:

```bash
python stable_audio_cli_generate.py \
  --prompt-pack /kaggle/input/genmusic-vn-prompt-pack/prompt_pack.json \
  --model small-music \
  --out /kaggle/working/genmusic_vn
```

## Fine-tune

Không fine-tune trên máy local. Nếu cần fine-tune:

1. Đưa dataset audio + metadata lên Kaggle Dataset.
2. Dùng notebook riêng có GPU.
3. Lưu checkpoint vào `/kaggle/working`.
4. Download checkpoint hoặc publish thành Kaggle Dataset.

MusicGen training/fine-tuning cần metadata JSON cạnh audio và cấu hình AudioCraft/Dora phù hợp. Nên bắt đầu bằng inference/prompt tuning trước khi fine-tune.

