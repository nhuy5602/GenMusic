# GenMusic VN

Pipeline tạo nhạc nền từ văn bản tiếng Việt:

1. Nhận đoạn text tiếng Việt.
2. Phân tích cảm xúc, sắc thái, năng lượng.
3. Suy luận tempo, key, scale, vòng hợp âm và hướng cao độ.
4. Viết lại đoạn text thành lyric draft tiếng Việt.
5. Sinh prompt tiếng Anh giàu metadata cho MusicGen hoặc Stable Audio.
6. Tạo guide track WAV/MIDI nhẹ ở local, hoặc xuất prompt pack để chạy model thật trên Kaggle GPU.

Local project này **không train model nặng trên máy**. Máy local chỉ làm orchestration, UI, prompt engineering và guide track. MusicGen/Stable Audio chạy trên Kaggle.

## Chạy nhanh local

```powershell
python -m genmusic_vn.cli generate --text "Một chiều mưa, tôi nhớ về những con phố cũ và một lời hứa chưa kịp nói." --duration 18
python -m genmusic_vn.server --port 8000
```

Mở `http://127.0.0.1:8000` để dùng UI.

Output nằm trong `outputs/<run_id>/`, gồm:

- `report.json`: toàn bộ phân tích, prompt, lyrics, harmony.
- `guide.wav`: bản phác nghe nhanh bằng synth nhẹ.
- `guide.mid`: MIDI melody/chord để chỉnh trong DAW.
- `prompt_pack.json`: file đưa lên Kaggle để sinh audio bằng model thật.

## Chạy MusicGen trên Kaggle

1. Tạo prompt pack local:

```powershell
python -m genmusic_vn.cli export-kaggle --text "Đêm thành phố sáng lên, lòng người vẫn tìm một nơi bình yên." --duration 30 --out outputs
```

2. Upload file `prompt_pack.json` trong output folder lên Kaggle Dataset hoặc notebook input.
3. Trong Kaggle Notebook bật GPU, chạy:

```bash
pip install -U audiocraft
python /kaggle/input/<project-files>/kaggle/musicgen_generate.py \
  --prompt-pack /kaggle/input/<your-dataset>/prompt_pack.json \
  --model facebook/musicgen-small \
  --out /kaggle/working/genmusic_vn
```

Với GPU mạnh hơn có thể dùng `facebook/musicgen-medium` hoặc `facebook/musicgen-melody`. AudioCraft/MusicGen yêu cầu GPU cho inference thực tế.

## Chạy Stable Audio trên Kaggle

Stable Audio Open có license riêng và thường cần GPU. Khi notebook đã cài `diffusers`, `torch`, `soundfile`:

```bash
pip install -U diffusers transformers accelerate soundfile
python /kaggle/input/<project-files>/kaggle/stable_audio_open_generate.py \
  --prompt-pack /kaggle/input/<your-dataset>/prompt_pack.json \
  --out /kaggle/working/genmusic_vn
```

Nếu dùng Stable Audio 3 repo/CLI, xem `kaggle/README.md` để chạy bằng `stable-audio`.

## Cấu trúc

```text
genmusic_vn/
  emotion.py          # phân tích cảm xúc tiếng Việt
  music_theory.py     # key, scale, chord, melody contour
  lyric_writer.py     # viết lại lyric draft tiếng Việt
  prompt_builder.py   # prompt MusicGen/Stable Audio
  generators/         # guide track local và adapter model
  pipeline.py         # orchestration
  cli.py              # command line
  server.py           # web UI/API bằng Python stdlib
web/
  index.html
  app.css
  app.js
kaggle/
  musicgen_generate.py
  stable_audio_open_generate.py
  stable_audio_cli_generate.py
```

## Nguồn kỹ thuật chính

- AudioCraft/MusicGen: https://github.com/facebookresearch/audiocraft
- MusicGen docs: https://raw.githubusercontent.com/facebookresearch/audiocraft/main/docs/MUSICGEN.md
- Stable Audio Open 1.0: https://huggingface.co/stabilityai/stable-audio-open-1.0
- Stable Audio 3: https://github.com/Stability-AI/stable-audio-3

## Automatic Kaggle demo flow

For product demos, use the automated path. Local still does not train heavy models; it prepares the job, submits it to Kaggle GPU through the Kaggle API, polls status, and downloads output artifacts back to `outputs/`.

### 1. Configure Kaggle API once

```powershell
pip install -U kaggle
mkdir $HOME\.kaggle
# Put kaggle.json from Kaggle Account Settings into $HOME\.kaggle\kaggle.json
```

You can also use environment variables:

```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

### 2. One-command GPU job

```powershell
python -m genmusic_vn.cli kaggle-auto --text "Mot dem thanh pho sang len, toi muon mot nen nhac binh yen nhung co hy vong." --duration 30 --wait
```

What this does:

1. Creates local analysis, lyrics, prompt pack, MIDI and guide WAV.
2. Creates a private Kaggle Dataset containing `prompt_pack.json`.
3. Creates and runs a private Kaggle Kernel with GPU enabled.
4. Polls the job until it finishes when `--wait` is used.
5. Downloads Kaggle output to `outputs/<run_id>/kaggle_job/downloaded_output/`.

If Kaggle credentials are missing, the command still stages all files in `outputs/<run_id>/kaggle_job/` and writes `run_commands.ps1` so the demo can continue locally.

### 3. Sync trained model or generated artifact back to local

If a trained checkpoint is published as a Kaggle Dataset:

```powershell
python -m genmusic_vn.cli sync-kaggle --source dataset --ref <username>/<dataset-slug> --out models/current
```

If the artifact is a Kaggle Kernel output:

```powershell
python -m genmusic_vn.cli sync-kaggle --source kernel --ref <username>/<kernel-slug> --out models/current
```

The sync command writes `models/current/model_manifest.json` with source, timestamp and file list.
