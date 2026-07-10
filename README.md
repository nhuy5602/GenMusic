# GenMusic VN

GenMusic VN là pipeline sinh bài hát tiếng Việt dùng trực tiếp model ASLP-lab/DiffRhythm. Project không còn dùng classifier để chọn nhạc, custom Transformer, symbolic composer hay TTS add-on làm backend sinh chính.

## Kiến trúc

```text
Input lyric/style
  -> chuẩn hóa tiếng Việt
  -> LRC sentence timing
  -> official DiffRhythm infer/infer.py
  -> WAV/MP3 có nhạc và vocal
  -> objective evaluation + telemetry
```

Project giữ lại phần orchestration, tạo LRC, random smoke dataset, Kaggle automation và báo cáo. Snapshot mã nguồn CFM/DiT, G2P, inference và training của upstream được vendor tại `third_party/DiffRhythm` để demo reproducible, không clone GitHub lúc chạy. Bản vendor bổ sung nhận diện nhánh `vi` và eSpeak Vietnamese cho lyric có dấu. Checkpoint DiT/VAE và MuQ vẫn được tải riêng theo cơ chế của DiffRhythm.

Upstream hiện có:
- infer/infer.py cho inference;
- model/cfm.py, model/dit.py, train/train.py;
- checkpoint ASLP-lab/DiffRhythm-1_2 và ASLP-lab/DiffRhythm-1_2-full;
- dataset train dạng train.scp với latent, LRC và style tensor;
- license Apache-2.0 cho code và DiT weights.

## Cài đặt

```powershell
pip install -e ".[diffrhythm]"
```

Cài espeak-ng theo hướng dẫn upstream. Windows cần cấu hình:

```
PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
PHONEMIZER_ESPEAK_PATH=C:\Program Files\eSpeak NG
```

DiffRhythm base cần GPU. Upstream khuyến nghị --chunked khi VRAM thấp; inference full song không phải job CPU nhẹ.

## Chạy web app

```powershell
python -m genmusic_vn.server --port 8000
```

Mở http://127.0.0.1:8000. Nút Generate sẽ stage/submit job Kaggle dùng backend ASLP-lab/DiffRhythm.

Input duration được quy đổi theo quy tắc upstream: 95 giây hoặc 96-285 giây. Nếu nhập ngắn hơn 95 giây, hệ thống dùng 95 giây và ghi lại duration yêu cầu trong request.json.

## Chạy CLI

Stage job, chưa submit Kaggle:

```powershell
python -m genmusic_vn.cli generate --text "Một đêm mưa, em còn nhớ con đường cũ." --duration 95 --genre "Vietnamese pop ballad, piano, warm strings" --no-submit
```

Submit và chờ output:

```powershell
python -m genmusic_vn.cli generate --text "Một đêm mưa, em còn nhớ con đường cũ." --duration 95 --wait
```

Chạy inference upstream trực tiếp tại local:

```powershell
python -m genmusic_vn.cli generate-local-diffrhythm --lyrics data/song.txt --style "Vietnamese pop ballad, piano, emotional strings" --duration 95 --out outputs/local_diffrhythm
```

## Random smoke dataset

Official train/train.py cần PyTorch. Lệnh sau tạo đúng format upstream: train.scp, latent/*.pt, lrc/*.pt, style/*.pt và config model nhỏ để smoke test:

```powershell
python -m genmusic_vn.cli make-random-diffrhythm-dataset --out datasets/random_diffrhythm --count 4 --max-frames 64 --seed 5602
python -m genmusic_vn.cli validate-diffrhythm-dataset --dataset datasets/random_diffrhythm
```

Chạy train chính thức với dataset random:

```powershell
python -m genmusic_vn.cli train-diffrhythm --dataset datasets/random_diffrhythm --epochs 1 --batch-size 1
```

Máy Windows hiện tại nếu thiếu torch sẽ trả trạng thái cần dependency, không tạo file giả. Hãy chạy ba lệnh này trong Kaggle GPU hoặc cài extra diffrhythm.

## Distillation

```powershell
python -m genmusic_vn.cli distill-diffrhythm --out outputs/diffrhythm_distillation_plan.json --teacher-ref ASLP-lab/DiffRhythm-1_2 --teacher-steps 32 --student-steps 4
```

Lệnh này chỉ tạo plan và metadata. Upstream cung cấp CFM/DiT training nhưng không kèm sẵn một checkpoint student 4 bước; project không giả mạo checkpoint distilled.

## Tiền xử lý lyric

```powershell
python -m genmusic_vn.cli normalize-lyrics --input data/song.txt --out outputs/song.normalized.txt
python -m genmusic_vn.cli lyrics-g2p --input outputs/song.normalized.txt --out outputs/song.g2p.json
python -m genmusic_vn.cli align-lyrics --audio data/song.wav --lyrics outputs/song.normalized.txt --segments data/song_segments.json --out outputs/song.lrc
```

G2P có fallback xác định rule-based-ipa; muốn dùng eSpeak hãy cài phonemizer.

## Đánh giá

MOS/CMOS được tạm bỏ khỏi gate theo yêu cầu. Đánh giá hiện tại chỉ dùng metric khách quan khi có đủ artifact:

```powershell
python -m genmusic_vn.cli evaluate-diffrhythm --generated outputs/generated.wav --reference data/reference.wav --generated-text outputs/asr.txt --reference-text data/lyrics.txt --out outputs/diffrhythm_evaluation
```

Báo cáo:
- FAD nếu có embedding audio;
- MCD nếu có audio reference;
- FFE khi bổ sung F0;
- WER khi có transcript reference/generated;
- plot objective metrics;
- MOS và CMOS ghi null/skipped-by-request.

`project-report` sinh telemetry và các biểu đồ project: thời gian từ input tới WAV/MP3, success/error, retry Kaggle, emotion vs BPM và user rating. Hai biểu đồ cuối hiển thị trạng thái chưa có dữ liệu thay vì tự bịa số khi request không chứa emotion/BPM hoặc MOS/rating đang được bỏ qua.

## Source vendor

Source DiffRhythm đã đóng gói trong `third_party/DiffRhythm` tại commit upstream `28ad63c0f096fe2ee258bcabbcf081d5d9366afd`. Khi stage job, project đóng gói snapshot này vào `genmusic_vn_source.zip`; kernel Kaggle giải nén và chạy source local. Chỉ model weights và các dependency được tải trong môi trường Kaggle.

## Kaggle

Job tự động tạo:
- private input dataset chứa request.json, lyrics.lrc;
- kernel metadata GPU;
- source zip chứa snapshot vendor DiffRhythm;
- install `third_party/DiffRhythm/requirements.txt`;
- tải checkpoint Hugging Face trong Kaggle;
- chạy infer/infer.py;
- tải WAV và MP3 nếu ffmpeg có sẵn.

Theo dõi job:

```powershell
python -m genmusic_vn.cli refresh-kaggle --state outputs/<run_id>/kaggle_job/job_state.json
python -m genmusic_vn.cli project-report --source outputs --out outputs/project_report
```

## Kiểm thử

```powershell
python -m compileall -q genmusic_vn
python -m unittest discover -s tests -v
```

Không commit checkpoint upstream, audio dataset hoặc output lớn vào Git.
