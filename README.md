# GenMusic VN

GenMusic VN là pipeline sinh giọng hát tiếng Việt bằng Conditional Flow Matching (CFM) và MicroDiT. Model nhận lyric tiếng Việt, backing mel và MuQ style anchor, sau đó sinh vocal mel và giải mã bằng Vocos.

README này là hướng dẫn bắt đầu từ một máy và một tài khoản Kaggle mới. Không cần sửa username hoặc dataset slug trong source code.

## Luồng khuyến nghị

```text
Clone repo
  -> tạo .env riêng
  -> smoke test local
  -> preprocess toàn bộ các part trên Kaggle
  -> train 20 epoch
  -> đánh giá Whisper/CFG/plot
  -> resume thêm epoch nếu lời chưa rõ
```

Local chỉ phù hợp để kiểm tra nhanh. Preprocess và train full dataset nên chạy trên Kaggle GPU.

## 1. Yêu cầu

- Git.
- [uv](https://docs.astral.sh/uv/) để quản lý Python và dependency.
- Python 3.13; `uv` sẽ đọc phiên bản từ `.python-version`.
- FFmpeg và `espeak-ng` có trong `PATH` nếu preprocess/generate local.
- Một tài khoản Kaggle đã có quyền dùng GPU.
- Kaggle access token mới, thường bắt đầu bằng `KGAT_` hoặc `KGAT-`.

Kiểm tra công cụ hệ thống:

```powershell
git --version
uv --version
ffmpeg -version
espeak-ng --version
```

## 2. Clone và cài dependency

```powershell
git clone https://github.com/nhuy5602/GenMusic.git
cd GenMusic
uv sync --extra dev
uv run python cli.py --help
```

Không chạy `pip install` hoặc tự tạo virtual environment. Tất cả lệnh Python trong project dùng `uv run`.

## 3. Tạo `.env`

PowerShell:

```powershell
Copy-Item .env.example .env
```

Bash:

```bash
cp .env.example .env
```

Tối thiểu sửa các dòng sau:

```env
KAGGLE_USERNAME=YOUR_KAGGLE_USERNAME
KAGGLE_API_TOKEN=KGAT_YOUR_ACCESS_TOKEN

# Dùng khi preprocess một dataset riêng lẻ.
KAGGLE_RAW_DATASET_REF=OWNER/vietnamese-music-dataset-version3-part6

# Chỉ điền một trong hai dòng sau khi đã có dữ liệu preprocess.
KAGGLE_PROCESSED_KERNEL_REF=
KAGGLE_PROCESSED_DATASET_REF=
```

Quy tắc cấu hình:

- `.env` được ưu tiên trước `~/.kaggle/kaggle.json`.
- `.env` đã nằm trong `.gitignore`; không commit hoặc gửi token cho người khác.
- `KAGGLE_PROCESSED_KERNEL_REF` là output của một Kaggle preprocess kernel và là lựa chọn ưu tiên.
- `KAGGLE_PROCESSED_DATASET_REF` chỉ dùng khi dữ liệu preprocess đã được publish thành Kaggle Dataset.
- Source code không có dataset ref thuộc tài khoản cá nhân làm fallback.

Kiểm tra token:

```powershell
uv run kaggle datasets list -s vietnamese-music-dataset-version3-part
```

Nếu nhận `401` hoặc `403`, hãy tạo token mới và kiểm tra tài khoản có quyền truy cập dataset/kernel tương ứng.

## 4. Smoke test local

Thêm 1–2 file `.wav`, `.mp3`, `.flac` hoặc `.m4a` vào `dataset/vietnamese_songs/`.

### 4.1. Preprocess

```powershell
uv run python cli.py preprocess-raw `
  --input dataset/vietnamese_songs `
  --output dataset/diff_rhythm_dataset `
  --whisper-model base `
  --max-files 2 `
  --keep-separated-count 2
```

Pipeline thực hiện Demucs, Whisper ASR, chuẩn hóa/G2P tiếng Việt, tạo vocal/backing mel và MuQ style anchor.

Kiểm tra dataset:

```powershell
uv run python cli.py validate-dataset --dataset dataset/diff_rhythm_dataset
```

Dataset hợp lệ có dạng:

```text
dataset/diff_rhythm_dataset/
  config.json
  records.jsonl
  mels/
    <id>_vocal.pt
    <id>_backing.pt
    <id>_style.pt
```

### 4.2. Train nhỏ

```powershell
uv run python cli.py train-self `
  --dataset dataset/diff_rhythm_dataset `
  --checkpoint outputs/local_smoke.pt `
  --epochs 1 `
  --batch-size 2 `
  --dim 128 `
  --depth 2 `
  --heads 4 `
  --frames-per-chunk 192 `
  --device cpu
```

### 4.3. Generate và đánh giá

Luôn truyền reference dataset để inference có backing mel và style anchor giống lúc train:

```powershell
uv run python cli.py generate-local `
  --text "Đêm nay thành phố lên đèn, em nghe tiếng mưa dịu dàng." `
  --checkpoint outputs/local_smoke.pt `
  --reference-dataset dataset/diff_rhythm_dataset `
  --duration 8 `
  --steps 32 `
  --guidance-scale 2 `
  --vocoder vocos `
  --out outputs/local_generation
```

```powershell
uv run python cli.py evaluate-self `
  --generated outputs/local_generation/final.wav `
  --generated-text "Đêm nay thành phố lên đèn, em nghe tiếng mưa dịu dàng." `
  --out outputs/local_evaluation
```

Smoke test chỉ xác nhận pipeline chạy đúng; một hoặc hai bài không đủ để tạo giọng hát rõ.

## 5. Full dataset trên Kaggle

### Hai cách nhận dữ liệu preprocess

1. Nhanh nhất: chủ tài khoản cũ chia sẻ hoặc public các preprocess kernel đã `COMPLETE`. Tài khoản mới chỉ cần các ref dạng `OWNER/KERNEL-SLUG`.
2. Độc lập hoàn toàn: tài khoản mới tự preprocess tất cả part theo phần dưới đây.

Kernel ref phải truy cập được từ tài khoản mới. Chỉ copy source code không tự cấp quyền vào private Kaggle kernel/dataset của tài khoản cũ.

### 5.1. Tự tìm và preprocess mọi `partX`

Script sau tự tìm các dataset có tên chính xác `vietnamese-music-dataset-version3-partX`, đếm toàn bộ file và không giới hạn 1.000 file:

```powershell
uv run python scripts/run_kaggle_all_parts.py `
  --max-new-jobs 2 `
  --whisper-model base
```

Kaggle thường chỉ cho hai batch GPU sessions đồng thời. Khi hai job đầu hoàn tất, chạy lại và truyền các ref đã xong bằng `--reuse`:

```powershell
uv run python scripts/run_kaggle_all_parts.py `
  --reuse 1=YOUR_USERNAME/PART1_KERNEL `
  --reuse 2=YOUR_USERNAME/PART2_KERNEL `
  --max-new-jobs 2 `
  --whisper-model base
```

Lặp lại với part 3–4 rồi 5–6. Mỗi lần chạy tạo một file:

```text
outputs/kaggle_all_parts/allparts-<timestamp>/state.json
```

Giữ lại sáu ref cuối cùng theo mẫu:

```text
1=YOUR_USERNAME/PART1_KERNEL
2=YOUR_USERNAME/PART2_KERNEL
3=YOUR_USERNAME/PART3_KERNEL
4=YOUR_USERNAME/PART4_KERNEL
5=YOUR_USERNAME/PART5_KERNEL
6=YOUR_USERNAME/PART6_KERNEL
```

Theo dõi job:

```powershell
uv run kaggle kernels status YOUR_USERNAME/PART1_KERNEL
uv run python scripts/check_kernel_progress.py YOUR_USERNAME/PART1_KERNEL 8
```

Từ log, cộng số record của sáu part. Bộ dữ liệu hiện tại từng cho 1.843 record, nhưng phải dùng tổng thực tế nếu dataset nguồn thay đổi.

### 5.2. Train lần đầu

`run_kaggle_iterative_self.py` là launcher khuyến nghị vì nó có validation, text-conditioning sensitivity, CFG sweep, Whisper ASR, MP3 tốt nhất và plot báo cáo.

Thay sáu kernel ref và `--expected-records` trước khi chạy:

```powershell
uv run python scripts/run_kaggle_iterative_self.py `
  --kernel 1=YOUR_USERNAME/PART1_KERNEL `
  --kernel 2=YOUR_USERNAME/PART2_KERNEL `
  --kernel 3=YOUR_USERNAME/PART3_KERNEL `
  --kernel 4=YOUR_USERNAME/PART4_KERNEL `
  --kernel 5=YOUR_USERNAME/PART5_KERNEL `
  --kernel 6=YOUR_USERNAME/PART6_KERNEL `
  --expected-records 1843 `
  --epochs 20 `
  --batch-size 4 `
  --frames-per-chunk 384 `
  --dim 384 `
  --depth 6 `
  --heads 6 `
  --learning-rate 8e-5 `
  --style-dropout 0.8 `
  --text-dropout 0.1 `
  --text-contrastive-weight 0.08 `
  --text-contrastive-margin 0.03 `
  --text-contrastive-prob 0.75 `
  --text-sensitivity-weight 2 `
  --text-sensitivity-target 0.20 `
  --minimum-text-sensitivity 0.18 `
  --dataset-validation-max-records 128 `
  --validation-fraction 0.05 `
  --validation-max-records 96 `
  --early-stopping-patience 4 `
  --minimum-epochs 18 `
  --evaluation-records 6 `
  --guidance-scales "1,2,3,4" `
  --accelerator NvidiaTeslaT4 `
  --online-assets `
  --session-timeout-seconds 36000
```

Launcher sẽ:

- Xác nhận cả sáu preprocess kernels đã `COMPLETE`.
- Gộp đúng tổng record mà không copy tensor nhiều lần.
- Train và lưu checkpoint giữa epoch/mỗi 200 step.
- Chọn best checkpoint chỉ khi lyric sensitivity đạt ngưỡng.
- Thử CFG 1, 2, 3, 4 và đánh giá bằng Whisper.
- Tạo MP3, JSON, CSV và plot cho báo cáo.

State local nằm tại:

```text
outputs/kaggle_iterative_self/iterative-self-<timestamp>/state.json
```

Máy local có thể sleep hoặc mất mạng sau khi Kaggle đã nhận job; job chạy trên server Kaggle. Không xóa kernel hoặc revoke quyền dataset khi job còn chạy.

### 5.3. Resume thêm epoch

Nếu job đầu đã `COMPLETE` nhưng lời chưa rõ, dùng chính kernel đó làm checkpoint nguồn. Giữ nguyên sáu `--kernel` và toàn bộ hyperparameter, sau đó đổi/thêm:

```powershell
  --epochs 28 `
  --minimum-epochs 26 `
  --resume-kernel-ref YOUR_USERNAME/INITIAL_TRAIN_KERNEL `
  --source-dataset-ref YOUR_USERNAME/SOURCE_DATASET_FROM_STATE_JSON `
  --session-timeout-seconds 14400
```

Nếu checkpoint của một job timeout đã được tải và upload thành private Kaggle Dataset, dùng:

```powershell
  --resume-dataset-ref YOUR_USERNAME/RESUME_CHECKPOINT_DATASET
```

Không truyền đồng thời `--resume-kernel-ref` và `--resume-dataset-ref`.

## 6. Tải và đọc output

Sau khi kernel `COMPLETE`:

```powershell
uv run kaggle kernels output YOUR_USERNAME/TRAIN_KERNEL `
  -p outputs/downloaded_training
```

Các file quan trọng:

```text
self_all_parts.pt                 checkpoint mới nhất
self_all_parts.best.pt            checkpoint validation tốt nhất
training_report.json              loss, validation, sensitivity, epoch
quality_evaluation/quality_report.json
iteration_result.json             kết luận vòng train
best_generated.mp3                mẫu có ASR tốt nhất
report_plots/                     PNG, CSV và REPORT.md cho báo cáo
```

Chỉ dừng khi đồng thời đạt các điều kiện:

- `intelligibility_pass=true`.
- Whisper transcript không rỗng và bám lyric mục tiêu.
- `text_conditioning_sensitivity >= 0.18`.
- Nghe trực tiếp `best_generated.mp3` có từ tiếng Việt nhận ra được.

Loss thấp hoặc audio “giống nhạc” chưa đủ chứng minh lời rõ.

## 7. Train nhanh với một nguồn preprocess

Nếu chỉ muốn kiểm tra một processed kernel/dataset, cấu hình `.env`:

```env
KAGGLE_PROCESSED_KERNEL_REF=YOUR_USERNAME/COMPLETED_PREPROCESS_KERNEL
# Hoặc:
KAGGLE_PROCESSED_DATASET_REF=YOUR_USERNAME/PUBLISHED_PROCESSED_DATASET
```

Sau đó:

```powershell
uv run python scripts/run_kaggle_training.py --epochs 5 --batch-size 4
```

Luồng này phù hợp với smoke test, không thay thế full six-part training và không tạo bộ báo cáo đầy đủ như iterative launcher.

## 8. Chạy web

Cấu hình một kernel checkpoint đã `COMPLETE` và một preprocess kernel có
`records.jsonl` cùng backing stems trong `.env`:

```env
GENMUSIC_KAGGLE_CHECKPOINT_REF=YOUR_USERNAME/GENMUSIC_CHECKPOINT
GENMUSIC_KAGGLE_BACKING_REF=YOUR_USERNAME/GENMUSIC_PREPROCESS_PART
```

Sau đó chạy:

```powershell
uv run python server.py
```

Mở `http://127.0.0.1:8000`. Mỗi request web chỉ chạy inference từ checkpoint,
dùng pronunciation prior để giữ lời tiếng Việt rõ rồi mix vocal với backing
thật. Web không train lại model theo từng lần bấm.

## 9. Kiểm thử khi sửa code

```powershell
uv run python -m pytest -q
uv run python -m py_compile cli.py scripts/run_kaggle_iterative_self.py
git diff --check
```

Không đưa `.env`, dataset, checkpoint hoặc output lớn vào Git.

## 10. Lỗi thường gặp

### `Maximum batch GPU session count`

Đã có hai Kaggle GPU jobs đang giữ session. Chờ một job kết thúc rồi submit tiếp; không tạo source dataset mới liên tục.

### Kaggle báo hết quota nhưng job vẫn `RUNNING`

Quota có thể đã được reserve khi submit. Job hiện tại thường vẫn tiếp tục; kiểm tra trạng thái/log thay vì submit trùng.

### `Expected ... records, found ...`

`--expected-records` không khớp tổng `records.jsonl` của các preprocess kernel. Đọc log sáu part và sửa đúng tổng.

### `401`, `403` hoặc không mount được kernel/dataset

Ref đang private hoặc thuộc tài khoản khác. Public/chia sẻ tài nguyên hoặc preprocess lại bằng tài khoản mới.

### Tesla P100 báo PyTorch không hỗ trợ `sm_60`

Các launcher có CUDA smoke test và tự sửa Torch khi cần. Không để job rơi xuống CPU mà vẫn tiếp tục train.

### `scipy.signal` không có `hann`

Đây là warning khi ước lượng beat/BPM và fallback về 120 BPM; nó không phải nguyên nhân trực tiếp khiến lyric không rõ.

### Whisper transcript rỗng dù loss giảm

Model chưa hội tụ theo lyric hoặc conditioning đang bị bỏ qua. Kiểm tra sensitivity, CFG sweep và best checkpoint; resume từ checkpoint thay vì train lại từ đầu.

## 11. Cấu trúc project

```text
GenMusic/
  cli.py                              CLI chính
  server.py                           web API local
  src/data/                           preprocess, lyric/G2P/alignment
  src/models/                         MicroDiT và CFM
  src/training/                       dataset, train, resume, validation
  src/evaluation/                     metric và plot
  src/integrations/                   Kaggle auth/API
  scripts/run_kaggle_all_parts.py     preprocess toàn bộ partX
  scripts/run_kaggle_iterative_self.py full train/evaluate/resume
  scripts/evaluate_generation_quality.py
  scripts/create_kaggle_report_plots.py
  dataset/                            raw/local dataset nhỏ
  datasets/                           runtime dataset lớn, Git-ignored
  outputs/                            state/checkpoint/audio, Git-ignored
  docs/                               thiết kế, thí nghiệm và báo cáo
```

Tài liệu kỹ thuật chi tiết nằm trong `docs/`. README là nguồn bắt đầu cho người mới; các script `--help` là nguồn chính xác cho toàn bộ tùy chọn hiện hành.
