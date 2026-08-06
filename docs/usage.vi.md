# Hướng dẫn sử dụng

> Bản Việt hóa của `docs/usage.md`. Các command, option, tên model và tên file
> được giữ nguyên. Khi có khác biệt, tài liệu gốc là nguồn chuẩn.

Đây là hướng dẫn thực hành cho preprocessing, huấn luyện, sinh nhạc, đánh giá
và tự động hóa Kaggle. Đọc `docs/architecture.md` trước nếu cần hiểu *tại sao*;
file này tập trung vào *cách làm*. Chạy `uv sync` một lần trước các bước dưới.

**Mọi tác vụ nặng (Demucs + Whisper, huấn luyện model, distillation) nên chạy
trên Kaggle**, không phải máy cục bộ. Chạy local bằng CPU phù hợp với smoke
test, nhưng train thật cần GPU. Quota GPU Kaggle có giới hạn; kernel bị treo
vẫn âm thầm tiêu quota, vì vậy hãy đọc mục ghi chú hạ tầng trước khi mở job dài.

## 1. Thiết lập một lần

Sao chép `.env.example` thành `.env`, rồi điền `KAGGLE_USERNAME` và
`KAGGLE_API_TOKEN` (hoặc `KAGGLE_KEY` kiểu cũ). Repository không commit
owner/slug của tài khoản phát triển.

Với demo V80 mặc định, bootstrap corpus thô một lần trong tài khoản mới:

```powershell
uv run python scripts/run_kaggle_native_waveform_dataset.py --wait
```

Lệnh ghi ref corpus đã gửi vào `.genmusic/kaggle.json` đã ignore; CLI và web
tự động đọc ref này.

## 2. Preprocess bài hát thô thành dataset huấn luyện

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small --max-files 100 --keep-separated-count 10
```

Lệnh tách stem vocal/backing bằng Demucs, nhận dạng lyric bằng Whisper, tính
Audio Style Anchor bằng MuQ-MuLan và ghi tensor mel đúng định dạng Vocos. Output
là `dataset/diff_rhythm_dataset/{config.json, records.jsonl, mels/}`. Nếu một
số file lỗi, lệnh trả `completed_with_warnings` và exit code khác 0; phải xem
danh sách lỗi trước khi train. Mặc định theo báo cáo cũng fail nếu model
MuQ-MuLan thật không tạo được anchor 512 chiều hữu hạn. `--allow-zero-style`
chỉ là lối thoát debug, không dùng để tái lập báo cáo.

Kiểm tra nhanh chỉ bằng phần mềm, không tải audio:

```powershell
uv run python cli.py make-random-dataset --out dataset/random_self_diffusion_training --count 16 --frames 128 --target-gb 1.0
uv run python cli.py validate-dataset --dataset dataset/random_self_diffusion_training
```

Trên Kaggle:

```powershell
uv run python scripts/run_kaggle_preprocess_all.py --max-files 40
```

## 3. Huấn luyện student

Chọn một không gian đặc trưng: mel thô mặc định, hoặc native latent ở mục 3b.
Backbone `MicroDiT` giống nhau. Huấn luyện chỉ từ ground truth:

```powershell
uv run python cli.py train-self --dataset dataset/diff_rhythm_dataset --checkpoint outputs/my_model.pt --epochs 60 --dim 256 --depth 4 --heads 4
```

Nên đặt `--epochs` là một mức trần lớn và để early stopping quyết định thời
điểm dừng, thay vì tự dò số epoch.

### 3a. Distillation từ teacher DiffRhythm2 thật

```powershell
$env:PYTHONPATH = "C:\path\to\DiffRhythm2"   # clone github.com/ASLP-lab/DiffRhythm2 trước
uv run python cli.py train-distill --dataset dataset/diff_rhythm_dataset --student-checkpoint outputs/distilled_student.pt --epochs 60 --dim 256 --depth 4 --heads 4 --alpha-feature 0.8
```

Cần repository DiffRhythm2 trong `PYTHONPATH`, các dependency Python và
`espeak-ng` cho lyric tokenizer. Nếu thiếu clone hoặc Internet,
`train-distill` **lỗi ngay** thay vì âm thầm chuyển thành ground-truth-only;
dùng `train-self` cho trường hợp đó. `--alpha-feature≈0.8` là mặc định đã cho
kết quả tốt qua đo lường, không phải `0.5`.

Trên Kaggle:

```powershell
uv run python scripts/run_kaggle_training.py --epochs 5 --batch-size 4
uv run python scripts/run_kaggle_distill.py
```

Hai script tự thiết lập GPU, clone DiffRhythm2 và cài dependency. Trước đó cần
credential Kaggle trong `.env` và ref dataset đã xử lý mà tài khoản hiện tại
có quyền truy cập (`KAGGLE_PROCESSED_KERNEL_REF` ưu tiên, hoặc
`KAGGLE_PROCESSED_DATASET_REF`).

### 3b. Native latent space

Thực hiện trên dataset mel có sẵn. Cùng backbone `MicroDiT`, không cần cờ kiến
trúc riêng:

```powershell
# 1. Pretrain encoder nhỏ với decoder BigVGAN thật đã đóng băng
uv run python cli.py train-latent-encoder --dataset dataset/diff_rhythm_dataset --checkpoint outputs/latent_encoder.pt --epochs 40 --batch-size 4 --kl-weight 0.15 --num-workers 4

# 2. Sanity-check encoder trước khi dùng downstream
uv run python scripts/check_latent_encoder_quality.py --encoder-checkpoint outputs/latent_encoder.pt --dataset dataset/diff_rhythm_dataset

# 3. Chuyển dataset mel sang latent 64 chiều/5 Hz
uv run python cli.py precompute-latent-dataset --source-dataset dataset/diff_rhythm_dataset --encoder-checkpoint outputs/latent_encoder.pt --out dataset/latent_dataset

# 4. Huấn luyện CFM trong latent space
uv run python cli.py train-self --dataset dataset/latent_dataset --checkpoint outputs/latent_cfm_model.pt --lambda-vocal 0 --epochs 300 --batch-size 8

# 5. Sinh nhạc; latent_mode tự decode bằng BigVGAN thật, không dùng Vocos
uv run python cli.py generate-local --text "..." --style "..." --checkpoint outputs/latent_cfm_model.pt --out outputs/latent_demo
```

Bước 1, 2 và 5 cần DiffRhythm2 trên `PYTHONPATH` vì `bigvgan` không phải
package pip. Luôn chạy bước 2 sau khi train lại encoder. Failure mode từng gặp
là encoder collapse: loss có thể trông bình thường nhưng
`pitch_std_semitones` gần 0 khi decode trực tiếp latent ground truth. Nguyên
nhân là thiếu probabilistic bottleneck; đã sửa bằng `mu`/`logvar`,
reparameterization và KL loss qua `--kl-weight`. Tiếng lạo xạo dù pitch metric
khỏe có thể đến từ overlap trong `decode_audio` theo chunk của BigVGAN, là lỗi
khác với encoder collapse. `train-distill` cũng chạy trực tiếp trên
`latent_dataset`, không cần bridge channel/tốc độ frame.

Trên Kaggle, chạy theo thứ tự:

```powershell
uv run python scripts/run_kaggle_latent_encoder.py --epochs 40 --batch-size 4 --kl-weight 0.15
uv run python scripts/run_kaggle_check_latent_encoder_quality.py --encoder-checkpoint outputs/.../latent_encoder.pt
uv run python scripts/run_kaggle_latent_pipeline.py --encoder-checkpoint outputs/.../latent_encoder.pt --cfm-epochs 300
```

Nếu Kaggle cắt ngang CFM vì giới hạn thời gian, dùng
`scripts/run_kaggle_latent_resume.py` với số epoch nhỏ, có giới hạn cho mỗi
lượt thay vì train lại từ đầu.

## 4. Sinh nhạc

```powershell
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --vocoder vocos --checkpoint outputs/my_dit_model.pt --out outputs/my_song
```

`vocos` là mặc định; `griffinlim` là phương án ước lượng pha lặp khi Vocos
không dùng được. Cả hai yêu cầu mel đúng quy ước native của Vocos.

Không có `--reference-dataset`, generation dùng vector style từ pooled text
thay cho MuQ-MuLan anchor thật. Để conditioning giống lúc train:

```powershell
uv run python cli.py generate-local --text "..." --duration 8.0 --checkpoint outputs/my_dit_model.pt --reference-dataset dataset/diff_rhythm_dataset --reference-id <record_id> --out outputs/my_song
```

Nếu bỏ `--reference-id`, record đầu tiên được dùng.

## 5. Đánh giá

```powershell
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```

Hoặc script metric đầy đủ hơn:

```powershell
uv run python scripts/evaluate_generation_quality.py outputs/my_dit_model.pt
uv run python scripts/run_kaggle_evaluate.py
```

Các metric gồm spectral flatness, voiced ratio và pitch-std; chúng là sanity
check, không thay thế việc nghe.

## 6. Script thí nghiệm Kaggle hợp nhất

Thay vì dùng năm GPU session cho các bước riêng, chạy toàn bộ chuỗi trong một
kernel:

```powershell
uv run python scripts/run_kaggle_full_experiment.py --max-files 40 --whisper-model tiny --baseline-epochs 60 --distill-epochs 30
```

Script chạy preprocess → sanity check vocoder → baseline → distillation →
generate → thống kê. Theo dõi URL in ra hoặc dùng
`uv run python -m kaggle kernels status <kernel_ref>`. Khi kết thúc, tải bằng:

```powershell
uv run python -m kaggle kernels output <kernel_ref> -p outputs/.../downloaded -o
```

Sau khi tải, xóa source code, clone DiffRhythm2 và teacher checkpoint được echo
lại vì chúng tái lập được và chỉ tốn đĩa; giữ dataset đã xử lý, checkpoint và
audio sinh.

Ablation `alpha_feature` và kích thước kiến trúc:

```powershell
uv run python scripts/run_kaggle_experiment_matrix.py --max-files 40 --whisper-model tiny --epochs 60
```

Huấn luyện trên corpus nhiều phần:

```powershell
uv run python scripts/run_kaggle_multi_part_training.py
```

Smoke test cục bộ trước khi dùng Kaggle:

```powershell
uv run python scripts/run_full_experiment.py --raw-dataset dataset/vietnamese_songs --output-root outputs/local_smoke_test --max-files 2 --whisper-model tiny --baseline-epochs 5 --distill-epochs 2 --batch-size 2
```

Trong `outputs/local_smoke_test/summary.json`, mong đợi
`preprocess.status: "completed"`, `vocoder_roundtrip.logmel_corr > 0.95`; nếu
không đặt `PYTHONPATH`, distillation phải báo `failed` vì không import được
DiffRhythm2. Đó là hành vi đúng.

## Ghi chú hạ tầng Kaggle

- `kaggle kernels output` chỉ trả file sau khi kernel kết thúc. Dùng
  `scripts/check_kernel_progress.py` để đọc SSE log-stream và xác nhận
  epoch/step thực sự tăng.
- Nếu launcher dùng `subprocess.run(..., capture_output=True)`, Kaggle UI có
  thể im lặng suốt subprocess. Các launcher latent hiện dùng `Popen` để stream
  log; launcher mới cũng nên làm vậy.
- Chạy job dài theo các chặng epoch nhỏ, hữu hạn. Cách này chứng minh job khỏe
  hoặc fail nhanh, tránh đốt gần hết quota trong một job không quan sát được.
- Không có lệnh `kaggle kernels stop`; khi kernel treo thật, dùng
  `uv run python -m kaggle kernels delete <kernel_ref>`.
- Kaggle đôi lúc cấp P100 `sm_60` không tương thích PyTorch cài sẵn. Các script
  `run_kaggle_latent_*.py` probe trường hợp này và cài lại bộ
  `torch`/`torchaudio`/`torchvision` tương thích.

## Gửi một job sinh nhạc Kaggle (`cli.py generate`)

Luồng sản phẩm mặc định là V80, giống web app:

```powershell
uv run python cli.py generate --text "Một ngày mới bắt đầu trong nắng mai dịu dàng" --duration 16 --wait
```

Muốn chạy luồng nghiên cứu CFM phải chọn rõ:

```powershell
uv run python cli.py generate --backend cfm-research --text "..." --duration 12 --wait
```

`--model` và `--dataset-ref` chỉ áp dụng cho `cfm-research`. Việc tách backend
ngăn demo V80 bị gắn nhãn sai thành kết quả MicroDiT.

## Web demo tương tác

```powershell
uv run python server.py
```

Mở `http://127.0.0.1:8000` để nhập prompt tiếng Việt và nghe V80. Trước demo,
kiểm tra `http://127.0.0.1:8000/api/health`; backend phải là
`native-waveform-v80`.

## Unit test

```powershell
uv run --with pytest python -m pytest -q
```
