# Dataset

Dataset runtime của model tự code dùng manifest `records.jsonl`:

```text
random_self_diffusion/
  records.jsonl
  mels/<id>.pt
  config.json
  dataset_report.json
  validation_report.json
```

Dataset smoke test lớn:

```text
random_self_diffusion_training/
  records.jsonl
  mels/<id>.pt
  config.json
  dataset_report.json
  validation_report.json
  kaggle_upload_report.json
```

Tạo và upload:

```powershell
python -m genmusic_vn.cli make-and-upload-dataset --out datasets/random_self_diffusion_training --target-gb 1
```

Dataset ref mặc định là `<KAGGLE_USERNAME>/genmusic-vn-self-diffusion-training`. Có thể đổi bằng `--dataset-ref owner/slug` hoặc `GENMUSIC_KAGGLE_DATASET_REF`. Pipeline Kaggle chỉ dùng dataset ref này; nếu chưa tồn tại, job trả lỗi.

Mỗi tensor mel có shape `[64, frames]`. Mỗi record chứa `text`, `style`, `mel_path` và `frames`. Với dataset 1 GB, file `.pt` còn có `augmentation_cache` để tăng kích thước mà không làm thay đổi contract loader. Dataset random chỉ xác nhận contract và train loop; không đại diện cho chất lượng âm nhạc.

Không commit audio, checkpoint hoặc tensor lớn vào Git.
