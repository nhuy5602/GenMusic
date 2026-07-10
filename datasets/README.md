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

Mỗi tensor mel có shape `[64, frames]`. Mỗi record chứa `text`, `style`, `mel_path` và `frames`. Dataset random chỉ xác nhận contract và train loop; không đại diện cho chất lượng âm nhạc.

Không commit audio, checkpoint hoặc tensor lớn vào Git.
