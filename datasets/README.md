# Dataset

Dataset hiện tại phục vụ trực tiếp DiffRhythm upstream.

## Dataset random smoke

```
random_diffrhythm/
  train.scp
  latent/<id>.pt
  lrc/<id>.pt
  style/<id>.pt
  diffrhythm-random.json
  random_dataset_report.json
```

train.scp có format:

```
utt_id|lrc_path|latent_path|style_path
```

Latent random dùng shape [1, 64, T], style [1, 512], LRC là dictionary có time và lrc, đúng contract của dataset/dataset.py upstream.

Không commit audio, checkpoint, tensor lớn hoặc output inference vào Git.
