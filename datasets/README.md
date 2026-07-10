# Bố Cục Dataset

## Các Thư Mục

- `evaluation/`: benchmark JSONL và các case đánh giá an toàn.
- `training/`: record huấn luyện tự sinh hoặc có license. Dataset shard lớn chỉ lưu local và được Git bỏ qua.
- `trained_models/`: artifact text model bootstrap được commit, dùng khi chưa có model local mới hơn.
- `vn_music_stylebank/`: tài nguyên gọn về nhạc cụ, thể loại, âm nhạc và mẫu lời được pipeline sử dụng.
- `sources/`: chỉ chứa manifest nguồn. Manifest phải ghi URL, license và quyền phê duyệt rõ ràng trước khi crawler fetch.

## Dataset Cho JAM/DiffRhythm

Manifest tiền xử lý cần giữ provenance và trỏ tới audio cùng lyric có quyền sử dụng. Output chuẩn gồm:

```text
processed/jam/
  manifest.jsonl
  latents/<id>.pt
  styles/<id>.pt
  lrc/<id>.lrc
  preprocessing_report.json
```

Mỗi latent phải có shape `[frames, 64]` ở tốc độ 21.5 Hz và style vector shape `[512]` theo cấu hình đang dùng. VAE/style proxy chỉ dành cho smoke test. Dataset lớn nên đóng gói thành tar shards bằng `pack-jam-webdataset`; không commit audio/latent lớn vào Git.
