# Thư mục dataset khi chạy

> Bản Việt hóa của `dataset/README.md`. Các lệnh và đường dẫn được giữ nguyên.

Thư mục này dành cho audio thô cục bộ và output sinh ra trong quá trình
preprocessing. Các file lớn đã được Git bỏ qua.

Cấu trúc dataset đã xử lý dự kiến:

```text
diff_rhythm_dataset/
  records.jsonl
  config.json
  mels/<id>_backing.pt
  mels/<id>_vocal.pt
  mels/<id>_style.pt   # embedding phong cách MuQ-MuLan, xem src/data/README.md
```

Tạo một dataset tổng hợp để smoke test bằng:

```powershell
uv run python cli.py make-random-dataset --out datasets/random_self_diffusion_1gb --target-gb 1
```

Preprocess file WAV/MP3 thật bằng:

```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model base
```

Dữ liệu tổng hợp chỉ kiểm tra hợp đồng dataset và vòng lặp huấn luyện. Nó không
đại diện cho chất lượng giọng hát thật.
