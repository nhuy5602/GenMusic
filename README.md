# GenMusic VN: Background Music & Vietnamese Vocal Generator

GenMusic VN là dự án phát triển mô hình khuếch tán điều kiện (conditional diffusion model) tự huấn luyện nhằm sinh lời hát tiếng Việt và nhạc nền.

Các tài liệu kỹ thuật chi tiết nằm trong thư mục `docs/`:
- [Kiến trúc hệ thống (System Architecture)](docs/architecture.md)
- [Mô hình máy học (Machine Learning Models)](docs/model.md)
- [Quy trình huấn luyện & Cải tiến (Training Pipelines)](docs/training.md)

---

## 📂 Tổng quan cấu trúc thư mục Dự án

```text
GenMusic/
├── src/                # Gói mã nguồn chính (xử lý dữ liệu, mô hình, huấn luyện)
│   ├── data/           # Tiền xử lý âm thanh, Whisper ASR, tách Demucs, G2P tiếng Việt
│   ├── models/         # Kiến trúc mạng khuếch tán (ResidualDenoiser, TextConditioner)
│   ├── training/       # Quy trình Dataset, DataLoader và Trainer
│   ├── evaluation/     # Bộ chỉ số đánh giá khách quan phổ âm thanh
│   └── integrations/   # Tích hợp dịch vụ đám mây Kaggle API
├── scripts/            # Các kịch bản chạy tự động (huấn luyện Kaggle, tiền xử lý mây)
├── web/                # Giao diện Web Client demo (HTML, CSS, JS)
├── docs/               # Tài liệu đặc tả hệ thống
├── dataset/            # Thư mục chứa dữ liệu thô (được bỏ qua khi đẩy lên git)
├── outputs/            # Checkpoints mô hình và tệp âm thanh đầu ra
├── cli.py              # Dòng lệnh điều khiển chính của dự án
└── server.py           # Web server API Backend
```

---

## 🛠️ Cài đặt & Thiết lập môi trường

### 1. Cài đặt các thư viện cần thiết

* **Sử dụng công cụ `uv` (Khuyên dùng - nhanh và bảo mật):**
  ```powershell
  uv sync
  ```

* **Sử dụng Pip tiêu chuẩn:**
  ```powershell
  pip install -r requirements.txt
  ```

### 2. Thiết lập Biến môi trường (.env)
Tạo tệp `.env` ở thư mục gốc của dự án dựa trên mẫu `.env.example`:
```env
# Môi trường chạy local
RAW_AUDIO_INPUT_DIR=dataset/vietnamese_songs
PROCESSED_DATASET_DIR=dataset/diff_rhythm_dataset
MODEL_CHECKPOINT_PATH=outputs/my_trained_model.pt
GENMUSIC_OUTPUT_DIR=outputs

# Tài khoản Kaggle (để đẩy tác vụ huấn luyện lên GPU mây)
KAGGLE_USERNAME=ten_tai_khoan_kaggle
KAGGLE_KEY=api_key_cua_ban
KAGGLE_RAW_DATASET_REF=sonlest/vietnamese-music-dataset-version3-part6
KAGGLE_PROCESSED_DATASET_REF=ten_tai_khoan_kaggle/vietnamese-music-processed-dataset
```

---

## 🚀 Hướng dẫn Sử dụng các Kịch bản (Scripts)

Toàn bộ các quy trình đã được đóng gói thành các tệp lệnh chạy tự động trong thư mục `scripts/`:

### 1. Chạy thử nghiệm toàn bộ luồng ở máy Local
Chạy chuỗi các bước: *Quét nhạc thô ➔ Tiền xử lý ➔ Huấn luyện mô hình mini ➔ Sinh thử nhạc ➔ Đánh giá kết quả*:
```powershell
uv run python scripts/run_pipeline.py
```

### 2. Chạy thử nghiệm Huấn luyện trên Kaggle GPU (1 file nhạc)
Tự động đẩy code lên Kaggle, trích xuất 1 file nhạc từ bộ dữ liệu của bạn, tiền xử lý và huấn luyện thử trên GPU T4 của Kaggle, sau đó tải model checkpoint `.pt` về máy local:
```powershell
uv run python scripts/run_kaggle_training.py
```

### 3. Tiền xử lý HÀNG LOẠT toàn bộ dataset trên mây Kaggle
Tách âm thanh (Demucs) + Chép lời (Whisper) + Tính F0 (pYIN) cho toàn bộ 250 file nhạc thô trên mây Kaggle và tự động tải lên tài khoản Kaggle của bạn làm một Dataset sạch mới:
```powershell
uv run python scripts/run_kaggle_preprocess_all.py
```

---

## 🎹 Dòng lệnh CLI nâng cao (`cli.py`)

### 1. Sinh nhạc thủ công (Local Generation)
Bạn có thể chọn giữa bộ giải mã toán học cổ điển (`istft` - siêu nhẹ) hoặc bộ giải mã chất lượng cao **Vocos** (`vocos` - khuyên dùng):
```powershell
# Sử dụng Vocos Vocoder cho chất lượng âm thanh hát tự nhiên hơn:
uv run python cli.py generate-local --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi." --duration 8.0 --vocoder vocos --out outputs/my_song
```

### 2. Tiền xử lý dữ liệu thủ công
```powershell
uv run python cli.py preprocess-raw --input dataset/vietnamese_songs --output dataset/diff_rhythm_dataset --whisper-model small
```

### 3. Đánh giá chất lượng tệp nhạc sinh ra
```powershell
uv run python cli.py evaluate-self --generated outputs/my_song/final.wav --out outputs/evaluation_report
```

---

## 🖥️ Giao diện Web Demo trực quan

Khởi động Web server API:
```powershell
uv run python server.py
```
Mở trình duyệt web của bạn và truy cập địa chỉ: `http://127.0.0.1:8000` để bắt đầu nhập văn bản tiếng Việt và nghe thử nhạc trực quan.

---

## 🧪 Chạy Kiểm thử tự động (Unit Tests)

Đảm bảo tính ổn định của mã nguồn bằng cách chạy kiểm thử:
```powershell
uv run python -m unittest discover -s tests -v
```
