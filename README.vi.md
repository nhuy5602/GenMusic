# GenMusic VN

> Bản Việt hóa của `README.md`. Các tên lệnh, API, model và đường dẫn được giữ
> nguyên. Khi có khác biệt, tài liệu gốc là nguồn chuẩn.

GenMusic VN là dự án nghiên cứu sinh nhạc từ lời tiếng Việt, với hai luồng chạy
được phân biệt rõ ràng:

- `native-waveform-v80`: luồng demo mặc định dài 16 giây. Luồng này truy xuất
  và nối các đơn vị giọng hát tiếng Việt thật, sau đó trộn với một backing track
  lấy từ bài khác.
- `cfm-research`: luồng Conditional Flow Matching/MicroDiT do nhóm tự xây dựng,
  dùng cho nghiên cứu model và các phép ablation.

Hai luồng được gắn nhãn riêng có chủ đích. Kết quả V80 không được báo cáo như
một kết quả diffusion.

## Clone mới → tài khoản Kaggle mới

Repository không chứa owner/slug Kaggle của người phát triển, credential,
checkpoint cục bộ, PDF hay tài liệu bảo vệ. Một bản clone mới sẽ xây corpus
native trong chính tài khoản Kaggle do người dùng cung cấp.

### Điều kiện cần

- Git
- [uv](https://docs.astral.sh/uv/)
- Tài khoản Kaggle có quyền dùng GPU và API token
- Kernel Kaggle có Internet để tải dataset Hugging Face và model Demucs đã
  ghi trong tài liệu

### 1. Cài đặt từ lock file

```powershell
git clone <repository-url> GenMusic
cd GenMusic
uv sync --locked
Copy-Item .env.example .env
```

Chỉ điền credential của chính bạn vào `.env`:

```env
KAGGLE_USERNAME=your_account
KAGGLE_API_TOKEN=your_token
```

`KAGGLE_KEY` vẫn được hỗ trợ cho credential Kaggle kiểu cũ. `.env` đã được
ignore và tuyệt đối không được commit.

### 2. Khởi tạo corpus V80 trong tài khoản đó

```powershell
uv run python scripts/run_kaggle_native_waveform_dataset.py --wait
```

Lệnh này sẽ:

1. đóng gói bản clone hiện tại thành source asset private không chứa token;
2. gửi một GPU kernel private dưới username Kaggle hiện tại;
3. materialize corpus vocal/backing 2.048 record, song-disjoint và có timestamp
   chính xác từ dataset Hugging Face công khai đã cấu hình;
4. lưu ref `owner/kernel` vào `.genmusic/kaggle.json` sau khi gửi job.

`.genmusic/`, dataset, output, cache và ghi chú cục bộ đều bị Git bỏ qua.
Không cần ref từ tài khoản của người phát triển ban đầu. Bước bootstrap tốn
nhiều tài nguyên nhưng chỉ chạy một lần cho mỗi tài khoản; các lần sinh nhạc
sau tái sử dụng output của kernel đã hoàn tất.

Nếu không muốn chờ trong terminal, bỏ `--wait`, theo dõi URL Kaggle được in ra
và chỉ sinh nhạc sau khi corpus kernel ở trạng thái `COMPLETE`.

### 3. Sinh một ứng viên MP3/WAV

Dùng 8–32 từ tiếng Việt. V80 giữ thời lượng đã kiểm chứng là 16 giây.

```powershell
uv run python cli.py generate `
  --text "Một chiều mưa tôi nhớ về những con phố cũ và lời hẹn trong tim" `
  --wait
```

State của lần sinh và audio tải về được ghi dưới `outputs/`. Việc xếp lịch
Kaggle và host nguồn có thể tạo ra khác biệt nhỏ về thời gian chạy, nhưng danh
sách shard, quy tắc chọn corpus, chính sách thời lượng và chính sách tổng hợp
V80 đều nằm trong repository, không phụ thuộc tài khoản cá nhân.

### 4. Chạy web app

```powershell
uv run python server.py
```

Mở [http://127.0.0.1:8000](http://127.0.0.1:8000). Web app dùng cùng file
`.genmusic/kaggle.json` đã ignore và cùng submitter V80 như CLI.

## Luồng nghiên cứu CFM

Vòng lặp nghiên cứu cục bộ nhỏ nhất:

```powershell
uv run python cli.py preprocess-raw `
  --input dataset/vietnamese_songs `
  --output dataset/diff_rhythm_dataset `
  --whisper-model tiny

uv run python cli.py train-self `
  --dataset dataset/diff_rhythm_dataset `
  --checkpoint outputs/cfm_student.pt `
  --epochs 30 --batch-size 4 --dim 256 --depth 4 --heads 4

uv run python cli.py generate-local `
  --text "Đêm nay Hà Nội ngập tràn tiếng mưa rơi" `
  --checkpoint outputs/cfm_student.pt `
  --out outputs/cfm_demo
```

Để đọc hợp đồng preprocessing, thí nghiệm latent codec, distillation và các
launcher nghiên cứu Kaggle, xem:

- [Kiến trúc](docs/architecture.vi.md)
- [Chuẩn bị dữ liệu](docs/data_preparation.vi.md)
- [Hướng dẫn sử dụng](docs/usage.vi.md)
- [Danh mục script](scripts/README.vi.md)

## Cấu trúc repository

```text
src/            code model, huấn luyện, audio, dữ liệu và tích hợp Kaggle
scripts/        entry point cục bộ/Kaggle có thể tái lập
tests/          test đơn vị và test hợp đồng tích hợp
web/            giao diện trình duyệt
docs/           tài liệu kỹ thuật
cli.py          giao diện dòng lệnh
server.py       HTTP server gọn nhẹ
```

Ghi chú báo cáo/PDF/bảo vệ không thuộc cây này. Bản sao cục bộ có thể được giữ
dưới `local_notes/` đã ignore mà không lọt vào commit hay source asset Kaggle.

## Kiểm tra trước khi nộp

```powershell
uv run python scripts/check_portability.py
uv run --with pytest python -m pytest -q
```

Portability audit kiểm tra các file Git nhìn thấy để phát hiện đường dẫn máy
cục bộ, Kaggle token bị nhúng và artifact chỉ dành cho báo cáo/bảo vệ. Test
cũng xác minh ref corpus native được đọc từ biến môi trường/cấu hình cục bộ,
không phải một giá trị mặc định cá nhân đã commit.
