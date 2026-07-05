# Quy Trình Kaggle

Project hiện dùng một hướng sinh nhạc duy nhất: MusicGen chạy trên Kaggle GPU.

## Test Nhanh Qua Web Local Trên macOS

Từ thư mục gốc project:

```bash
cd /Users/user/IdeaProjects/GenMusic
python3 -m unittest discover -s tests -v
python3 -m genmusic_vn.server --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

Nhập text tiếng Việt và bấm `Generate MP3`. Nút này sẽ submit Kaggle job thật, nên cần `.env` có `KAGGLE_USERNAME` và `KAGGLE_API_TOKEN`.

Luồng demo tự động do `genmusic_vn.kaggle_auto` xử lý:

1. Local nhận văn bản tiếng Việt gốc.
2. Local tạo `request.json`.
3. Local đóng gói source code và dataset `datasets/vn_music_stylebank` vào `genmusic_vn_source.zip`.
4. Local dùng Kaggle API token để upload `request.json` và `genmusic_vn_source.zip` thành Kaggle Dataset riêng tư.
5. Local dùng Kaggle API token để push Kaggle Kernel riêng tư có bật GPU.
6. Kaggle giải nén source, đọc stylebank, chạy pipeline AI và MusicGen.
7. Kaggle chuyển WAV sang MP3.
8. Local tải MP3 từ kernel output về máy.

## Kết Nối Bằng Kaggle API Token

Trên macOS dùng `python3`:

```bash
python3 -m pip install --user -U kaggle
```

Tạo `.env` hoặc `.env.local` ở thư mục gốc:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_token
```

Code sẽ tự đọc token mới từ `.env`, `.env.local`, environment variables hoặc `~/.kaggle/access_token`, rồi truyền vào Kaggle CLI. Không commit file token lên GitHub.

## Lệnh Tự Động

```bash
python3 -m genmusic_vn.cli generate --text "Một đoạn văn tiếng Việt..." --duration 30 --wait
```

Nếu chưa có token, chương trình vẫn stage đầy đủ file trong:

```text
outputs/<run_id>/kaggle_job/
```

Sau khi thêm token, có thể chạy lại các lệnh đã được tạo trong `outputs/<run_id>/kaggle_job/`.
Trên macOS/Linux, dùng `run_commands.sh`:

```bash
bash outputs/<run_id>/kaggle_job/run_commands.sh
```

Trên Windows, dùng `run_commands.ps1`.
