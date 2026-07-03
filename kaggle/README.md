# Quy Trình Kaggle

Project hiện dùng một hướng sinh nhạc duy nhất: MusicGen chạy trên Kaggle GPU.

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

Tạo `.env` hoặc `.env.local` ở thư mục gốc:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Code sẽ tự đọc token và truyền vào Kaggle CLI qua environment variables. Không commit file token lên GitHub.

## Lệnh Tự Động

```powershell
python -m genmusic_vn.cli generate --text "Một đoạn văn tiếng Việt..." --duration 30 --wait
```

Nếu chưa có token, chương trình vẫn stage đầy đủ file trong:

```text
outputs/<run_id>/kaggle_job/
```

Sau khi thêm token, có thể chạy lại các lệnh trong `run_commands.ps1`.

