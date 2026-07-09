# Quy Trình Kaggle

Project có hai phần chạy trên Kaggle:

- train lại `genmusic_text_model.json` từ dataset tiếng Việt tự sinh
- sinh MP3 bằng custom composer + F5-TTS/MMS-TTS

Luồng demo tự động do `genmusic_vn.kaggle_auto` xử lý:

1. Local nhận văn bản tiếng Việt gốc.
2. Local load trained text model để phân tích emotion/style nếu có artifact.
3. Local tạo `request.json`.
4. Local đóng gói source code và dataset `datasets/vn_music_stylebank` vào `genmusic_vn_source.zip`.
5. Local dùng Kaggle API token để upload `request.json` và `genmusic_vn_source.zip` thành Kaggle Dataset riêng tư.
6. Local dùng Kaggle API token để push Kaggle Kernel riêng tư có bật GPU.
7. Kaggle giải nén source, đọc stylebank, chạy pipeline AI, custom composer và TTS.
8. Kaggle chuyển WAV sang MP3.
9. Local tải MP3 từ kernel output về máy.

## Kết Nối Bằng Kaggle API Token

Tạo `.env` hoặc `.env.local` ở thư mục gốc:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Code sẽ tự đọc token và truyền vào Kaggle CLI qua environment variables. Không commit file token lên GitHub.

## Lệnh Tự Động

Train text model trên Kaggle:

```powershell
python -m genmusic_vn.cli train-text-model --samples 1200 --seed 5602 --wait --model-out models/current/genmusic_text_model.json
```

```powershell
python -m genmusic_vn.cli generate --text "Một đoạn văn tiếng Việt..." --duration 30 --wait
```

Nếu chưa có token, chương trình vẫn stage đầy đủ file trong:

```text
outputs/<run_id>/kaggle_job/
```

Sau khi thêm token, có thể chạy lại các lệnh trong `run_commands.ps1`.
