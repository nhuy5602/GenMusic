# Kaggle

Kaggle là môi trường GPU để chạy trainer và inference của model conditional diffusion tự code.

Job `genmusic-vn.cli generate` sẽ:

1. Tạo dataset private gồm request, LRC và source project.
2. Tạo kernel GPU.
3. Cài dependency Python từ package index.
4. Tạo dataset random để smoke train.
5. Train checkpoint self-diffusion.
6. Sinh WAV/MP3 từ lyric và style prompt.

Stage job:

```powershell
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --no-submit
```

Submit và chờ:

```powershell
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --wait
```

Kaggle cần `KAGGLE_USERNAME`, `KAGGLE_KEY` và CLI tương ứng. Dataset random cần được thay bằng dữ liệu audio/mel có quyền sử dụng để model học được chất lượng thực.
