# Kaggle

Kaggle là môi trường GPU để chạy trainer và inference của model conditional diffusion tự code.

Job `genmusic-vn.cli generate` sẽ:

1. Tạo dataset private gồm request, LRC và source project.
2. Tạo kernel GPU.
3. Cài dependency Python từ package index.
4. Gắn dataset training cố định từ `GENMUSIC_KAGGLE_DATASET_REF` hoặc `<KAGGLE_USERNAME>/genmusic-vn-self-diffusion-training`.
5. Train checkpoint self-diffusion trên dataset đó.
6. Sinh WAV/MP3 từ lyric và style prompt.

Chuẩn bị dataset bằng một lệnh:

```powershell
python -m genmusic_vn.cli make-and-upload-dataset --out datasets/random_self_diffusion_training --target-gb 1
```

Đổi dung lượng bằng `--target-gb 5`, hoặc đổi ref bằng `--dataset-ref owner/slug`. Pipeline sẽ dừng với lỗi rõ ràng nếu dataset ref cố định không tồn tại.

Stage job:

```powershell
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --no-submit
```

Submit và chờ:

```powershell
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --wait
```

Kaggle cần `KAGGLE_USERNAME`, `KAGGLE_KEY` và CLI tương ứng. Dataset synthetic chỉ kiểm tra pipeline; muốn model học chất lượng thực, thay nội dung dataset bằng audio/mel và lyric có quyền sử dụng nhưng giữ nguyên dataset ref.

Upload dataset đã tạo sẵn:

```powershell
python -m genmusic_vn.cli upload-dataset --dataset datasets/random_self_diffusion_training --dataset-ref owner/genmusic-vn-self-diffusion-training
```

Web app hiển thị trực tiếp link dataset training cố định và kernel của job sau khi stage/submit thành công.
