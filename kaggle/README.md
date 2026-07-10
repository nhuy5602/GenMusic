# Kaggle

Kaggle là môi trường chạy chính cho DiffRhythm vì upstream yêu cầu PyTorch, torchaudio, MuQ, phonemizer và GPU.

Job genmusic-vn.cli generate sẽ:
1. tạo private dataset chứa lyric LRC và request;
2. tạo kernel GPU;
3. clone ASLP-lab/DiffRhythm;
4. cài requirements.txt upstream;
5. chạy infer/infer.py với checkpoint ASLP-lab/DiffRhythm-1_2;
6. tải WAV/MP3 về outputs/<run_id>/kaggle_job/downloaded_output.

Train random smoke:

```powershell
python -m genmusic_vn.cli make-random-diffrhythm-dataset --out datasets/random_diffrhythm --count 4 --max-frames 64
python -m genmusic_vn.cli train-diffrhythm --dataset datasets/random_diffrhythm --epochs 1 --batch-size 1
```

Nếu random .pt không tạo được tại Windows vì thiếu torch, tạo dataset ở một kernel Kaggle có torch sẵn rồi chạy official train/train.py.
