# GenMusic VN

GenMusic VN là model sinh nhạc từ văn bản do project tự code. Backend là conditional diffusion: text được mã hóa bằng character embedding, denoiser Conv1D dự đoán noise trên mel-spectrogram, sau đó mel được đổi thành WAV/MP3.

## Luồng chính

```text
Lyric + style
  -> chuẩn hóa tiếng Việt và timestamp LRC
  -> text conditioner
  -> conditional diffusion denoise
  -> mel-to-waveform Griffin-Lim
  -> WAV/MP3 + telemetry + objective metrics
```

Project không dùng model có sẵn, classifier, TTS hoặc source code bên ngoài làm backend. Checkpoint sinh ra từ chính trainer trong `genmusic_vn/models` và `genmusic_vn/training`.

## Cài đặt

```powershell
pip install -e ".[self]"
```

Nếu dùng G2P đầy đủ trên Windows, cài eSpeak NG và đặt `PHONEMIZER_ESPEAK_LIBRARY` tới `libespeak-ng.dll`. Khi thiếu binary, G2P vẫn có fallback rule-based.

## Chạy local

Tạo dataset random đúng format model tự code:

```powershell
python -m genmusic_vn.cli make-random-dataset --out datasets/random_self_diffusion --count 16 --frames 128
python -m genmusic_vn.cli validate-dataset --dataset datasets/random_self_diffusion
```

Train checkpoint:

```powershell
python -m genmusic_vn.cli train-self --dataset datasets/random_self_diffusion --checkpoint outputs/self_music.pt --epochs 1 --batch-size 4
```

Sinh thử một đoạn nhạc:

```powershell
python -m genmusic_vn.cli generate-local --text "Mưa rơi nhẹ nhàng, em còn nhớ con đường xưa." --style "Vietnamese pop ballad, warm piano, clear melody" --duration 4 --checkpoint outputs/self_music.pt --steps 6 --out outputs/local_self_music
```

Không truyền `--checkpoint` để kiểm tra inference trước khi train. Khi đó model dùng trọng số random, chỉ phù hợp smoke test chứ chưa phải chất lượng sản phẩm.

## Kaggle

Lệnh `generate` đóng gói source project hiện tại vào dataset private rồi tạo kernel Kaggle. Kernel tự tạo random dataset, train checkpoint self-diffusion và sinh output; không clone repo ngoài và không tải model nhạc có sẵn.

```powershell
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu trên con phố quen." --duration 12 --genre "Vietnamese indie pop, acoustic guitar" --no-submit
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu trên con phố quen." --duration 12 --genre "Vietnamese indie pop, acoustic guitar" --wait
```

Kaggle cần `KAGGLE_USERNAME`, `KAGGLE_KEY` và CLI tương ứng. Dataset random chỉ kiểm tra pipeline; muốn model có chất lượng phải thay bằng dataset audio/mel và lyric thật có quyền sử dụng.

## G2P và căn chỉnh lyric

```powershell
python -m genmusic_vn.cli normalize-lyrics --input data/song.txt --out outputs/song.normalized.txt
python -m genmusic_vn.cli lyrics-g2p --input outputs/song.normalized.txt --out outputs/song.g2p.json
python -m genmusic_vn.cli align-lyrics --audio data/song.wav --lyrics outputs/song.normalized.txt --out outputs/song.lrc --allow-heuristic
```

## Đánh giá và biểu đồ

MOS/CMOS tạm bỏ qua. Metric khách quan và telemetry project vẫn được ghi:

```powershell
python -m genmusic_vn.cli evaluate-self --generated outputs/local_self_music/final.wav --out outputs/self_evaluation
python -m genmusic_vn.cli project-report --source outputs --out outputs/project_report
```

`project-report` sinh biểu đồ thời gian input tới audio, success/error, retry Kaggle, emotion/BPM và user rating. Khi chưa có dữ liệu rating hoặc emotion/BPM, biểu đồ ghi rõ thiếu dữ liệu thay vì tạo số giả.

## Web app

```powershell
python -m genmusic_vn.server --port 8000
```

Mở `http://127.0.0.1:8000`.

## Kiểm thử

```powershell
python -m compileall -q genmusic_vn tests
python -m pytest -q
```

Checkpoint, audio và dataset lớn được giữ ngoài Git.
