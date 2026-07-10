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

Tạo và upload dataset training vào một đường dẫn Kaggle cố định (mặc định là `<KAGGLE_USERNAME>/genmusic-vn-self-diffusion-training`):

```powershell
python -m genmusic_vn.cli make-and-upload-dataset --out datasets/random_self_diffusion_training --target-gb 1
```

`--target-gb` dùng cache augmentation để đạt kích thước tối thiểu và kiểm tra đường ống dữ liệu. Đây không phải dữ liệu âm nhạc có chất lượng; muốn cải thiện model cần thay bằng audio/mel và lyric có quyền sử dụng.
Có thể chọn dung lượng khác, ví dụ `--target-gb 5`. Nếu muốn chỉ rõ đường dẫn của tài khoản khác, dùng `--dataset-ref owner/slug` hoặc đặt `GENMUSIC_KAGGLE_DATASET_REF=owner/slug`.

Train checkpoint:

```powershell
python -m genmusic_vn.cli train-self --dataset datasets/random_self_diffusion --checkpoint outputs/self_music.pt --epochs 1 --batch-size 4
```

Sinh thử một đoạn nhạc:

```powershell
python -m genmusic_vn.cli generate-local --text "Mưa rơi nhẹ nhàng, em còn nhớ con đường xưa." --style "Vietnamese pop ballad, warm piano, clear melody" --duration 4 --checkpoint outputs/self_music.pt --steps 6 --out outputs/local_self_music
```

Không truyền `--checkpoint` để kiểm tra inference trước khi train. Khi đó model dùng trọng số random, chỉ phù hợp smoke test chứ chưa phải chất lượng sản phẩm.

## Tự cải thiện qua input mới

Chạy 10 input đa dạng không có trong dataset. Mỗi vòng sinh bản trước, train candidate bằng mel feedback, sinh bản sau và chỉ giữ checkpoint nếu điểm tổng hợp tăng:

```powershell
python -m genmusic_vn.cli self-improve --dataset datasets/random_self_diffusion_1gb --checkpoint outputs/self_music_1gb_subset.pt --out outputs/self_improve_10 --rounds 10 --duration 4 --steps 4 --max-records 64
```

Kết quả nằm trong `outputs/self_improve_10/self_improve_report.json` và checkpoint được chấp nhận nằm ở `outputs/self_improve_10/final_checkpoint.pt`. Coverage và vocal presence hiện là proxy; WER/độ có vocal thật cần ASR tiếng Việt và vocal stem.

## Kaggle

Lệnh `generate` đóng gói request/source vào dataset private, attach dataset training cố định rồi tạo kernel Kaggle. Kernel không tự tạo dataset mới theo từng request.

```powershell
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu trên con phố quen." --duration 12 --genre "Vietnamese indie pop, acoustic guitar" --no-submit
python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu trên con phố quen." --duration 12 --genre "Vietnamese indie pop, acoustic guitar" --wait
```

Kaggle cần `KAGGLE_USERNAME`, `KAGGLE_KEY` và CLI tương ứng. Dataset random chỉ kiểm tra pipeline; muốn model có chất lượng phải thay bằng dataset audio/mel và lyric thật có quyền sử dụng.
Nếu dataset training cố định chưa tồn tại hoặc chưa ở trạng thái `ready`, job trả lỗi và hướng dẫn chạy `make-and-upload-dataset`.
Khi stage hoặc submit job, web app hiển thị link bấm được tới dataset và kernel Kaggle nếu đã cấu hình credential.

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
