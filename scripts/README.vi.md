# Danh mục script

> Bản Việt hóa của `scripts/README.md`. Tên script, option và thuật ngữ API
> được giữ nguyên để có thể đối chiếu trực tiếp với mã nguồn.

Logic model và huấn luyện nằm trong `src/`; các script chỉ là entry point mỏng
bọc quanh phần code đó. Phần lớn file `run_kaggle_*.py` đóng gói một Kaggle job
có giới hạn rõ ràng. Xem `docs/usage.md` để có hướng dẫn đầy đủ.

## Demo V80 có tính di động

- **`run_kaggle_native_waveform_dataset.py`** — bootstrap một lần cho tài
  khoản Kaggle mới. Script đóng gói clone hiện tại, xây corpus vocal/backing
  thô 2.048 bài và lưu ref kết quả trong cấu hình cục bộ đã ignore.
- **`run_kaggle_native_waveform.py`** — gửi một lần sinh V80 bằng corpus đã
  hoàn tất; `cli.py generate` và web app cùng gọi luồng này.
- **`check_portability.py`** — kiểm tra các file sẽ được nộp để phát hiện
  đường dẫn cục bộ, credential và artifact chỉ dành cho báo cáo.

## Pipeline không gian mel (không gian đặc trưng mặc định)

- **`run_kaggle_preprocess_all.py`** — preprocess audio thô theo batch
  (Demucs + Whisper + MuQ-MuLan) thành dataset huấn luyện.
- **`run_kaggle_training.py`** — huấn luyện student bằng `train-self`, không
  có teacher.
- **`run_kaggle_distill.py`** — huấn luyện student bằng `train-distill` với
  teacher DiffRhythm2 thật.
- **`run_kaggle_evaluate.py`** — chạy `evaluate_generation_quality.py`
  (spectral flatness, voiced ratio, pitch-std) cho một checkpoint trên Kaggle.
- **`run_kaggle_full_experiment.py`** (chạy từ xa
  `run_full_experiment.py`) — preprocess → kiểm tra vocoder → huấn luyện
  baseline → distill → generate → thống kê sanity trong một kernel. Đây là
  cách khuyến nghị để chạy trọn thí nghiệm mel-space; xem `docs/usage.md`.
- **`run_kaggle_experiment_matrix.py`** (chạy từ xa
  `run_experiment_matrix.py`) — so sánh baseline, nhiều giá trị
  `alpha_feature` và một kiến trúc nhỏ hơn trên cùng dataset đã preprocess.
- **`run_kaggle_multi_part_training.py`** — preprocess và huấn luyện trên
  nhiều phần dataset để mở rộng vượt corpus một phần; đây là workflow tách
  biệt có chủ đích với các script một dataset ở trên.
- **`run_kaggle_preprocess_raw_audio.py`** — cùng preprocessing nhưng dùng
  `--raw-audio`: bỏ mel, giữ tensor waveform 24 kHz trong `waveforms/*.pt`.
  `train-latent-encoder` và `precompute-latent-dataset` phát hiện
  `raw_audio_mode: true`, cộng trực tiếp waveform vocal/backing và bỏ qua
  Vocos. `train-self` vẫn không dùng được dạng này vì cần dataset mel.
- **`run_kaggle_multi_part_preprocess_raw_audio.py`** — bản `--raw-audio`
  tương ứng của workflow nhiều phần. Script gửi kernel cho từng phần trong
  `RAW_DATASETS`, tôn trọng giới hạn hai GPU session chạy đồng thời của
  Kaggle; `--wait-and-loop` tự nối các phần còn lại và
  `submitted_state.json` giúp lần chạy lại bỏ qua phần đã gửi.

## Pipeline latent native (cùng backbone `MicroDiT`, dataset `latent_mode`)

Pipeline này đưa student vào chính không gian Music VAE 64 chiều/5 Hz của
DiffRhythm2 thay vì mel thô. Xem mục “Native latent backbone and encoder”
trong `docs/architecture.md` để hiểu lý do, lỗi encoder collapse ban đầu và
cách sửa. Chạy theo thứ tự:

1. **`run_kaggle_latent_encoder.py`** — pretrain `LatentAudioEncoder` với
   decoder BigVGAN thật đã đóng băng, dùng reconstruction và cyclical KL
   (beta tối đa mặc định theo báo cáo là `0.15`). Phải sanity-check trước khi
   đi tiếp: loss phẳng/dao động hoặc `pitch_std_semitones` gần 0 khi decode
   latent ground truth nghĩa là cần huấn luyện lại encoder lâu hơn. Script
   nhận nhiều `--processed-kernel-ref` để gộp dataset, hoặc shortcut
   `--raw-audio-part 1 2 3 4 5 6` khi đã đặt các biến môi trường tương ứng.
2. **`run_kaggle_latent_pipeline.py`** — precompute dataset latent bằng
   encoder, huấn luyện CFM student và sinh một mẫu. Shortcut
   `--raw-audio-part` ở đây chỉ nhận một phần vì
   `precompute-latent-dataset` chưa gộp nhiều nguồn như bước huấn luyện
   encoder.
3. **`run_kaggle_latent_resume.py`** — resume CFM từ checkpoint đã tải nếu
   bước 2 bị giới hạn thời gian Kaggle cắt ngang. Mỗi lượt chỉ nên dùng số
   epoch nhỏ, có giới hạn; xem docstring của script.
4. **`run_kaggle_latent_generate_only.py`** — cách rẻ nhất để spot-check một
   checkpoint có sẵn: chỉ sinh một mẫu, không huấn luyện, không dataset
   (khoảng 10 phút). Dùng giữa các vòng train thay vì chạy lại cả pipeline.
- **`run_kaggle_check_latent_encoder_quality.py`** — launcher Kaggle cho
  `check_latent_encoder_quality.py`: tải checkpoint encoder và thực hiện
  sanity check encode/decode ground truth với decoder `bigvgan` thật. Phải
  chạy sau bước 1 và trước khi tin encoder ở bước 2.

## Tiện ích

- **`evaluate_generation_quality.py`** — cài đặt metric thật (spectral
  flatness, voiced ratio, pitch-std theo semitone), được
  `run_kaggle_evaluate.py` và các phép đo khác sử dụng. Cũng có thể chạy độc
  lập trên checkpoint/WAV cục bộ.
- **`check_latent_encoder_quality.py`** — sanity-check checkpoint
  `LatentAudioEncoder` trước khi tin bất kỳ CFM downstream nào: encode audio
  ground truth thật, decode qua decoder đã đóng băng, rồi báo
  `pitch_std_semitones`. Nhờ vậy phát hiện được failure mode encoder collapse
  mà không cần script dùng một lần. Cần decoder thật nên chỉ chạy trên
  Kaggle.
- **`check_kernel_progress.py`** — theo dõi log của Kaggle kernel đang chạy
  qua SSE log-stream. `kaggle kernels output` chỉ trả file khi kernel kết
  thúc, nên đây là cách xem tiến độ trực tiếp.
