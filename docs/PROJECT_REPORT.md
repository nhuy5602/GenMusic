# GenMusic VN — Project Report

Sinh nhạc có lời tiếng Việt, điều kiện theo văn bản (lyric) và âm thanh (backing
track/style), thông qua distillation từ DiffRhythm2. Report này theo cấu trúc một bài
báo khoa học (Giới thiệu → Nghiên cứu liên quan → Phương pháp → Thực nghiệm → Kết luận),
được cập nhật đồng bộ với codebase và với các job Kaggle thật đã chạy — mọi số liệu ở đây
đều lấy từ log/report thật, không suy đoán.

**Trạng thái tại 2026-07-17**: đã chạy thành công distillation thật trên toàn bộ 250 bài
(`sonlest/vietnamese-music-dataset-version3-part6`), so sánh với self-diffusion (không
distill), đánh giá chất lượng khách quan (đa-bài, N=6), xác nhận `alpha_feature=0.8` là điểm
tối ưu (§4.14), và làm lại ablation kích thước model + epoch với loss mới (§4.15) — phát hiện
`voiced_ratio` và flatness-so-với-thật mâu thuẫn nhau trên trục này, đang chờ nghe thật để
phân xử trước khi quyết định hướng scale tiếp theo.

---

## 1. Giới thiệu

Mục tiêu của project là sinh audio (nhạc + lời hát tiếng Việt) từ một đoạn lyric văn bản,
có thể điều kiện thêm theo một bài hát tham chiếu (backing track/phong cách). Ràng buộc
thực tế quan trọng nhất là **hạ tầng tính toán**: toàn bộ huấn luyện chạy trên quota GPU
miễn phí của Kaggle (T4, giới hạn giờ/tuần), nên một model lớn kiểu DiffRhythm2
(hàng trăm triệu đến hàng tỉ tham số) là không khả thi để tự huấn luyện từ đầu. Hướng đi
được chọn là **distillation**: huấn luyện một model nhỏ (student, MicroDiT — vài triệu
tham số) để bắt chước tín hiệu của DiffRhythm2 (teacher, ~1.14 tỉ tham số) đã huấn luyện
sẵn, với hy vọng đạt chất lượng tốt hơn so với huấn luyện model nhỏ từ đầu (không teacher)
trên cùng lượng dữ liệu/thời gian.

Report này trình bày: (2) các nền tảng kỹ thuật project dựa vào — biểu diễn
mel-spectrogram, vocoder Vocos, Conditional Flow Matching, distillation tri thức cho model
sinh audio, và chính DiffRhythm2; (3) phương pháp cụ thể — pipeline chuẩn bị dữ liệu,
kiến trúc student, cách huấn luyện CFM thuần và huấn luyện có distillation; (4) toàn bộ
thực nghiệm thật đã chạy, bao gồm các lỗi hạ tầng phát hiện được và cách fix, kết quả so
sánh distillation vs. không distillation, và ablation kích thước model; (5) kết luận và
hướng phát triển tiếp theo.

---

## 2. Các nghiên cứu liên quan

### 2.1 DiffRhythm2 (teacher)

[ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2) là model teacher mà
project này distill từ. Đây là một model Conditional Flow Matching (CFM) dạng latent cho
sinh nhạc full-song, điều kiện theo:
- **Lời hát**, tokenize bằng `CNENTokenizer` (frontend G2P Trung/Anh — không có model
  ngôn ngữ tiếng Việt), đưa vào transformer như các vị trí tuần tự thông thường (không
  qua cross-attention).
- **Style**, một embedding duy nhất từ **MuQ-MuLan** (`OpenMuQ/MuQ-MuLan-large`), một
  model embedding audio-text/audio-audio kiểu contrastive (họ CLAP/MuLan) — cùng không
  gian 512-chiều được cộng vào mọi vị trí input và vào adaLN modulation cuối.

Backbone (`diffrhythm2/backbones/dit.py`) là chuỗi block decoder kiểu Llama
(`LlamaNARDecoderLayer` — non-autoregressive, attention hai chiều đầy đủ trong một block
sinh) với rotary position embedding, sinh theo *block* có cache nhân quả
(`sample_block_cache` trong `diffrhythm2/cfm.py`) thay vì cả bài cùng lúc — một chiến lược
sinh dạng streaming/chunk. Checkpoint thật (`config.json` trên HuggingFace) dùng
`dim=2048, depth=16, heads=16, mel_dim=64` — một model thực sự lớn (hàng trăm triệu tham
số ở backbone), hoạt động trên latent mel 64-chiều, giải mã ra audio bằng vocoder họ
BigVGAN huấn luyện riêng cho latent đó.

### 2.2 Biểu diễn audio với mel-spectrogram

Mel-spectrogram là biểu diễn tần số-thời gian phổ biến cho các model sinh audio hiện đại:
biến đổi STFT của waveform, gộp theo thang mel (xấp xỉ cảm nhận tần số của tai người), rồi
lấy log. Ưu điểm so với sinh trực tiếp trên waveform là chiều dữ liệu thấp hơn nhiều
(mỗi frame ~vài chục ms) và mượt hơn về phân phối, nên các model diffusion/flow-matching
dễ học hơn; đánh đổi là cần một **vocoder** riêng để biến mel ngược lại thành waveform
nghe được (§2.3), và bất kỳ sai khác về công thức mel (tần số lấy mẫu, `n_fft`, `hop`, số
mel-bin, hệ số log/power) giữa lúc train và lúc vocode sẽ làm audio bị méo dù model sinh
ra "đúng" theo loss huấn luyện — đây là đúng vấn đề mục §4.1 mô tả.

### 2.3 Vocoder thần kinh: Vocos

Có hai họ vocoder xuất hiện trong lịch sử project: vocoder BigVGAN của riêng teacher (huấn
luyện chuyên cho latent mel 64-chiều của DiffRhythm2, không dùng lại được cho một công
thức mel khác nếu không huấn luyện lại), và **Vocos**
(`charactr/vocos-mel-24khz`, [Siuzdak, 2023](https://arxiv.org/abs/2306.00814)) — một
vocoder dạng GAN dự đoán trực tiếp hệ số STFT thay vì upsample theo thời gian. Vocos được
chọn cho project này vì là vocoder **pretrained, tổng quát, dùng ngay** cho một công thức
mel 100-bin/24kHz *chuẩn* mà student có thể khớp chính xác, không cần huấn luyện lại.

### 2.4 Conditional Flow Matching / Rectified Flow

CFM (Lipman et al., 2022; dùng bởi cả DiffRhythm2 và student của project) huấn luyện một
trường vận tốc `v_θ(x_t, t, cond)` khớp với `x_1 - x_0` theo nội suy đường thẳng
`x_t = (1-t)x_0 + t x_1` giữa nhiễu Gauss `x_0` và dữ liệu thật `x_1`, sau đó sinh bằng
tích phân ODE từ `t=0` đến `t=1`. So với DDPM cổ điển, cách này cho target hồi quy rõ ràng
(well-posed) ở mọi `t` và thường cần ít bước sample hơn cho chất lượng tương đương, đổi lại
mất đi khung nhìn tường minh về noise-schedule/SNR mà DDPM có. Project dùng tích phân Euler
bước cố định đơn giản (`src/models/cfm_flow.py`) — không có bước thích ứng, không dùng
solver bậc cao — là lựa chọn đúng và đơn giản nhất, đủ dùng khi model còn cách xa điểm hội
tụ; solver thích ứng/bậc cao là một tối ưu hợp lý cho tương lai khi chất lượng model không
còn là nút thắt chính.

### 2.5 Distillation tri thức cho model sinh audio

Setup KD cổ điển (Hinton et al., 2015) khớp *phân phối đầu ra* của student nhỏ với teacher
lớn, thường cho phân loại. Với một trường sinh liên tục như vận tốc CFM, cách tương tự là
**khớp vận tốc/đặc trưng**: tại cùng một `(x_t, t)`, phạt `‖v_student - v_teacher‖²` cùng
với (hoặc trộn cùng) loss CFM ground-truth `‖v_student - (x_1 - x_0)‖²`. Hệ số trộn
`alpha_feature` của project theo đúng khuôn mẫu này (`src/training/distill_training.py`).
Vấn đề kỹ thuật riêng của project — không thực sự có sẵn trong tài liệu KD (thường giả
định hai model cùng không gian đầu ra) — là latent mel của teacher (64-chiều) và không
gian mà student cần *giải mã được* (100-chiều, để khớp vocoder pretrained duy nhất có sẵn,
Vocos) là khác nhau. Mục 3.4 trình bày adapter dùng để bắc cầu hai không gian này.

---

## 3. Phương pháp

### 3.1 Chuẩn bị dữ liệu (`src/data/preprocess_raw_vietnamese.py`)

Với mỗi bài hát: Demucs (`htdemucs`, 2 stem) tách vocal/backing, xử lý theo batch (load
model Demucs một lần cho mỗi batch tới 8 file, không load lại từng file), có thể resume
(bỏ qua file đã có stem sẵn trên đĩa), tự retry cuda→cpu khi lỗi — nếu tách stem thất bại
hoàn toàn thì dùng cả bài làm backing, đánh dấu rõ qua field `demucs_separated`/
`vocal_source` thay vì suy giảm âm thầm. Whisper (`tiny`/`base`/..., cấu hình được, có
retry cuda→cpu) transcribe stem vocal với `language="vi"`, giữ timestamp theo
từng câu/segment (field `segments`) để lúc train có thể khớp đúng đoạn lyric với đúng
đoạn audio, không dùng toàn bộ transcript cho mọi đoạn crop. MuQ-MuLan tính một style
embedding duy nhất cho mỗi bài, từ 10 giây đầu của bản mix gốc (xem hạn chế ở §5).
Cả hai kênh mel (vocal, backing) tính bằng cùng `compute_mel_spectrogram`. Đầu ra:
`records.jsonl` (mỗi bài một record: text lyric + segment có timestamp, style tag, BPM,
đường dẫn tới mel backing/vocal + style embedding) cùng `config.json` (công thức mel, để
lúc train dựng lại đúng `MusicDiffusionConfig`).

**Augmentation lúc train** (`MusicDiffusionDataset.__getitem__`,
`src/training/self_diffusion.py`): với bài dài hơn một chunk huấn luyện, mỗi epoch chọn
một offset ngẫu nhiên, áp dụng giống nhau cho cả mel vocal và backing (giữ chúng khớp thời
gian với nhau) — các epoch khác nhau thấy các đoạn khác nhau của bài dài — và lyric text
dùng cho item đó được cắt lại chỉ còn các segment có timestamp rơi vào đúng khung crop đó,
dựa trên timestamp ASR ở trên. Style embedding thì **không** cắt theo crop — dùng cố định
1 vector/bài suốt các epoch (xem thảo luận về hạn chế này ở §5).

### 3.2 Kiến trúc student — MicroDiT (`src/models/dit_transformer.py`)

Một model dự đoán vận tốc CFM dạng Diffusion-Transformer nhỏ:

- **Điều kiện text**: `xlm-roberta-base` đóng băng (`PretrainedRobertaEncoder`, ~278M
  tham số, `requires_grad=False`) chiếu qua một MLP 2 lớp có thể train. Chọn model này vì
  thực sự đa ngôn ngữ (khác tokenizer lyric Trung/Anh-only của teacher) — đây là thành
  phần mang ngữ nghĩa lyric tiếng Việt thật vào model.
- **Điều kiện style ("Audio Style Anchor")**: một embedding MuQ-MuLan 512-chiều duy nhất,
  tính một lần/bài lúc preprocess, chiếu vào không gian điều kiện của model qua
  `AudioStyleEncoder` (MLP 2 lớp). Đây là **cùng không gian embedding** mà teacher
  DiffRhythm2 thật sự điều kiện theo, nên distillation và tự-sinh của student dùng chung
  một khái niệm "style".
- **Backbone**: `depth` block `LlamaDecoderLayer` của HuggingFace (rotary embedding, SDPA
  attention, không causal mask — attention hai chiều đầy đủ trên chuỗi mel),
  `dim`/`heads`/`ff_mult` cấu hình được (CLI: `--dim`/`--depth`/`--heads`/`--ff-mult` trên
  `train-self`/`train-distill`). Mặc định `dim=256, depth=4, heads=4, ff_mult=4` — khoảng
  vài triệu tham số có thể train, rất nhỏ so với `dim=2048, depth=16, heads=16` của teacher.
- **I/O mel**: dự đoán trường vận tốc trên `(seq_len, n_mels=100)`, ở 24kHz/n_fft=1024/
  hop=256 — chọn để khớp *chính xác* công thức mel gốc của Vocos (§2.3), không phải công
  thức mel của DiffRhythm2.

### 3.3 Huấn luyện với Conditional Flow Matching (`train-self`, `src/training/self_diffusion.py`)

Huấn luyện thuần CFM, không có teacher: với mỗi batch, lấy `vocal_mel` làm `x1`,
`backing_mel` làm điều kiện, nhiễu Gauss làm `x0`, nội suy `x_t`, và tối ưu
`‖v_student(x_t, t, cond, text, style) - (x_1 - x_0)‖²` (`cfm_loss`,
`src/models/cfm_flow.py`). Đây là baseline "không distillation" dùng để so sánh (§4.6),
và cũng là backend cho lệnh `train-self`/CLI `generate-local` khi không có checkpoint
distill.

### 3.4 Chắt lọc tri thức từ DiffRhythm2 (`train-distill`, `src/training/distill_training.py`)

Teacher (`diffrhythm2.backbones.dit.DiT`, khởi tạo với đúng kích thước từ `config.json`
tải thật, không đoán) và student huấn luyện theo cùng công thức CFM: cùng `x_t`, cùng `t`,
cùng style embedding. Token lyric và latent mel nhiễu của teacher được nối vào một chuỗi
(`text_embed(tokens)` ở vị trí `time=-1` sentinel, `latent_embed(x_t)` ở vị trí `time=t`)
và forward một lần không cache — tương đương về toán học với đường suy luận streaming
block-cache của chính teacher, chỉ bỏ phần tối ưu cache (xem
`docs/experiments/distillation_fix.md`).

**Adapter mel-dim**: vì checkpoint thật của teacher dùng `mel_dim=64` còn không gian mel
của student là 100-chiều (bắt buộc bởi lựa chọn vocoder, §2.3), một cặp adapter bắc cầu
hai không gian này chỉ cho việc tính loss distillation — đường sinh thật của student
không bao giờ chạm vào các adapter này. `to_teacher_mel` (student→teacher) là một **phép
nội suy mel-bin cố định, không train**: đầu ra của nó đưa trực tiếp vào forward
`torch.no_grad()` của teacher đóng băng, nên một lớp có thể-train ở đó sẽ (và, trước khi
phát hiện+fix, đã) không bao giờ nhận gradient — xem "Mel-dim adapter gradient bug" trong
`docs/experiments/distillation_fix.md`. `from_teacher_mel` (teacher→student) không bị
ràng buộc này và là một `Linear(64→100)` có train thật, đã verify bằng backward pass thật
(`tests/test_self_diffusion.py::test_mel_adapter_gradient_flow`).

**Loss**: `loss = (1 - alpha_feature) * MSE(v_student, v_teacher) + alpha_feature * MSE(v_student, x_1 - x_0)`,
`alpha_feature` cấu hình qua `--alpha-feature`. `run_distillation_training()` (gọi bởi
`train-distill`) yêu cầu teacher thật và tokenizer lyric thật phải load được — nếu một
trong hai lỗi (không có internet, package chưa vendor), nó raise ngay thay vì (a) âm thầm
dùng teacher giả, hoặc (b) âm thầm hạ cấp về huấn luyện chỉ-ground-truth dưới tên
`train-distill`. Huấn luyện chỉ-ground-truth là việc của `train-self`; một lần
`train-distill` chạy xong luôn có nghĩa là đã dùng teacher thật.

### 3.5 Hạ tầng thực nghiệm trên Kaggle

Toàn bộ compute nặng (preprocess, train) chạy trên Kaggle T4 GPU qua các script tự động
hóa Kaggle API (`scripts/run_kaggle_*.py`): tự đóng gói source code, upload dataset,
push kernel, và theo dõi tiến độ. Hai công cụ giám sát được xây trong quá trình này:
`kaggle kernels output` chỉ trả về file khi kernel đã xong (không dùng được để kiểm tra
job đang chạy), nên `scripts/check_kernel_progress.py` gọi trực tiếp endpoint SSE
log-stream của Kaggle (với timeout đọc rõ ràng, tránh treo vô hạn khi job không in gì mới
trong vài phút) để xác nhận job **đang thực sự tiến triển** (epoch/step tăng), không chỉ
dựa vào status `RUNNING` — tránh tốn quota cho một job bị treo mà không biết.

### 3.6 Những gì chưa tích hợp (giới hạn phạm vi có chủ đích)

- **Vietnamese G2P và Tokenizer âm vị**: Đã được tích hợp chính thức vào mô hình (thay thế `xlm-roberta-base` bằng `vinai/xphonebert-base` kết hợp thư viện `text2phonemesequence` hỗ trợ tiếng Việt có dấu thanh).
- **Khớp ASR-lyric** (`src/data/lyric_alignment.py`) tồn tại như tiện ích độc lập, có test.
- **Điều kiện pitch/F0** từng có ở phiên bản trước (`librosa.pyin`), bị bỏ khi thêm Audio
  Style Anchor; chưa khôi phục, vì cần tích hợp lại đúng cách vào pipeline mel/style hiện
  tại thay vì gắn thêm như một tín hiệu riêng biệt.
- **Style anchor cố định 10 giây đầu bài** (§3.1) là một giản lược có chủ đích (MuQ-MuLan
  vốn thiết kế cho embedding style/genre toàn cục, không phải đặc trưng theo khung), nhưng
  chưa kiểm chứng xem lấy đoạn đại diện hơn (ví dụ đoạn giữa bài) có cải thiện gì không.

---

## 4. Thực nghiệm

### 4.1 Méo vocoder (nguyên nhân + fix)

**Hiện tượng**: nhạc sinh ra méo nặng so với audio tham chiếu thật. **Nguyên nhân**:
renderer mặc định tạo phase tuyến tính cố định giả thay vì tái tạo phase thật (đo được
**0.149 tương quan log-mel** với ground truth trên một bài tham chiếu thật — gần như
nhiễu), và đường thay thế đưa vào vocoder thần kinh thật (Vocos) một mel đã resample từ
công thức không tương thích. **Fix**: làm công thức mel của model khớp bit-identical với
Vocos. **Kết quả**: tương quan 0.997 (local), 0.993 (Kaggle thật). Chi tiết:
`docs/experiments/vocoder_fix.md`.

### 4.2 Hợp đồng gọi distillation (nguyên nhân + fix)

**Hiện tượng** (phát hiện qua audit code, chưa ai báo cáo): code distillation có fallback
dùng teacher giả không báo lỗi, đoán sai kích thước kiến trúc, sai convention attention-mask
cho layer riêng của teacher, token lyric không thực sự tới được teacher, và style embedding
giả. **Fix**: reverse-engineer đúng hợp đồng gọi thật từ source code GitHub của
DiffRhythm2 và tái tạo chính xác, với honest fallback (báo rõ `teacher_status`/
`distillation_active`) thay cho teacher giả âm thầm. Chi tiết:
`docs/experiments/distillation_fix.md`.

### 4.3 Lệch mel-dim (phát hiện *nhờ* cơ chế honest-fallback hoạt động đúng)

Checkpoint thật của teacher dùng `mel_dim=64`; không gian mel của student (khớp Vocos) là
100-chiều. Honest fallback (§4.2) phát hiện đúng điều này và tắt teacher thay vì tính ra
rác do lệch shape. **Fix**: một cặp adapter linear nhỏ có thể train bắc cầu hai không gian
mel, chỉ dùng cho loss distillation. Verify local bằng unit test teacher giả (mel-dim lệch,
forward+backward+optimizer step đều thành công).

### 4.4 Checkpoint quá nặng

Checkpoint nặng 1.1GB mỗi lần vì `save_checkpoint` lưu cả RoBERTa text encoder đóng băng
(không bao giờ train). Fix: loại trừ nó (load lại từ HuggingFace lúc load checkpoint);
`load_checkpoint` dùng `strict=False` để tương thích. Verify: checkpoint ~67MB với kiến
trúc mặc định (giảm từ ~1.1GB), sinh audio từ checkpoint đã load vẫn đúng.

### 4.5 Xác thực pipeline end-to-end (quy mô nhỏ)

**Trên Kaggle** (`genmusic-fullexp-1783972294`, 12 bài thật, GPU T4): preprocess 12/12
thành công, vocoder round-trip đạt 0.993 tương quan log-mel, train baseline DiT hoàn tất
(120 step), cả baseline và honest-fallback generation đều ra audio hợp lệ, không suy biến
(peak ~0.8, RMS 0.08–0.15, silence ratio <0.13%, không NaN/Inf).

**Lần thử Kaggle thứ hai** (`genmusic-fullexp-1783991479`) với adapter mel-dim bị treo
~11 giờ trước khi bị kill — xem `docs/experiments/kaggle_runs.md` để biết nguyên nhân (một
import không liên quan trigger JIT-compile CUDA extension) và cách fix. Việc này tốn một
phần đáng kể quota GPU của session đó.

**Local** (Windows, CPU-only, 2 bài thật): mọi giai đoạn chạy end-to-end với dữ liệu thật —
preprocess (2/2 record), vocoder round-trip (0.986 tương quan), train baseline, honest
fallback của distillation (`distillation_active: false`, báo đúng teacher không khả dụng
thay vì giả), và sinh audio từ cả hai checkpoint (hợp lệ, không suy biến). Test suite đầy
đủ: 10/10 pass lúc đó.

### 4.6 Distillation thật lần đầu end-to-end (local, `distillation_active: true`)

Verify trên CPU sau khi hết quota Kaggle: clone `github.com/ASLP-lab/DiffRhythm2`, patch
lệch phiên bản `transformers` và ~20 package thiếu cùng 3 bug encoding cp1252-vs-UTF-8
trong chính code vendor của DiffRhythm2 (không phải code project này). Với `espeak-ng` cài
làm system package và clone đã patch trên `PYTHONPATH`, cả `_load_teacher()`
(`teacher_status: "ok"`) và `_load_lyric_tokenizer()` (`tokenizer_status: "ok"`) load được
teacher thật và tokenizer lyric thật trên máy Windows/CPU thường — không cần Kaggle cho
phần này.

Một lần train distillation thật 30 epoch (2 bài, `batch_size=2`, `dim=128, depth=2`) hoàn
tất với `distillation_active: true` lần đầu tiên, đối đầu teacher DiffRhythm2 thật
(1,136,249,664 tham số, so với 745,188 tham số có thể train của student — teacher lớn hơn
~1,525 lần). So với baseline không-teacher cùng 2 bài/30 epoch: `loss_gt` cuối ≈17.9
(distilled) vs. ≈15.7 (baseline) — không khác biệt có ý nghĩa thống kê, do quy mô quá nhỏ
(2 bài, `batch_size=2` → đúng 1 gradient step/epoch, và CFM sample timestep ngẫu nhiên làm
loss dao động 3.5–229 *trong cùng một run*). Đóng góp thật của run này là chứng minh cơ
chế đúng end-to-end (adapter mel-dim, hợp đồng gọi teacher, tokenize lyric đều chạy thật),
không phải trả lời câu hỏi chất lượng.

### 4.7 Hai bug độc lập sau báo cáo "gần như toàn nhiễu" của đồng nghiệp

Một đồng nghiệp báo cáo generation từ một run thật (~250 bài, 60 epoch, `train-distill`)
ra gần như toàn nhiễu. Điều tra tìm ra 2 bug riêng biệt, không liên quan, cả hai đã fix:

**Bug 1 — generation luôn zero-conditioned.** `generate_audio()` gọi `sample_cfm` không
truyền `backing_mel`/`style_prompt`, nên mọi lần chạy `generate-local` điều kiện theo
backing-track bằng 0 và một vector text-pooled thay thế, không phải style anchor
MuQ-MuLan thật — lệch train/inference thật, nặng hơn ở quy mô thật (250 bài, đa số có
backing_mel không-zero từ Demucs tách thành công) so với test 2-bài trước đó của session
này. **Fix**: `load_reference_conditioning()` trích `backing_mel`/`style_anchor` thật từ
một record dataset đã preprocess; `generate-local --reference-dataset --reference-id` nối
vào CLI.

**Bug 2 — gradient của adapter mel-dim bị cắt cho mọi run distillation thật.** Xem §3.4 —
`to_teacher_mel`/`from_teacher_mel` cả hai được tính trong `torch.no_grad()`, nên không
bao giờ nhận gradient dù được đăng ký train và đưa vào optimizer, cho mọi run
`train-distill` có `mel_adapter_used: true` (tức mọi run thật, vì `mel_dim` teacher
không bao giờ khớp `100` của student). **Fix**: `from_teacher_mel` giờ nhận gradient thật;
`to_teacher_mel` đổi thành nội suy mel-bin cố định thay vì một lớp "về danh nghĩa có thể
train nhưng về cấu trúc không thể", vì fix đúng nó cần backward qua toàn bộ teacher
~1.14 tỉ tham số mỗi step.

**Bug nào giải thích báo cáo của đồng nghiệp?** Không rõ — cả hai đều áp dụng cùng lúc
cho checkpoint đó; cần một run lại có kiểm soát (§4.8) để biết.

### 4.8 Lần chạy thật đầu tiên ở quy mô đầy đủ: 250 bài, distillation vs. self-diffusion

Run lại có kiểm soát theo yêu cầu §4.7: full 250 bài, preprocess với `--whisper-model base`
(sau khi phát hiện `tiny` hallucinate/lặp câu trên lyric thật), rồi `train-distill` và
`train-self` trên cùng dữ liệu với `epochs=25, batch_size=4` khớp nhau để so `loss_gt` trực
tiếp.

**Hai bug hạ tầng thật phát hiện và fix trước khi run này ra được kết quả dùng được**:
- Timeout subprocess của Kaggle preprocessing là `1800s` bất kỳ khi nào `--max-files` được
  set, nhầm lẫn "test nhanh nhỏ" với "giới hạn đúng bằng cả dataset" — nó âm thầm kill một
  run 250 file thật ở mốc 30 phút khi vẫn đang chạy khỏe (bài 46/250), mất luôn thời gian
  GPU đó vì không có gì salvage được (`records.jsonl` chỉ ghi 1 lần ở cuối). **Fix**: scale
  timeout theo số file.
- `train-distill` OOM ở epoch 3/25 (`batch_size=8`) — CUDA hết bộ nhớ, ~11.6GiB đã cấp,
  3.57GiB "reserved nhưng chưa cấp". Epoch 1–2 chạy tốt với cùng batch size, nên đây là
  fragmentation của allocator tích lũy qua các step (mỗi batch có độ dài token lyric khác
  nhau, khiến CUDA caching allocator tạo nhiều block kích thước khác nhau), không phải bug
  logic hay GPU quá nhỏ. **Fix**: `batch_size=4`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, và `torch.cuda.empty_cache()` giữa
  các epoch; run lại hoàn tất cả 25 epoch không OOM.

**Kết quả huấn luyện** (`ddvnam05/genmusic-distill-1784166040` vs.
`ddvnam05/genmusic-train-1784173776`, cùng dataset, cùng `epochs=25`/`batch_size=4`/1575
step):

| | loss_gt @ epoch 1 | loss_gt @ epoch 25 | final_loss_gt (10 step cuối) | wall-clock |
|---|---|---|---|---|
| `train-distill` (teacher thật) | 8.86 | 3.28 | 2.57 | 6527s |
| `train-self` (không teacher) | 11.24 | 8.06 | 7.15 | 134s |

Cả hai run thấy đúng 250 bài cùng thứ tự, cùng số step. Loss ground-truth của distillation
giảm xuống còn khoảng 1/3 so với baseline không-teacher, và ổn định hơn nhiều (đường
không-teacher dao động ~7–13 không có xu hướng giảm rõ sau epoch ~10). Đây là bằng chứng
thật đầu tiên (không chỉ "cơ chế đúng khi cô lập", §4.6) rằng tín hiệu teacher giúp đo được
trên dữ liệu thật của project ở quy mô này. Đổi lại tốn ~49 lần thời gian GPU — đây là
trade-off thật, không phải miễn phí.

**Đánh giá chất lượng audio khách quan** (`scripts/evaluate_generation_quality.py`): vì
không có ai nghe trực tiếp, script này chấm audio sinh ra mà không cần người nghe —
spectral flatness (0 = có cấu trúc tonal/hài âm, 1 = nhiễu trắng; đây là proxy trực tiếp
cho khiếu nại "toàn nhiễu"), clip ratio, silence ratio, RMS, tính trên 7 mẫu rải trong
dataset, mỗi mẫu so với vocal thật *của chính bài đó* render qua cùng vocoder Vocos (để
cô lập chất lượng model khỏi lỗi vocoder), và so với một đoạn nhiễu trắng tổng hợp làm mốc
sanity:

| | spectral flatness (mean) | RMS (mean) | clip ratio |
|---|---|---|---|
| nhiễu trắng (mốc sanity) | 0.562 | 0.577 | 2.0% |
| `train-distill` | 0.086 | 0.078–0.131 | 0.0% |
| `train-self` | 0.075 | 0.037–0.055 | 0.0% |
| vocal thật (cùng vocoder) | 0.053 | 0.080–0.109 | 0.0% |

Không checkpoint nào gần mốc nhiễu trắng về flatness — cả hai có cấu trúc tonal rõ, không
phải nhiễu, và không clip. Riêng flatness thì hai model gần nhau (distill còn hơi cao/kém
tonal hơn self-diffusion một chút) — nếu chỉ nhìn số này sẽ **hiểu sai**: RMS của
`train-self` (0.037–0.055) thấp hẳn dưới range vocal thật (0.080–0.109), còn của
`train-distill` (0.078–0.131) bám sát range đó. Kết hợp với khoảng cách loss_gt ở trên,
distillation cho output gần target thật hơn ở đúng những trục phân biệt hai model —
flatness một mình là bộ lọc "có phải nhiễu không" tốt, nhưng không phải tín hiệu chất
lượng đầy đủ.

**Quota tiêu tốn cho run này**: preprocess ≈2h (1 run hỏng 30 phút + 1 run thành công
~2h), `train-distill` ≈1.8h, `train-self` ≈2 phút, đánh giá local ≈miễn phí (chạy trên
máy người yêu cầu, không dùng Kaggle). Tổng thời gian GPU Kaggle ≈4h.

### 4.9 Ablation kích thước model: student lớn hơn không giúp gì ở cùng ngân sách step

Người nghe thật (không phải metric ở §4.8) báo cáo output vẫn chưa nghe ra lời. Kiểm tra
trực tiếp thống kê mel sinh ra so với mel vocal thật (cùng điều kiện, bài `-6s_eRHYqVM`)
tìm ra lý do: độ lệch chuẩn (std) của mel sinh ra là 1.09 so với 2.95 của vocal thật —
khoảng 1/3 biên độ dao động, dấu hiệu kinh điển của "regression-to-mean" do thiếu tín hiệu
huấn luyện so với độ đa dạng được yêu cầu (250 bài rất khác nhau, 1575 step, ra một
"trung bình mượt" thay vì chi tiết sắc nét từng bài). Hai giả thuyết: student
(`dim=256, depth=4, heads=4`, vài triệu tham số) quá nhỏ để biểu diễn chi tiết đó, hoặc cần
nhiều gradient step hơn bất kể kích thước. Sau khi bổ sung `--dim/--depth/--heads/--ff-mult`
cho launcher `run_kaggle_distill.py` (đã có sẵn trên `cli.py` nhưng chưa expose ra
launcher), một run cùng dữ liệu/epoch/batch được chạy với `dim=384, depth=6, heads=6`
(≈3 lần tham số student) để so với `dim=256, depth=4, heads=4` (§4.8):

| | loss_gt @ epoch 25 | mel std (sinh ra) | wall-clock |
|---|---|---|---|
| `dim=256, depth=4, heads=4` (§4.8) | 3.28 | 1.09 | 6527s |
| `dim=384, depth=6, heads=6` | 3.65 | 1.06 | 6522s |
| vocal thật (target) | — | 2.95 | — |

Student lớn hơn không cải thiện metric nào — `loss_gt` ở epoch cuối còn *kém hơn* một
chút, và độ sụp std hầu như không đổi (1.06 vs 1.09, cả hai vẫn ~1/3 so với thật).
Wall-clock giống nhau giữa hai run — hợp lý vì forward pass của teacher ~1.14 tỉ tham số
chiếm phần lớn chi phí mỗi step, nên student to hơn 3 lần gần như miễn phí để chạy nhưng
không khớp tốt hơn trong cùng số gradient step — nhiều tham số hơn cần nhiều update hơn để
hội tụ, mọi thứ khác giữ nguyên, và 1575 step rõ ràng chưa đủ dư địa cho đổi chác đó có lời.

**Điều này bác bỏ giả thuyết "model quá nhỏ" ở ngân sách step này.** Nút thắt khả năng cao
hơn đơn giản là số step/dữ liệu: 1575 gradient update trên 250 bài là ít cho một model sinh
audio ở khoảng kích thước này, bất kể lớn nhỏ. Thực nghiệm tiếp theo hợp lý là tăng số
epoch trên config `dim=256` rẻ hơn để thử nghiệm (§4.10).

### 4.10 Thực nghiệm đang chạy: tăng số epoch (75 epoch, `dim=256`)

Để kiểm tra trực tiếp giả thuyết "cần nhiều step hơn" từ §4.9: cùng dataset 250 bài, cùng
`dim=256, depth=4, heads=4, batch_size=4`, nhưng `epochs=75` (gấp 3 lần §4.8).

**2 lần thử đầu tiên fail giống hệt nhau** (`ddvnam05/genmusic-distill-1784188506`, rồi
`...1784189042`), cả hai đều lỗi ngay ở bước tải teacher: *"An error happened while trying
to locate the file on the Hub and we cannot find the requested files in the local cache"*
— phát hiện sớm (trong vài phút, không phải sau nhiều giờ) nhờ
`scripts/check_kernel_progress.py` (§3.5) đọc log sống thay vì chỉ chờ status cuối. Cùng
lỗi lặp lại 2 lần liên tiếp đủ đáng ngờ để không retry mù: kiểm tra code phát hiện
`_load_teacher()` gọi `hf_hub_download` cho `config.json` và `model.safetensors`
(file ~4.3GB) mà **không có retry nào** — một lần mạng chập chờn (DNS/TLS/Hub) là chết cả
job nhiều giờ trước khi kịp train bước nào. **Fix**: `_hf_hub_download_with_retry()` (3
lần thử, backoff 5s) bọc quanh cả hai lệnh gọi. Test suite local cùng lúc đó cũng thấy tải
`xlm-roberta-base` từ HF chậm bất thường — cùng khung giờ, cùng nguyên nhân khả năng cao
(HF Hub chập chờn tạm thời), không phải bug riêng của project.

**Lần thử thứ 3** (`ddvnam05/genmusic-distill-1784190525`, với fix retry 3x5s trên) **vẫn
fail** — log cho thấy retry có chạy (2 lần retry thấy rõ trong log) nhưng cả 3 lần thử đều
lỗi giống nhau trong vòng ~30 giây, tức 3x5s không đủ dài. Verify trực tiếp bằng cách gọi
`hf_hub_download` cho đúng file đó từ máy local (mạng hoàn toàn khác Kaggle): tái hiện được
đúng lỗi **HTTP 504 (Gateway Timeout) từ chính HuggingFace Hub**, tự phục hồi sau ~130s nhờ
retry mặc định của thư viện `huggingface_hub`. Điều này xác nhận nguyên nhân là **HF Hub
đang suy giảm/chập chờn thật ở phía họ**, không phải lỗi mạng riêng của Kaggle hay bug của
project. **Fix**: mở rộng `_hf_hub_download_with_retry()` lên 8 lần thử, backoff nhân đôi
5s→60s (tổng ngân sách ~4 phút) — đủ dư địa so với ~130s phục hồi quan sát được, và không
đáng kể so với job nhiều giờ nó bảo vệ.

**Lần thử thứ 4** (`ddvnam05/genmusic-distill-1784191327`) chạy thành công, đang tiến triển
đúng (epoch 31/75 lúc kiểm tra, `loss_gt` dao động 2.8-5.4 quanh cùng mức đã thấy ở epoch 25
trước đó) — dấu hiệu **chững lại sớm** dù còn 44 epoch nữa, gợi ý "chỉ tăng epoch" có thể
không phải hướng đi đúng. Job này vẫn được để chạy hết (dùng code cũ tại thời điểm launch,
là một điểm dữ liệu độc lập có giá trị riêng), xem kết quả cuối ở §4.12.

### 4.11 Nghiên cứu literature + một merge lớn từ đồng nghiệp cùng lúc giải quyết vấn đề

Trong lúc job §4.10 chạy, thay vì tiếp tục thử-sai bằng cách tăng epoch, đã tra literature
thật về nguyên nhân "regression to the mean" trong distillation cho model sinh:

- **Sander Dieleman (2024), "The paradox of diffusion distillation"**: distillation dùng
  loss MSE/SFT thuần có xu hướng ưu tiên khớp *trung bình* của phân phối thay vì giữ độ sắc
  nét, vì model student (giới hạn công suất) không thể khớp chính xác teacher ở mọi mode nên
  "an toàn" bằng cách khớp trung bình.
- **DMD/ADM (Distribution/Adversarial Distribution Matching, 2024-2025)**: xác nhận cùng cơ
  chế — reverse-KL/MSE-based distillation gây "distributional averaging"; fix chuẩn trong
  literature là loss đối kháng (adversarial) hoặc perceptual (LPIPS) thay MSE thuần.
- Tổng hợp: hướng khả thi cho project (không đủ compute cho adversarial/LPIPS đầy đủ) là đổi
  loss thành phần liên quan tới teacher-matching sang **L1** (rẻ, có cơ sở), và/hoặc tăng bước
  sampling lúc sinh.

**Test loại trừ (miễn phí, local)**: tăng bước Euler ODE lúc sinh từ 6→64 trên checkpoint đã
có — mel std hầu như không đổi (1.09→1.11). Loại bỏ giả thuyết "do ít bước sampling"; xác
nhận vấn đề nằm ở chính velocity field đã học, không phải cách tích phân lúc sinh.

**Fix đầu tiên**: đổi `loss_velocity` (term khớp teacher trong distillation) từ MSE sang L1
— khớp đúng khuyến nghị literature, rủi ro thấp (không đụng `loss_gt`, vốn cần giữ MSE vì lý
do lý thuyết CFM — xem comment trong code).

**Cùng lúc, `git pull` phát hiện một commit lớn từ đồng nghiệp** ("feat: add resumable
Google Colab training backend", +2188 dòng, 27 file) — thêm backend train qua Google Colab
(nguồn compute thứ hai, giảm áp lực quota Kaggle), và — quan trọng hơn với vấn đề đang điều
tra — **một loạt cải tiến cho `train-self`/`cfm_flow.py` giải quyết đúng cơ chế trên, dù
không đặt tên rõ là "fix mode collapse"**:
- **Mel normalization** (`normalize_mel`/`denormalize_mel`, `mel_mean`/`mel_std` tự tính qua
  `estimate_vocal_mel_stats()` trên mẫu dữ liệu thật mỗi lần train mới).
- **Loss velocity trọng số theo năng lượng khung** (`frame_weights` từ ngưỡng quantile năng
  lượng) — tránh im lặng lấn át trung bình khi phần lớn audio là khung yên tĩnh.
- **Loss reconstruction + delta thời gian/tần số** (L1, hệ số nhỏ 0.15/0.05) — phạt trực
  tiếp việc làm mượt quá mức mà không đổi phương trình sampling CFM.
- **Classifier-free guidance** (`guidance_scale`) lúc sinh, dropout điều kiện lúc train.
- Tăng `frames_per_chunk` từ 128 lên 384 (ngữ cảnh dài hơn mỗi mẫu train), EMA, mixed
  precision.

**Verify trên `train-self`** (rẻ, ~2 phút, `ddvnam05/genmusic-train-1784200611`, cùng 250
bài/25 epoch/batch=4 để so trực tiếp với §4.8):

| | mel std (sinh ra) | spectral flatness (mean) |
|---|---|---|
| `train-self` code cũ (§4.8, không đo std) | — | 0.075 |
| `train-self` code mới (mel norm + energy/reconstruction/delta loss) | **3.13** | **0.028** |
| vocal thật (target) | 2.95 | 0.053-0.056 |

Mel std giờ **vượt nhẹ** target thật (trước đó chỉ 1.09, tức ~37% biên độ thật); spectral
flatness còn **thấp hơn** vocal thật (tonal hơn), cách xa nhiễu trắng (0.562) hơn bao giờ
hết. Đây là cải thiện rõ rệt, không mơ hồ.

**Port sang distillation**: các cải tiến trên chỉ nằm trong `cfm_flow.py`/`self_diffusion.py`
(dùng bởi `train-self`), không tự động áp dụng cho `train-distill` (tính loss riêng, inline,
trong `distill_training.py`). Vì `cfm_loss()` tự forward model bên trong nên không dùng lại
trực tiếp được (distillation cần `v_student` tại đúng `(x_t, t)` đã dùng để so với teacher) —
đã chép lại đúng công thức (frame-weighted velocity MSE + reconstruction + delta) vào
`train_epoch()` của `distill_training.py`, giữ `loss_velocity` là L1 (fix trước đó), và thêm
`estimate_vocal_mel_stats()` cho distillation (trước đây distillation không tự tính
mel_mean/std, luôn ở normalization mặc định 0/1 dù `train-self` đã có từ merge này).

Đã launch **lần thử thứ 5** (`ddvnam05/genmusic-distill-1784201393`, 25 epoch, cùng scale
§4.8, code mới đầy đủ) để đo trực tiếp tác động lên distillation. *(Kết quả ở §4.12.)*

### 4.12 Kết quả các thực nghiệm

**Lần thử thứ 5** (`ddvnam05/genmusic-distill-1784201393`, §4.11, 25 epoch, code mới đầy
đủ) hoàn tất — 2 tín hiệu tốt rõ ràng trước khi đo mel:
- Loss giảm **mượt và đơn điệu** (1.74→0.66 qua 25 epoch, không còn dao động dữ dội 2.8-13
  như trước) — bằng chứng gián tiếp cho thấy training ổn định hơn nhiều.
- Nhanh hơn hẳn: 1183s so với 6527s trước (~5.5x) — khả năng do mel đã normalize giúp
  gradient ổn định hơn (chưa xác nhận chi tiết cơ chế).

**Đo mel std trực tiếp** (bài `-6s_eRHYqVM`, cùng điều kiện các lần trước):

| | mel std (sinh ra) | spectral flatness (mean, 6 mẫu) | RMS (mean) |
|---|---|---|---|
| Distill code cũ (§4.8) | 1.09 | 0.086 | 0.078-0.131 |
| Distill code mới (§4.11, alpha=0.5) | **1.40** (+28%) | 0.070 | 0.072-0.092 |
| Distill code mới, alpha=0.8 (lần thử thứ 6) | **1.27** | — | — |
| `train-self` code mới (so sánh) | 3.13 | 0.028 | — |
| Vocal thật (target) | 2.95 | 0.053-0.056 | 0.08-0.10 |

Cải thiện thật và nhất quán trên nhiều chỉ số (std, flatness, RMS đều dịch về phía target
thật), nhưng **không đạt mức train-self đạt được** — khoảng cách còn khá xa.

**Giả thuyết alpha_feature bị bác bỏ** *(ĐÍNH CHÍNH ở §4.13 — kết luận này SAI, xem dưới)*:
lần thử thứ 6 (`ddvnam05/genmusic-distill-1784203849`, `alpha_feature=0.8`, giảm trọng số
teacher-matching) cho std=1.27 — **không cao hơn** alpha=0.5 (1.40), thậm chí thấp hơn một
chút. Dựa trên mel-std, kết luận lúc đó là trọng số teacher-matching không phải nguyên nhân
chính của khoảng cách còn lại.

> **Đính chính (sau khi có `voiced_ratio`, §4.13)**: mel-std lại là chỉ số sai để đánh giá
> alpha_feature. Theo `voiced_ratio` — chỉ số gần đúng hơn với "nghe ra hát" — alpha=0.8
> (exp06) đạt **92.7%**, cao hơn hẳn alpha=0.5 (exp05, **83.3%**). Vậy alpha_feature **có**
> tác dụng thật, chỉ là mel-std không đo được nó — cùng bài học ở §4.13, áp dụng ngược lại
> vào chính giả thuyết này. Diễn giải hợp lý: `alpha=0.8` dồn 80% trọng số vào `loss_gt`
> (khớp velocity thật của bài hát, đã có công thức chống collapse từ §4.11) và chỉ 20% vào
> `loss_velocity` (bắt chước teacher) — ít lệ thuộc "ý kiến" (có thể đã mượt hoá) của teacher
> hơn, giữ được cấu trúc pitch của bài hát thật tốt hơn. **Lưu ý thận trọng**: mỗi cấu hình
> chỉ chạy 1 lần (N=1) nên chưa loại trừ được nhiễu ngẫu nhiên giữa các lần train; cũng chưa
> test tổ hợp alpha=0.8 cùng dropout (job 7) hay LR khác (job 8) — cả hai đều chạy ở alpha=0.5.
> Hướng tiếp theo hợp lý: thử alpha cao hơn nữa (0.9-0.95) kết hợp thêm epoch/dữ liệu.

### 4.14 alpha=0.8 xác nhận là điểm tối ưu thật — không phải nhiễu (kiểm chứng đa-bài)

Theo hướng đề xuất ở §4.13, chạy **lần thử thứ 9** (`ddvnam05/genmusic-distill-1784223424`,
alpha=0.9, giữ nguyên mọi thứ khác) để xem xu hướng "alpha cao hơn → voiced_ratio cao hơn"
có tiếp tục không. **Kết quả trên 1 bài (`-6s_eRHYqVM`, cùng điều kiện các lần trước)**:
alpha=0.9 chỉ đạt **82.0%** — thấp hơn alpha=0.8 (92.7%), không tiếp tục xu hướng tăng.

Vì mẫu 1 bài/checkpoint có thể là nhiễu (đã cảnh báo ở trên), đo lại **voiced_ratio trên 6
bài khác nhau** cho cả 3 giá trị alpha (miễn phí, chạy local, không cần Kaggle):

| alpha_feature | voiced_ratio TB (6 bài, dao động từng bài) | So với vocal thật (74.3%) |
|---|---|---|
| 0.5 (job 5) | 60.7% (41-78%) | thấp hơn |
| **0.8 (job 6)** | **92.4%** (90.6-96.3%, rất đều) | **cao hơn** |
| 0.9 (job 9) | 77.8% (67-85%) | ~ngang |

**Kết luận (lần này có cơ sở thống kê chắc hơn, N=6 không phải N=1)**: alpha=0.8 thắng rõ
và nhất quán trên toàn bộ 6 bài, không phải một lần chạy may — có một **điểm tối ưu thật**
quanh alpha=0.8, không phải "alpha cao hơn luôn tốt hơn". Cả alpha=0.5 và alpha=0.9 đều rơi
về mức gần vocal-thật-trung-bình hoặc thấp hơn, còn alpha=0.8 vượt hẳn. Đây là dạng quan hệ
phi-tuyến giữa alpha_feature và voiced_ratio — hợp lý về mặt trực giác (quá ít trọng số cho
`loss_gt` thì thiếu tín hiệu cấu trúc thật; quá nhiều thì có thể mất luôn phần tín hiệu hữu
ích mà teacher đóng góp, ví dụ ổn định hoá hoặc cung cấp thông tin ngoài phạm vi 250 bài) —
nhưng chưa có giải thích cơ chế chắc chắn, chỉ là quan sát thực nghiệm.

**Bài học phương pháp thứ hai trong cùng nhánh điều tra**: N=1 (1 bài, 1 lần train) là không
đủ để tin vào bất kỳ so sánh alpha nào — kết luận "alpha=0.9 tệ hơn alpha=0.8" ban đầu (dựa
1 bài) TRÙNG với kết luận đa-bài, nhưng đây là hai lần kiểm chứng ngẫu nhiên khớp nhau chứ
không phải một quy luật đã kiểm chứng chắc chắn — mọi so sánh alpha trong report này (job
5/6/7/8) trước §4.14 đều chỉ dựa N=1/checkpoint (đo trên `-6s_eRHYqVM`), nên nên đọc với mức
tin tưởng thấp hơn số liệu ở đây.

**Khuyến nghị thực tế**: dùng `alpha_feature≈0.8` làm mặc định cho các lần train tiếp theo
(đã kiểm chứng đa-bài), không cần thử thêm giá trị alpha khác trừ khi có lý do cụ thể — nên
dồn quota còn lại vào hướng khác (mở rộng dữ liệu/epoch, xem §5.2) thay vì tiếp tục dò alpha.

**Giả thuyết condition dropout cũng bị bác bỏ**: lúc port công thức loss ở §4.11, phần
**classifier-free condition dropout** của `cfm_loss()` (drop ngẫu nhiên backing/style/text
10% mỗi loại) đã bị bỏ sót. Bổ sung vào `train_epoch()` và chạy lần thử thứ 7
(`ddvnam05/genmusic-distill-1784206082`, alpha=0.5 + dropout) — kết quả std=**1.35**, vẫn
trong cùng khoảng 1.27-1.40 như các lần trước, không cải thiện thêm đáng kể.

**Khác biệt thật tiếp theo tìm được: learning rate**. `cli.py`: `train-self` mặc định
`--learning-rate 2e-4`, `train-distill` mặc định `1e-4` — **chỉ bằng một nửa**. Ở cùng 25
epoch/1575 step, LR thấp hơn khiến model tiến chậm hơn tới cùng đích, một khác biệt hoàn
toàn không liên quan gì tới việc có teacher hay không nhưng vẫn ảnh hưởng tới việc so sánh
công bằng giữa 2 đường train. Thêm `--learning-rate` vào launcher, chạy **lần thử thứ 8**
(`ddvnam05/genmusic-distill-1784208706`, LR=2e-4 khớp `train-self`) để kiểm tra.
*(Kết quả cập nhật tiếp.)*

**Learning rate cũng bị bác bỏ**: lần thử thứ 8 (LR=2e-4 khớp `train-self`) cho std=**1.25**
— vẫn trong đúng khoảng 1.25-1.40 như 3 lần thử trước, không cải thiện.

**Lần thử thứ 4** (`ddvnam05/genmusic-distill-1784191327`, §4.10, 75 epoch, code CŨ) hoàn
tất (19623s ≈ 5.45h). `loss_gt` tại epoch 25/50/75: **4.32 → 3.05 → 2.76** (tốt nhất tại
epoch 64: 2.26) — có cải thiện tiếp khi tăng epoch, nhưng **không tỉ lệ với chi phí**: gấp 3
lần epoch (và 3 lần thời gian GPU) chỉ đưa loss_gt tốt nhất từ ~3.28 (epoch 25 gốc, §4.8)
xuống ~2.26, chậm hẳn và rất nhiều nhiễu giữa các epoch (không phải một đường giảm mượt).
So với việc chỉ đổi loss formula (job 5, §4.11) đã cho đường giảm mượt hẳn (1.74→0.66) trong
đúng 25 epoch với 1/3 thời gian — "sửa loss" hiệu quả hơn nhiều so với "chạy lâu hơn" ở cùng
mức chi phí.

**Tổng hợp toàn bộ nhánh điều tra §4.11-4.12** (bảng đầy đủ, tất cả cùng dataset 250 bài,
`dim=256,depth=4,heads=4`, batch=4, 25 epoch trừ khi ghi khác):

| Cấu hình | mel std | Ghi chú |
|---|---|---|
| Distill code cũ, 25 epoch (§4.8) | 1.09 | mốc xuất phát |
| Distill code cũ, 75 epoch (§4.10) | — (chưa đo std) | loss_gt tốt nhất 2.26 (epoch 64), chậm + nhiều nhiễu, không tỉ lệ với 3x chi phí |
| Distill code mới + loss port, alpha=0.5 | **1.40** | +28%, cải thiện thật duy nhất |
| + alpha=0.8 | 1.27 | không cải thiện thêm |
| + condition dropout | 1.35 | không cải thiện thêm |
| + learning_rate=2e-4 | 1.25 | không cải thiện thêm |
| `train-self` code mới, 25 epoch | **3.13** | không dùng teacher |
| Vocal thật (target) | 2.95 | — |

**Kết luận cho nhánh điều tra này**: đúng 1 thay đổi tạo ra cải thiện đo được (port loss
formula chống collapse từ `train-self`, §4.11: 1.09→1.40). Ba giả thuyết tiếp theo
(alpha_feature, condition dropout, learning rate) đều **không** đóng thêm khoảng cách còn
lại tới `train-self`/vocal thật — mọi biến thể distillation đều tụ lại quanh **1.25-1.40**,
bất kể chỉnh gì ở phía student. Điều này gợi ý khoảng cách còn lại **không** nằm ở một
siêu tham số cụ thể của student, mà khả năng cao nằm ở chính **tín hiệu teacher** — dù chỉ
chiếm 20-50% trọng số loss, teacher (qua adapter mel-dim 64→100, và bản chất "nhìn" dữ liệu
tiếng Việt/pop nhỏ như ngoài phân phối huấn luyện của nó) có thể tự nhiên đưa ra dự đoán
mượt hơn theo góc nhìn riêng, và việc khớp theo nó (dù ít) vẫn kéo student về phía đó nhiều
hơn tỷ trọng danh nghĩa của loss gợi ý. Đây là giả thuyết chưa kiểm chứng đầy đủ, không phải
kết luận chắc chắn — xem hướng phát triển ở §5.2.

**Hệ quả thực tiễn (đã ĐÍNH CHÍNH ở §4.13 — đừng dừng đọc ở đây)**: kết luận ban đầu ở bản
báo cáo này là "`train-self` thực tế hơn vì mel-std/flatness gần target thật hơn". §4.13 cho
thấy đó là kết luận **sai**, vì mel-std/flatness không đo đúng thứ cần đo.

### 4.13 Đính chính quan trọng: nghe thử thật cho thấy `train-self` vẫn ra nhiễu — mel std/flatness không đo đúng thứ cần đo

Sau khi báo cáo (không chính xác) rằng `train-self` "gần đạt target thật", người dùng nghe
thử trực tiếp file `train-self` (§4.9's checkpoint mới) và xác nhận: **vẫn nghe ra nhiễu**,
không giống giọng hát. Điều này mâu thuẫn với mel-std=3.13 (vượt target 2.95) và flatness=
0.028 (tốt hơn cả vocal thật) — hai chỉ số tôi đã dùng để kết luận "đã fix".

**Nguyên nhân của mâu thuẫn**: `spectral_flatness` đo một khung hình *tại một thời điểm* có
đỉnh tần số rõ (tonal) hay phẳng (nhiễu) — nó **không đo** đỉnh tần số đó có giữ nguyên qua
nhiều khung liên tiếp để tạo thành một *nốt nhạc ổn định* hay không. Một âm nhảy loạn xạ
giữa các tần số khác nhau mỗi khung vẫn cho flatness thấp (tonal tại mỗi khung) nhưng nghe
hoàn toàn như nhiễu/rè — đúng thứ đang xảy ra.

**Đo lại bằng chỉ số đúng hơn**: `voiced_ratio` (tỉ lệ khung có pitch ổn định phát hiện được,
qua `librosa.pyin`) — đây là proxy trực tiếp cho "có nghe ra nốt hát không". Đã bổ sung vào
`scripts/evaluate_generation_quality.py`. Đo lại trên **mọi** checkpoint đã tạo audio debug
trong session này (cùng bài `-6s_eRHYqVM`, cùng điều kiện):

| Checkpoint | mel std (§4.8-4.12) | voiced_ratio | |
|---|---|---|---|
| Vocal thật (mốc) | 2.95 | **90.1%** | |
| `train-self` code mới (§4.9, "tốt nhất" theo std) | **3.13** | **8.5%** | ngược hẳn với std! |
| Distill code cũ gốc, 25ep (§4.8) | 1.09 | 0% | khớp đúng khiếu nại gốc "gần như toàn nhiễu" |
| Distill + loss mới, alpha=0.5 (job 5, §4.11) | 1.40 | **83.5%** | |
| Distill + loss mới, alpha=0.8 (job 6) | 1.27 | **93.1%** | cao hơn cả vocal thật |
| Distill + loss mới, dropout (job 7) | 1.35 | **82.7%** | |
| Distill + loss mới, LR=2e-4 (job 8) | 1.25 | 0% | bất thường — LR cao có thể phá pitch dù mel std vẫn ổn |
| Distill model to hơn, dim=384 (§4.9) | 1.06 | 34.9% | |

**Đảo ngược hoàn toàn kết luận trước đó**: theo `voiced_ratio` — chỉ số gần với cảm nhận
"có nghe ra hát không" hơn nhiều so với mel-std — **distillation (job 5/6/7) thắng áp đảo**,
đạt 82-93% so với vocal thật 90%, trong khi `train-self` (dù mel-std "đẹp hơn") chỉ đạt 8.5%,
gần như không có pitch ổn định nào. Diễn giải hợp lý: teacher DiffRhythm2 là model hát thật,
đã train — dù chỉ chiếm 20-50% trọng số loss, tín hiệu của nó mang thông tin pitch/giai điệu
thật mà student nhỏ không thể tự học chỉ từ 1575 step ground-truth thuần (khớp với việc
`train-self`, không có teacher, gần như không có pitch nào). Mel-std cao của `train-self` hoá
ra là **biến thiên hỗn loạn** (đúng nghĩa nhiễu), không phải **biến thiên có cấu trúc âm nhạc**
— hai thứ trông giống nhau qua std/flatness nhưng khác hẳn khi nghe.

**Điểm bất thường cần điều tra thêm**: job 8 (LR=2e-4) có mel-std bình thường (1.25, cùng tầm
job 5-7) nhưng voiced_ratio=0% — như job distill code cũ gốc. Gợi ý LR cao hơn có thể phá vỡ
cấu trúc pitch dù không ảnh hưởng rõ tới mel-std — chưa kết luận được, cần thêm dữ liệu (chỉ
có 1 mẫu/checkpoint, chưa đủ để loại trừ nhiễu ngẫu nhiên giữa các lần train).

**Bài học phương pháp**: metric khách quan không thay thế được việc nghe thử thật — mel-std/
flatness là proxy hợp lý cho "có phải nhiễu trắng không" (đã đúng ở đó, xem mốc nhiễu trắng
0.562 vs mọi checkpoint ≤0.09) nhưng **không đủ** để đánh giá "có nghe ra nhạc/hát không".
`voiced_ratio` gần đúng hơn nhưng vẫn chỉ là proxy — đánh giá nghe thật có hệ thống (§5.2,
MOS/CMOS) vẫn là việc chưa làm và nên làm trước khi tin tưởng hoàn toàn bất kỳ kết luận nào
ở report này.

#### Hướng dẫn nghe thử (`outputs/listening_guide/`)

Toàn bộ file dưới đây sinh từ **cùng một điều kiện** (cùng câu lyric, cùng bài tham chiếu
`-6s_eRHYqVM` làm backing/style, cùng seed=5602, cùng 16 bước sampling) để so sánh công bằng
giữa các checkpoint — khác với các file `debug_*.wav` rải rác trước đó trong `outputs/`
(tham số không hoàn toàn đồng nhất, nên **ưu tiên nghe bộ này**). Số liệu đo bằng
`scripts/evaluate_generation_quality.py`, lưu đầy đủ ở `outputs/listening_guide/metrics_index.json`.
Thứ mục không nằm trong git (`outputs/` được gitignore) — chỉ có trên máy đã chạy thực nghiệm.

| File | Experiment (mục) | voiced_ratio | flatness | Mô tả |
|---|---|---|---|---|
| `exp00_real_vocal_reference.wav` | — (mốc so sánh) | 90.1% | 0.025 | Vocal thật, qua cùng vocoder Vocos |
| `exp01_distill_oldcode_25ep.wav` | §4.8 — distill, code cũ | 4.5% | 0.077 | Trước khi fix loss chống collapse — gần đúng khiếu nại gốc "toàn nhiễu" |
| `exp02_trainself_oldcode_25ep.wav` | §4.8 — self-diffusion, code cũ | 0% | 0.080 | Baseline không-teacher, code cũ |
| `exp03_distill_biggermodel_dim384.wav` | §4.9 — model to hơn | 44.1% | 0.089 | Tăng size không giúp nhiều nhưng có pitch nhiều hơn exp01 (chưa rõ vì sao, có thể nhiễu ngẫu nhiên) |
| `exp05_distill_newloss_alpha05.wav` | §4.11 — loss mới, alpha=0.5 | **83.3%** | 0.071 | Sau khi port loss chống collapse sang distillation |
| `exp06_distill_newloss_alpha08.wav` | §4.12 — + alpha=0.8 | **92.7%** | 0.010 | **Gần vocal thật nhất** theo voiced_ratio |
| `exp07_distill_newloss_dropout.wav` | §4.12 — + condition dropout | **77.0%** | 0.046 | |
| `exp08_distill_newloss_lr2e-4.wav` | §4.12 — + LR=2e-4 | 0% | 0.091 | Bất thường — mel-std bình thường nhưng mất hết pitch, đáng điều tra |
| `exp09_trainself_newloss.wav` | §4.9/§4.13 — self-diffusion, loss mới | 8.4% | 0.028 | Mel-std/flatness "đẹp nhất" trên giấy nhưng vẫn nghe ra nhiễu — đúng lý do có đính chính này |
| `exp10_distill_newloss_alpha09.wav` | §4.14 — + alpha=0.9 | 82.0% (1 bài); 77.8% (TB 6 bài) | 0.032 | Không vượt được exp06 — xác nhận alpha=0.8 là điểm tối ưu, không phải "cao hơn luôn tốt" |
| `exp11_distill_newloss_alpha08_50ep.wav` | §4.15 — + 50 epoch (thay 25) | 77.0% (1 bài); 85.1% (TB 6 bài) | 0.046 | loss_gt tốt hơn (0.447 vs 0.605) nhưng voiced_ratio thấp hơn exp06 — **CHƯA nghe thử để xác nhận cái nào tự nhiên hơn** |
| `exp12_distill_alpha08_dim384.wav` | §4.15 — model to hơn (dim=384) | 87.0% (1 bài); 66.0% (TB 6 bài) | 0.037 | loss_gt tốt hơn (0.411) nhưng voiced_ratio TB 6 bài thấp nhất trong nhóm mới — **CHƯA nghe thử** |
| `exp13_distill_alpha08_dim512.wav` | §4.15 — model to hơn nữa (dim=512) | 81.7% (1 bài); 79.2% (TB 6 bài) | 0.058 | flatness gần vocal thật nhất (0.058 vs 0.056) trong toàn bộ report — **CHƯA nghe thử** |

*(§4.10 — 75 epoch code cũ — không có file nghe vì chỉ tải log/report lúc đó, không tải
checkpoint, để tiết kiệm băng thông.)*

**Bộ phụ — test loại trừ giả thuyết sampling steps** (§4.9, dùng checkpoint `exp01`, chỉ đổi
số bước Euler lúc sinh): `exp01_distill_oldcode_steps06/16/32/64.wav` — nghe gần như giống
nhau dù 6 vs 64 bước, khớp kết luận "không phải do ít bước sampling" (mel std 1.09→1.11,
không đổi đáng kể).

**Gợi ý thứ tự nghe nếu muốn tiết kiệm thời gian**: `exp00` (mốc) → `exp02` hoặc `exp08`
(ví dụ rõ nhất về "toàn nhiễu") → `exp06` (kết quả tốt nhất hiện tại) — nghe 3 file này đủ
để cảm nhận sự khác biệt voiced_ratio đang đo được.

**Xác nhận bằng nghe thật (2026-07-17)**: sự khác biệt giữa `exp06` và `exp08` nghe rõ ràng,
khớp với voiced_ratio đo được — `exp06` là kết quả khả quan nhất hiện có. Nhưng **ngay cả
`exp06` vẫn chưa nghe ra được lời hát hay giai điệu mạch lạc**, dù voiced_ratio (92.7%) gần
bằng vocal thật (90.1%). Điều này lộ ra hạn chế tiếp theo của `voiced_ratio`: nó chỉ đo "có
một pitch ổn định tại mỗi khung không", không đo (a) pitch đó có nối thành một *giai điệu*
mạch lạc theo thời gian, hay (b) cấu trúc phoneme/formant có đủ rõ để nghe ra *từ ngữ* —
cả hai đều là tầng cấu trúc cao hơn hẳn so với "có tonal hay không". Với quy mô hiện tại
(250 bài, model vài triệu tham số, 1575 step), nhiều khả năng đây là **hạn chế về quy mô
dữ liệu/step thật**, không phải một bug cụ thể còn sót — đã thử lần lượt loss formula, alpha,
dropout, learning rate (§4.11-4.12) và không có hướng nào tạo bước nhảy lớn hơn từ pitch ổn
định lên lời/giai điệu mạch lạc. Hướng khả thi hơn để kiểm chứng thêm (chưa làm, xem §5.2):
mở rộng dữ liệu qua toàn bộ ~1843 bài (không chỉ 250), và/hoặc đo Word Error Rate bằng cách
transcribe lại audio sinh ra với Whisper (đã có sẵn trong pipeline) so với lyric gốc — cho
một con số cụ thể về mức độ "nghe ra từ" thay vì chỉ dựa cảm nhận.

### 4.15 Ablation kích thước model làm lại với loss mới + alpha=0.8 — kết quả đảo ngược §4.9, và một mâu thuẫn chỉ số mới

§4.9 kết luận "tăng kích thước model không giúp gì" — nhưng kết luận đó dùng **code loss cũ**
và đo bằng **mel std** (§4.13 đã chỉ ra mel std không đáng tin). Với loss chống collapse +
`alpha_feature=0.8` đã xác nhận tối ưu (§4.14), câu hỏi này đáng làm lại. Chạy thêm 3 lần
thử, giữ `alpha_feature=0.8`/dataset 250 bài/batch=4 cố định:

- **exp11** (`ddvnam05/genmusic-distill-1784246328`): giữ nguyên kiến trúc exp06
  (`dim=256, depth=4, heads=4`), tăng **epoch 25→50** (không có resume thật — chạy lại từ
  đầu với gấp đôi epoch, vì `train-distill` không hỗ trợ tiếp tục từ checkpoint, khác
  `train-self`).
- **exp12** (`ddvnam05/genmusic-distill-1784252775`): `dim=384, depth=6, heads=6` (khớp
  đúng cấu hình "to hơn" đã thử ở §4.9, để so sánh trực tiếp), 25 epoch.
- **exp13** (`ddvnam05/genmusic-distill-1784253088`): `dim=512, depth=8, heads=8` (to hơn
  nữa), 25 epoch.

| Cấu hình | epoch | loss_gt cuối | voiced_ratio (TB 6 bài) | flatness (TB 6 bài) | silence_ratio (TB 6 bài) |
|---|---|---|---|---|---|
| `dim=256` (exp06, mốc, §4.14) | 25 | 0.605 | **92.4%** | 0.011 | 0.0014 |
| `dim=256` (exp11) | 50 | **0.447** | 85.1% | 0.051 | 0.0206 |
| `dim=384` (exp12) | 25 | **0.411** | 66.0% | 0.039 | 0.0414 |
| `dim=512` (exp13) | 25 | **0.410** | 79.2% | 0.065 | 0.0343 |
| Vocal thật (mốc) | — | — | 74.3% | 0.056 | — |

**Quan sát 1 — `loss_gt` cải thiện rõ theo cả epoch lẫn kích thước, nhưng bão hoà nhanh**:
tăng epoch (0.605→0.447) hoặc tăng size (0.605→0.411) đều giảm loss_gt đáng kể — đảo ngược
kết luận §4.9 ("size không giúp gì"), vì lần này dùng đúng loss đã chống collapse. Nhưng từ
`dim=384` lên `dim=512`, loss_gt gần như không đổi (0.411→0.410) — dấu hiệu bão hoà ở
khoảng 3-8 lần tham số baseline, cùng ngân sách 1575 step.

**Quan sát 2 — nhưng `voiced_ratio` di chuyển NGƯỢC HƯỚNG với `loss_gt` trên cả hai trục**:
cấu hình khớp ground-truth tốt hơn (loss_gt thấp hơn) lại có voiced_ratio **thấp hơn** hẳn
so với exp06 (92.4% → 66-85%), dù vẫn cao hơn hoặc ngang vocal thật (74.3%). Đây là quan hệ
**ngược chiều nhất quán** trên cả trục epoch và trục kích thước — không phải nhiễu ngẫu
nhiên một lần.

**Quan sát 3 — `flatness` lại di chuyển CÙNG HƯỚNG với `loss_gt`, tiến sát giá trị thật**:
flatness của exp06 (0.011) thấp hơn hẳn vocal thật (0.056) — bất thường, gợi ý phổ tần số
quá "sạch"/đơn giản so với giọng hát thật. Ba cấu hình mới đều có flatness tiến gần hơn tới
0.056 (0.051, 0.039, 0.065 — `dim=512` thậm chí vượt qua, hơi nhiễu hơn thật một chút).

**Diễn giải (giả thuyết, chưa kiểm chứng bằng nghe thật)**: `voiced_ratio` cao bất thường
của exp06 có thể không phải dấu hiệu "hát tốt hơn" mà là dấu hiệu **output quá đơn điệu** —
một pitch gần như đứng yên (giữ nguyên một nốt/một hoạ âm suốt cả đoạn) sẽ được `librosa.pyin`
chấm rất cao vì cực kỳ dễ track, nhưng nghe có thể giống một tiếng hum/drone hơn là một giai
điệu thật (khớp với xác nhận nghe thật ở §4.13: exp06 vẫn chưa nghe ra giai điệu mạch lạc dù
voiced_ratio 92.7%). Các cấu hình khớp ground-truth tốt hơn (epoch nhiều hơn hoặc model to
hơn) có thể tạo ra phổ tần số phức tạp/chi tiết hơn, gần với kết cấu thật hơn (flatness gần
0.056 hơn) — nhưng phần chi tiết thêm đó, nếu không phải là pitch ổn định theo đúng cách
`pyin` nhận diện, sẽ bị tính là "kém voiced hơn" dù có thể nghe gần giọng hát thật hơn với
tai người. Đây là **lần đầu tiên trong toàn bộ investigation này hai chỉ số khách quan
(voiced_ratio và độ lệch flatness-so-với-thật) mâu thuẫn nhau về việc cấu hình nào "tốt
hơn"** — đúng dạng vấn đề đã gặp ở §4.13 (metric không đo đúng thứ cần đo), lặp lại ở một
trục khác (size/epoch thay vì distill-vs-self).

**Không thể kết luận chắc bằng số liệu — cần nghe thật**: đã sinh file nghe tương ứng
(`exp11_distill_newloss_alpha08_50ep.wav`, `exp12_distill_alpha08_dim384.wav`,
`exp13_distill_alpha08_dim512.wav`, cùng điều kiện với các file khác trong
`outputs/listening_guide/`, xem bảng cập nhật bên dưới) nhưng **chưa có xác nhận nghe thật
cho 3 file này** — không nên vội kết luận "exp06 vẫn tốt nhất" (theo voiced_ratio) hay
"exp13 tốt hơn" (theo flatness) chỉ từ hai chỉ số đang mâu thuẫn nhau ở đây.

**Giới hạn cần lưu ý**: mỗi cấu hình (epoch/size) trong bảng trên chỉ chạy **một lần** (khác
với ablation alpha ở §4.14 đã có N=6 bài × so sánh đối chứng rõ) — nên xu hướng nhất quán
trên 2 trục độc lập (epoch và size) làm quan sát đáng tin hơn một điểm dữ liệu đơn lẻ, nhưng
vẫn chưa có repeat-run để loại trừ hoàn toàn nhiễu ngẫu nhiên giữa các lần train khác nhau.

**Khuyến nghị**: nghe thử 3 file mới này trước khi quyết định hướng tiếp theo. Nếu nghe thật
xác nhận cấu hình loss_gt thấp hơn (epoch nhiều hơn hoặc model to hơn) nghe **tự nhiên hơn**
dù voiced_ratio thấp hơn, thì kết luận ở §4.14 ("alpha=0.8/dim=256/25ep là cấu hình tốt
nhất") cần được xem lại — có thể voiced_ratio đã hướng lựa chọn tối ưu sai chỗ, về phía một
cấu hình "hát một nốt rất đều" thay vì "hát có giai điệu thật".

---

## 5. Kết luận và hướng phát triển

### 5.1 Kết luận

- Fix có ảnh hưởng lớn nhất trong lịch sử project **không phải** là thay đổi model hay
  huấn luyện — mà là đường render audio. Một model chỉ có thể được đánh giá đúng khi output
  của nó được chuyển thành âm thanh trung thực; trước khi fix, điều này không đúng (0.15
  tương quan log-mel với ground truth — đường output mặc định ra nhiễu có cấu trúc bất kể
  chất lượng model thật ra sao) (§4.1).
- Muốn distillation hoạt động thật cần đọc *đúng source code* của teacher để biết hợp đồng
  gọi, không đoán qua tên biến hay giá trị mặc định của class — `mel_dim=100` mặc định vs.
  `mel_dim=64` thật của checkpoint là ví dụ rõ nhất, và lỗi này lặp lại độc lập ở nhánh
  song song `origin/master` cùng session — một tín hiệu cho thấy đây là cái bẫy tự nhiên
  của loại tích hợp này, không phải lỗi ngẫu nhiên một lần (§4.2).
- **Câu hỏi "distillation có giúp gì không" giờ có câu trả lời thật: có, đo được, ở quy mô
  250 bài/25 epoch/số step khớp nhau (§4.8).** `loss_gt` của `train-distill` (2.57 cuối)
  chỉ bằng khoảng 1/3 của `train-self` (7.15 cuối) trên cùng dữ liệu, và kiểm tra chất
  lượng audio khách quan xác nhận cả hai checkpoint rõ ràng không phải nhiễu, không clip,
  với biên độ output của `train-distill` bám sát audio vocal thật hơn. Đổi lại,
  distillation tốn ~49 lần thời gian GPU của baseline không-teacher — đây là trade-off
  thật, không phải chiến thắng miễn phí.
- Tăng kích thước model (§4.9, code loss cũ) **không** giúp ở cùng ngân sách 1575 step —
  nhưng **§4.15 (loss mới + alpha=0.8) đảo ngược lại**: tăng epoch hoặc kích thước đều giảm
  `loss_gt` rõ rệt (bão hoà sau ~dim=384). Đồng thời phát hiện `voiced_ratio` và
  flatness-so-với-thật **mâu thuẫn nhau** về cấu hình nào "tốt hơn" trên trục này — cần
  nghe thật để phân xử, chưa có kết luận chắc chắn (§4.15).
- Hai bug hạ tầng thật (timeout preprocess giết một run lành mạnh, CUDA OOM do
  fragmentation allocator) được phát hiện và fix trong lúc chạy thực nghiệm ở quy mô đầy
  đủ — không phải bug logic distillation hay hạn chế công suất model (§4.8).
- `train-distill` giờ raise ngay nếu không load được teacher/tokenizer thật, thay vì âm
  thầm hạ cấp về huấn luyện chỉ-ground-truth dưới tên `train-distill` (§4.6) — một
  `train-distill` chạy xong luôn có nghĩa là đã dùng teacher thật.
- **Regression-to-the-mean có nguyên nhân trong literature — nhưng mel-std/flatness KHÔNG
  đủ để xác nhận đã fix, và kết luận chiến lược ban đầu dựa trên 2 chỉ số đó là SAI** (§4.9-
  §4.13; §4.13 là đính chính sau khi nghe thử thật). Mel-variance sụp khớp hiện tượng
  "distributional averaging" (Dieleman 2024, DMD/ADM) — kết hợp mel normalization + loss
  trọng số theo năng lượng + reconstruction/delta + condition dropout đưa `train-self` từ
  std=1.09 lên 3.13 (vượt target 2.95), flatness còn tốt hơn vocal thật. **Nhưng nghe thử
  thật vẫn ra nhiễu** — đo lại bằng `voiced_ratio` (tỉ lệ khung có pitch ổn định, §4.13) lộ
  ra: `train-self` chỉ 8.5% voiced (gần như không có nốt hát nào), còn các bản `train-distill`
  dùng cùng công thức loss mới đạt **82-93% voiced** (khớp vocal thật 90%). **Kết luận đảo
  ngược hoàn toàn so với bản trước**: `train-distill` (với loss mới) mới là hướng cho kết quả
  nghe được, không phải `train-self` — dù rẻ hơn nhiều, `train-self` gần như không tạo ra
  pitch/giai điệu thật nào ở quy mô dữ liệu/step hiện tại. Bài học: mel-std/flatness là proxy
  tốt cho "có phải nhiễu trắng không" nhưng không đo được "có nghe ra hát không" —
  `voiced_ratio` gần đúng hơn nhưng vẫn chỉ là proxy, chưa thay được việc nghe thật.

### 5.2 Hướng phát triển

- **Ưu tiên cao nhất ngay lúc này (§4.15): nghe thử 3 file mới** —
  `exp11_distill_newloss_alpha08_50ep.wav` (50 epoch), `exp12_distill_alpha08_dim384.wav`,
  `exp13_distill_alpha08_dim512.wav` — so với `exp06` (mốc hiện tại). §4.15 phát hiện
  `voiced_ratio` và flatness-so-với-vocal-thật **mâu thuẫn nhau** khi tăng epoch/kích thước:
  loss_gt và flatness đều cải thiện (tiến gần vocal thật hơn) nhưng voiced_ratio giảm so với
  exp06. Chưa rõ đây là do exp06 thật ra "hát" tốt hơn, hay do exp06 chỉ đơn điệu/dễ track-pitch
  hơn (một dạng regression-to-mean nhẹ mà voiced_ratio không bắt được, giống cách mel-std đã
  từng đánh lừa ở §4.13). Kết luận "alpha=0.8/dim=256/25ep là cấu hình tốt nhất" ở §4.14 **có
  thể cần xem lại** sau khi nghe — đừng scale tiếp dựa 100% vào voiced_ratio trước khi có xác
  nhận nghe thật cho nhánh size/epoch này.
- **Nếu nghe thật xác nhận exp06 vẫn tốt nhất**: `alpha_feature≈0.8` + `dim=256` + 25 epoch
  vẫn là mặc định hợp lý; dồn quota vào (a) mở rộng dữ liệu qua toàn bộ ~1843 bài, (b) đo
  Word Error Rate bằng Whisper để đo trực tiếp "nghe ra từ" thay vì suy luận qua voiced_ratio,
  (c) điều tra vì sao job 8 (LR=2e-4, alpha=0.5) lại 0% voiced dù mel-std bình thường.
- **Nếu nghe thật xác nhận epoch/size lớn hơn nghe tự nhiên hơn**: cần một chỉ số khách quan
  mới thay voiced_ratio (ví dụ đo độ biến thiên pitch theo thời gian — pitch contour, không
  chỉ có-pitch-hay-không) trước khi tiếp tục scale mù theo voiced_ratio.
- **Mở rộng dữ liệu thay vì tăng epoch**: §4.10 xác nhận tăng epoch đơn thuần (25→75, code
  cũ) không giúp `loss_gt` cải thiện thêm — chững lại thật, không phải chưa đủ thời gian.
  Hướng tiếp theo hợp lý hơn là mở rộng dữ liệu qua các script của đồng nghiệp
  (`run_kaggle_all_parts.py`, `run_kaggle_multi_part_training.py`) để vượt quy mô
  250-bài/1-part hiện tại (toàn bộ dataset có ~1843 bài theo `--expected-records` của
  script đó) — nên làm với code loss mới (§4.11), chưa thử.
- **Vì sao tín hiệu teacher giữ mel-variance thấp dù trọng số nhỏ?** (§4.12) — giả thuyết
  đáng kiểm chứng: đo trực tiếp variance của `v_teacher` (sau adapter `from_teacher_mel`)
  so với `target_velocity`, xem teacher có tự nhiên "mượt" hơn ground-truth không, độc lập
  với mọi cấu hình phía student.
  - **Đã thử một góc rẻ** (local, không cần Kaggle): literature về mất cân bằng gradient
    trong multi-loss training (GradNorm, ICLR 2018) gợi ý hệ số trộn scalar như
    `alpha_feature` không kiểm soát được ảnh hưởng thật lên optimization — cái quyết định
    là *độ lớn gradient* mỗi loss term tạo ra, có thể khác hẳn tỉ lệ trọng số danh nghĩa.
    Đo thử bằng fake teacher (ngẫu nhiên, không train, dùng lại fixture test có sẵn):
    ratio gradient velocity/gt = 0.53 ở alpha=0.5, giảm xuống 0.13 ở alpha=0.8 — tức
    `alpha_feature` **có** điều chỉnh đúng hướng gradient theo lý thuyết, nhưng thực tế
    trên Kaggle (job 5 vs job 6, teacher thật) lại không thấy std thay đổi tương ứng. Kết
    quả **không xác nhận rõ** giả thuyết mất cân bằng gradient là nguyên nhân — có thể vì
    fake teacher (ngẫu nhiên, không cấu trúc) không phải proxy tốt cho hành vi của teacher
    thật (đã train, có cấu trúc). Cần đo lại đúng với teacher thật mới kết luận được.
  - Nếu xác nhận teacher thật tự nhiên "mượt" hơn ground-truth, hướng fix hợp lý là áp
    `loss_velocity` chỉ ở early training rồi giảm dần (warmup ngược), hoặc chuẩn hóa
    gradient riêng cho 2 loss term (kiểu GradNorm) thay vì chỉ trộn theo `alpha_feature`.
- **Style anchor đại diện hơn**: hiện lấy cố định 10 giây đầu bài cho mọi crop huấn luyện
  của bài đó (§3.1, §3.6) — thử lấy đoạn đại diện hơn (ví dụ đoạn giữa bài) hoặc trung bình
  nhiều đoạn, xem có cải thiện chất lượng không; cần preprocess lại nên cân nhắc quota
  trước khi làm.
- **Vietnamese G2P có dấu thanh + khớp ASR-lyric**: Phần G2P âm vị tiếng Việt đã được kéo thành công bằng cách tích hợp `XPhoneBERT` + `text2phonemesequence` vào mô hình. Phần còn lại (ASR-lyric alignment) và điều kiện pitch/F0 vẫn là các lever tiềm năng cho nghiên cứu tiếp theo.
- **Điều kiện pitch/F0**: từng có, đã bỏ, chưa khôi phục đúng cách (§3.6).
- **Đánh giá bằng người nghe thật**: mọi số liệu trong report này là khách quan (loss,
  spectral flatness, mel std) — chưa có đánh giá nghe chủ quan có hệ thống (MOS/CMOS,
  `src/evaluation/jam_metrics.py` đã có hạ tầng cho việc này nhưng chưa dùng thật).
- **Solver ODE bậc cao/thích ứng** cho CFM sampling (§2.4) — hợp lý khi chất lượng model
  không còn là nút thắt chính.
- **MeanFlow (2025) — đã thử prototype, chưa thành công, không nên tích hợp vội**: literature
  gần đây (MusFlow — flow matching cho music generation, đúng domain; MeanFlow — thay dự
  đoán vận tốc tức thời `v(x_t,t)` bằng vận tốc trung bình `u(x_t,r,t)` tích hợp theo thời
  gian, sinh một-bước) né hẳn kiểu bài toán "khớp một điểm rồi tích phân nhiều bước" mà
  §4.9/§4.12 đang gặp. Đã tự implement + verify một prototype nhỏ (toy 2D, mixture 2 Gaussian,
  local, không tốn quota Kaggle) để kiểm tra trước khi động vào model thật — **không thành
  công**: MSE(u,v) tại r=t (điều kiện biên, đáng lẽ phải ~0 vì công thức suy biến về flow
  matching thường) vẫn ở mức 1.3-1.5 dù đã thử (a) ép một phần batch sample đúng r=t, (b) bỏ
  `torch.no_grad()` quanh lệnh `jvp` (nghi ngờ ban đầu, hoá ra không phải nguyên nhân — kết quả
  giống hệt), (c) tăng step 4000→12000 + gradient clipping (chỉ cải thiện nhẹ). Baseline flow
  matching thường trên **đúng** bài toán/network/hyperparameter đó converge tốt (mean dist
  0.34, gần 0% kẹt giữa) — xác nhận lỗi nằm ở phần MeanFlow-specific (khả năng cao là công
  thức JVP/identity bị hiểu sai ở một chi tiết chưa tìm ra, không phải do thiếu step). **Kết
  luận**: đúng như lo ngại ban đầu — MeanFlow khó implement đúng hơn các fix đã thử trong
  §4.11/§4.12, cần đọc kỹ chính paper (không chỉ tóm tắt qua search) trước khi thử lại. Không
  đưa vào model/pipeline thật ở trạng thái hiện tại.
