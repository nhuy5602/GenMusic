# GenMusic VN — Project Report

Sinh nhạc có lời tiếng Việt, điều kiện theo văn bản (lyric) và âm thanh (backing
track/style), thông qua distillation từ DiffRhythm2. Report này theo cấu trúc một bài
báo khoa học (Giới thiệu → Nghiên cứu liên quan → Phương pháp → Thực nghiệm → Kết luận),
được cập nhật đồng bộ với codebase và với các job Kaggle thật đã chạy — mọi số liệu ở đây
đều lấy từ log/report thật, không suy đoán.

**Trạng thái tại 2026-07-17**: đã chạy thành công distillation thật trên toàn bộ 250 bài
(`sonlest/vietnamese-music-dataset-version3-part6`), so sánh với self-diffusion (không
distill), đánh giá chất lượng khách quan (đa-bài, N=6), và làm lại ablation kích thước model
+ epoch với loss mới + `alpha_feature=0.8` (§4.15-4.16). Phát hiện quan trọng: `voiced_ratio`
(chỉ số đã dùng để chọn "cấu hình tối ưu" ở §4.14) có một góc mù — nó không phân biệt được
"hát có giai điệu" với "giữ đứng yên một nốt"; đo thêm `pitch_std_semitones` (§4.16) cho
thấy cấu hình từng được chọn là tối ưu (`dim=256`, 25 epoch) thực ra đơn điệu nhất trong 4
cấu hình đã thử. Trong lúc đó, đồng nghiệp đổi kiến trúc (sequence-concat + XPhoneBERT,
§4.17) — lần đo đầu tiên trên kiến trúc mới ban đầu tưởng là "đột phá" nhưng hoá ra do BUG
đo lường (load sai frozen text encoder, đã fix tận gốc, §4.18). So sánh cô lập đúng biến
kiến trúc (§4.19, `dim=256/25ep` khớp exp06): kiến trúc mới không cải thiện loss_gt, **chậm
hơn 4.4 lần**, và đổi đặc tính output (voiced_ratio thấp hơn, pitch_std/flatness gần thật
hơn) theo hướng chưa rõ là tốt hơn hay chỉ khác. Đồng nghiệp sau đó phát hiện một bug nghiêm
trọng hơn hẳn: teacher thực ra chạy ở 5 Hz, nhưng mọi lần distill trong report này (kể cả
§4.19) đưa vào teacher chuỗi dài gấp **18.75 lần** phân phối huấn luyện thật của nó (§4.20,
đã fix). Chạy lại đúng cấu hình exp06 với fix này (§4.21) cho kết quả mơ hồ nhất trong toàn
bộ report theo số liệu (flatness khớp gần hoàn hảo vocal thật nhưng voiced_ratio sụp về 0%) —
**đã xác nhận bằng nghe thật (2026-07-17): vẫn toàn nhiễu ở nội dung lời hát**, dù cấu trúc
thời gian (đoạn nào ứng với lời nào) nghe có vẻ đúng. Sau đó, 4 thử nghiệm truy tìm nguyên
nhân gốc (§4.22 — tăng bước sampling, train-self không teacher, so sánh spectrogram, và **cho
chính teacher DiffRhythm2 tự sinh bằng pipeline gốc của nó**) đều loại trừ các giả thuyết bug/
kiến trúc sai, và hội tụ về một kết luận: **teacher tự sinh cho spectrogram có cấu trúc rõ,
giống vocal thật — chứng minh công thức CFM/kiến trúc là đúng — trong khi mọi checkpoint
student (bất kể alpha/size/kiến trúc/có-hay-không-teacher) đều thiếu cấu trúc thời gian đó.**
Nguyên nhân còn lại hợp lý nhất là **quy mô dữ liệu + số step huấn luyện của student quá nhỏ**
(250 bài, 1575-3900 step), không phải một bug còn sót. Xem §5.2 cho hướng tiếp theo (ưu tiên
cao nhất: mở rộng dữ liệu qua ~1843 bài ở phiên có quota mới, không tiếp tục dò hyperparameter
ở quy mô 250 bài hiện tại).

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

Một model dự đoán vận tốc CFM dạng Diffusion-Transformer nhỏ, đã được tái cấu trúc sang cơ chế **Ghép chuỗi thống nhất (Sequence Concatenation)** để khớp hoàn toàn với DiffRhythm2:

- **Điều kiện text**: `vinai/xphonebert-base` đóng băng kết hợp G2P `text2phonemesequence` hỗ trợ tiếng Việt có dấu thanh. Đầu ra là các vector nhúng âm vị thô (không bị kéo giãn hay lặp lại theo khung thời gian).
- **Điều kiện style ("Audio Style Anchor")**: một embedding MuQ-MuLan 512-chiều duy nhất, tính một lần/bài lúc preprocess, chiếu qua `AudioStyleEncoder` (MLP 2 lớp) và cộng trực tiếp vào chuỗi đặc trưng đầu vào.
- **Cơ chế ghép chuỗi (Sequence Concatenation)**: Thay vì kéo giãn text và ghép kênh song song (concat theo chiều đặc trưng), chúng ta ghép nối Text embeddings và Mel embeddings dọc theo **chiều dài chuỗi (sequence dimension)** tại lớp `InputEmbedding`, tạo ra một chuỗi thống nhất có độ dài `text_len + seq_len`.
- **Thông tin Vị trí và Thời gian**: 
  - Vị trí (`pos_ids`) được ghép nối tương ứng: `0..text_len-1` cho text và `0..seq_len-1` cho Mel.
  - Thời gian (`time`) được thiết lập: `-1.0` sentinel cho các token text và timestep `t` thực tế cho các khung Mel. Sau đó, chuỗi thời gian 2D này được đưa qua `TimestepEmbedding` để sinh ra vector nhúng thời gian tương ứng từng vị trí.
- **Backbone**: `depth` block `LlamaDecoderLayer` của HuggingFace hoạt động trên chuỗi thống nhất đã ghép nối, sử dụng **Attention hai chiều đầy đủ (Non-causal bidirectional attention)** thông qua một mặt nạ 4D (`attn_mask_4d`) che đi các token text padding. Lớp residual fusion `text_fusion_linears` cũ đã được loại bỏ hoàn toàn để bám sát thiết kế tối giản của teacher.
- **Đầu ra**: Sau khi đi qua các khối Transformer, mô hình cắt lấy phân đoạn ứng với Mel (`x[:, text_len:]`), điều chế (modulate) qua lớp AdaLN-Zero cuối cùng sử dụng tổng embedding thời gian thực + style, và chiếu ra không gian phổ 100-mel của học sinh.

### 3.3 Huấn luyện với Conditional Flow Matching (`train-self`, `src/training/self_diffusion.py`)

Huấn luyện thuần CFM, không có teacher: với mỗi batch, lấy `vocal_mel` làm `x1`, nhiễu Gauss làm `x0`, nội suy `x_t`, và tối ưu `‖v_student(x_t, t, text, style) - (x_1 - x_0)‖²` (`cfm_loss`, `src/models/cfm_flow.py`). Điều kiện nhạc nền (`backing_mel`) đã được loại bỏ hoàn toàn khỏi hàm loss và mô hình để đảm bảo tính đồng bộ cấu hình với teacher.

### 3.4 Chắt lọc tri thức từ DiffRhythm2 (`train-distill`, `src/training/distill_training.py`)

Nhờ việc chuyển đổi mô hình học sinh sang cơ chế ghép chuỗi thống nhất (Sequence Concatenation), cấu trúc chuỗi đầu vào của Student lúc này đã **khớp toán học hoàn hảo** với cấu trúc chuỗi của Teacher (`text_emb` ở vị trí sentinels và `latent_embed(x_t)` ở vị trí `time=t`). Điều này loại bỏ hoàn toàn sự lệch pha kiến trúc, giúp việc chưng cất tri thức trực tiếp từ các biểu thị ẩn của DiffRhythm2 đạt hiệu quả tối đa mà không cần thông qua các cơ chế căn chỉnh trung gian.

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

> **Lưu ý đọc trước khi vào chi tiết**: §3 (Phương pháp) mô tả kiến trúc **hiện tại** của
> repo (sequence concatenation + XPhoneBERT + additive style conditioning, kể từ
> 2026-07-17 — §4.17). Các thực nghiệm §4.1-4.16 dưới đây chạy trên các phiên bản kiến
> trúc **trước đó** (parallel-channel concat + `xlm-roberta-base` + backing-track
> conditioning thật) — nên một số chi tiết được nhắc tới (`backing_mel`, `chunk_backing`,
> `text_fusion_linears`, `PretrainedRobertaEncoder`) không còn khớp 1:1 với code hiện tại
> của §3. Đây là ảnh chụp lịch sử trung thực của quá trình thực nghiệm, không phải mô tả
> kiến trúc hiện hành — xem §4.17 để biết chính xác điểm chuyển giao.

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
| `exp14_distill_seqconcat_dim384_alpha08_40ep.wav` | §4.18 — kiến trúc mới, dim=384/40ep | 2.1% (1 bài, đo ĐÚNG encoder) | 0.062 | Kiến trúc mới scale lớn — kém hơn hẳn exp06, gần như mất pitch (4/6 bài N=6 không có voiced frame) — **CHƯA nghe thử** |
| `exp15_distill_seqconcat_dim256_alpha08_25ep.wav` | §4.19 — kiến trúc mới, cùng size exp06 | 43.8% (1 bài); 53.5% (TB 6 bài) | 0.033 | So sánh cô lập đúng biến kiến trúc — pitch_std/flatness gần thật hơn exp06 nhưng voiced_ratio thấp hơn, chậm hơn 4.4x — **CHƯA nghe thử** |
| `exp16_distill_vaefix_dim256_alpha08_25ep.wav` | §4.21 — fix VAE-rate 5Hz/18.75x mismatch | **0%** (1 bài + TB 6 bài, cả 2 đều 0%) | **0.0523** (gần khớp tuyệt đối vocal thật 0.056) | Mâu thuẫn chỉ số ở mức cực đại trong cả report — flatness gần hoàn hảo nhưng hoàn toàn không có pitch ổn định — **BẮT BUỘC nghe thử, không đoán được từ số liệu** |

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

### 4.16 Chỉ số mới (rẻ, local, không tốn quota) phân xử được §4.15 mà không cần chờ nghe thật: `pitch_std_semitones` cho thấy exp06 đơn điệu hơn hẳn

§4.15 dừng lại ở "cần nghe thật để phân xử" giữa voiced_ratio (ủng hộ exp06) và flatness
(ủng hộ model to hơn/nhiều epoch hơn). Trước khi chờ nghe thật, thử một chỉ số rẻ hơn:
`librosa.pyin` đã trả về cả **giá trị f0** (không chỉ có-voiced-hay-không) — lấy std của f0
(đổi sang semitone) qua các khung có voiced, đo được **pitch có di chuyển theo thời gian hay
đứng yên một nốt**. Đây chính xác là trục mà `voiced_ratio` **không** đo (nó chỉ hỏi "khung
này có pitch ổn định không", không hỏi "pitch có đổi qua các khung không") — một tiếng hum
giữ nguyên một nốt suốt 8 giây sẽ vẫn có voiced_ratio ~100% nhưng pitch_std ~0.

Đo trên toàn bộ 6 bài đã sinh sẵn ở §4.15 (không cần sinh lại, không tốn quota Kaggle):

| Cấu hình | pitch_std_semitones (TB 6 bài) | Từng bài |
|---|---|---|
| `dim=256, 25ep` (exp06, "tốt nhất" theo voiced_ratio) | **0.908** | 0.70 – 1.05 (rất đều, luôn thấp) |
| `dim=256, 50ep` (exp11) | 1.636 | 1.29 – 2.01 |
| `dim=384, 25ep` (exp12) | 2.615 | 2.06 – 3.07 |
| `dim=512, 25ep` (exp13) | 2.627 | 2.34 – 2.85 |
| Vocal thật (mốc) | **6.393** | — |

**Kết quả rất rõ ràng, nhất quán trên cả 6 bài cho mỗi cấu hình (không phải nhiễu)**: exp06
— cấu hình "thắng" ở §4.14 theo voiced_ratio — có pitch gần như đứng yên (std 0.7-1.05
semitone, chưa bằng 1 nốt nhạc dao động) trên **mọi bài trong cả 6 bài test**, thấp hơn hẳn
3 cấu hình còn lại và chỉ bằng **~14%** độ dao động pitch của vocal thật. Ngược lại, tăng
epoch (exp11) hoặc tăng kích thước model (exp12/13) đều tăng pitch_std theo đúng thứ tự đã
thấy ở loss_gt/flatness (§4.15) — dim=384 và dim=512 đạt ~41% độ dao động pitch thật, gần
gấp 3 lần exp06.

**Kết luận (giờ đã có 3/4 chỉ số đồng thuận, không chỉ 1)**: loss_gt, flatness-so-với-thật,
và giờ pitch_std_semitones **đều nhất quán xếp hạng** exp06 (dim=256, 25 epoch) là cấu hình
**kém melodyc/kém tự nhiên nhất** trong 4 cấu hình đã thử, dù nó có voiced_ratio cao nhất.
Diễn giải hợp lý nhất hiện tại: exp06 đạt voiced_ratio cao bằng cách hội tụ về **một dạng
regression-to-mean tinh vi hơn** những gì mel-std/flatness từng bắt được ở §4.13 — không
phải "toàn nhiễu" (đã fix ở §4.11-4.12) mà là "giữ đúng một nốt/âm sắc ổn định suốt cả đoạn",
điều mà `voiced_ratio` tính là tín hiệu tốt (vì rất dễ track pitch) nhưng thực ra là một dạng
sụp giảm đa dạng khác — giống hiện tượng distributional averaging (Dieleman 2024, §2.4) áp
dụng ở tầng "giai điệu" thay vì tầng "biên độ mel" như lần trước.

**Vẫn cần nghe thật để xác nhận cuối cùng** (pitch_std chỉ là proxy khác, chưa phải sự thật
tuyệt đối — ví dụ một pitch dao động nhiều cũng có thể là dao động hỗn loạn/lạc điệu chứ
chưa chắc là giai điệu mạch lạc), nhưng với 3/4 chỉ số độc lập đồng thuận theo cùng một
hướng, khuyến nghị thực tế đã đủ rõ để hành động ngay: **§4.14 chọn alpha=0.8+dim=256+25ep
làm mặc định dựa hoàn toàn vào voiced_ratio — giờ có bằng chứng khá vững rằng lựa chọn đó
tối ưu nhầm hướng "hát ổn định một nốt" thay vì "hát có giai điệu"**. Cấu hình nên ưu tiên
scale tiếp là **dim=384 hoặc dim=512 (không phải dim=256), cùng alpha=0.8, cùng nhiều epoch
hơn nếu quota cho phép** — cả 3 chỉ số độc lập đều chỉ về hướng này.

`pitch_std_semitones` đã được thêm vào `scripts/evaluate_generation_quality.py` như một
chỉ số chuẩn (song song `voiced_ratio`) cho mọi lần đánh giá tiếp theo.

### 4.17 Đồng nghiệp đổi kiến trúc song song (XPhoneBERT + additive conditioning) — mọi kết quả §4.8-4.16 thuộc kiến trúc CŨ, không so sánh trực tiếp được nữa

Ngay trong lúc phân tích §4.15-4.16, đồng nghiệp push 2 commit kiến trúc lớn gần như đồng
thời (dùng chung repo, đã ghi nhận từ đầu report — xem lưu ý về cộng tác song song):

1. **Đổi text encoder**: `xlm-roberta-base` (subword, đa ngôn ngữ chung) → `vinai/xphonebert-base`
   + `text2phonemesequence` (G2P tiếng Việt ra IPA phoneme trước khi vào encoder). Đây là
   encoder DUY NHẤT trong toàn hệ thống mang ngữ nghĩa lyric tiếng Việt (bản thân tokenizer
   của teacher DiffRhythm2 không hiểu tiếng Việt, chỉ dùng như tín hiệu điều kiện chung chung
   — xem comment gốc trong `distill_training.py`) — nên đây là đòn bẩy hợp lý nhất cho đúng
   vấn đề tồn đọng lâu nhất của report này: "chưa nghe ra lời hát" (§4.13 cuối, §5.2 cũ).
2. **Bỏ hẳn backing-track conditioning, chuyển sang additive style conditioning**: trước đó
   `generate_audio`/`sample_cfm` crop một đoạn backing mel thật và ghép vào mỗi chunk sinh ra;
   giờ tham số `backing_mel` vẫn tồn tại trên chữ ký hàm (không phá vỡ lời gọi cũ) nhưng bị
   bỏ qua hoàn toàn — điều kiện hoá giờ chỉ còn qua style anchor/prompt cộng thẳng vào
   embedding, không còn ghép mel thật nữa.

**Hệ quả trực tiếp cho report này**: mọi số liệu ở §4.8-4.16 (bao gồm toàn bộ kết luận alpha=0.8
và ablation kích thước model) được đo trên kiến trúc **backing-track + XLM-RoBERTa (cũ)** —
không còn tái lập được y hệt bằng code hiện tại của repo, vì `load_checkpoint`/`generate_audio`
giờ luôn đi qua `PretrainedPhonemeEncoder` (phonemize trước khi tokenize, bất kể tên model
truyền vào) và bỏ qua `backing_mel`. Các checkpoint exp06/11/12/13 vẫn load được (không lỗi,
vì text encoder là module đông cứng không lưu trong checkpoint) nhưng **hành vi inference đã
đổi** so với lúc đo — không nên coi các con số §4.8-4.16 là tái lập được 1:1 từ giờ trở đi,
chỉ nên coi là ảnh chụp lịch sử của một kiến trúc đã ngừng tồn tại trên `master`.

**Hai bug hạ tầng phát hiện được khi kiểm tra thay đổi này** (đã fix trước khi chạy tiếp,
đúng tinh thần "luôn kiểm tra trước khi tốn quota" của report này):
- Kernel Kaggle cho `train-distill` chỉ cài `muq` làm dependency bổ sung
  (`scripts/run_kaggle_distill.py`), không cài `text2phonemesequence` — lần chạy thử đầu
  tiên trên kiến trúc mới (`dim=384, alpha=0.8, 40 epoch`) chết ngay với
  `ModuleNotFoundError` sau khi tốn ~8 phút cho bước cài đặt/tải dữ liệu. Đã fix bằng cách
  thêm dòng `pip install text2phonemesequence` vào kernel script.
- Trước khi tin fix đó, đã kiểm tra local (không tốn quota Kaggle): `text2phonemesequence`
  chạy được, sinh phoneme IPA hợp lệ cho câu tiếng Việt mẫu, dù có một lần tải file dictionary
  phụ trợ trả về 404 (không chặn kết quả — model G2P chính vẫn là neural byT5, dictionary chỉ
  là tra cứu bổ sung). Chạy `pytest` (20 test trong `test_cfm_conditioning.py`,
  `test_self_diffusion.py`, `test_model_improvements.py`) — toàn bộ pass. Sau đó chạy thêm một
  smoke test rẻ trên Kaggle thật (5 epoch, `dim=256`, ~5 phút) để xác nhận kiến trúc mới train
  ổn định trên GPU thật trước khi chạy job đầy đủ — loss giảm mượt (1.60→0.91 qua 5 epoch),
  không có dấu hiệu NaN/explosion.

**Quyết định**: chạy tiếp một job đầy đủ trên kiến trúc MỚI (`dim=384, depth=6, heads=6,
alpha_feature=0.8, 40 epoch` — giữ các siêu tham số đã xác nhận tốt ở §4.14-4.16, giờ áp
dụng cho baseline mới) thay vì cố khôi phục/đóng băng kiến trúc cũ để so sánh song song —
việc đồng nghiệp thay text encoder trực tiếp nhắm vào vấn đề "chưa nghe ra lời" quan trọng
hơn việc giữ khả năng so sánh 1:1 với các con số cũ.

### 4.18 Kết quả job đầu tiên trên kiến trúc mới — VÀ một đính chính ngay sau đó: đo lần đầu dùng SAI frozen text encoder, số liệu "đột phá" ban đầu là giả

Job §4.17 (`ddvnam05/genmusic-distill-1784263994`, `dim=384, alpha=0.8, 40 epoch`, kiến trúc
sequence-concatenation + XPhoneBERT + additive conditioning) hoàn tất. `loss_gt` cuối
**0.388** — thấp nhất từng đo trong report này (so với 0.605 của exp06 baseline cũ, 0.411
của exp12 cùng size ở kiến trúc cũ) — nhưng vì kiến trúc/size/epoch đều đổi cùng lúc so với
exp06, không thể quy kết cải thiện này cho riêng một biến số.

**Lần đo N=6 bài đầu tiên (§4.18 bản gốc, ĐÃ SAI — giữ lại đây để minh bạch quá trình,
xem đính chính ngay dưới)** báo cáo `pitch_std_semitones=6.04` (94.5% vocal thật) — được mô
tả (sai) là "phát hiện quan trọng nhất của cả report". Số liệu đó dùng `load_checkpoint()`
với default cứng `roberta_model="xlm-roberta-base"` — nhưng checkpoint này **thực ra được
train với `vinai/xphonebert-base`** (default mới của `MicroDiT`, §4.17), vì
`distill_training.py` tại thời điểm chạy chưa truyền `roberta_model` vào constructor. Vì
text encoder đông cứng không lưu trong checkpoint (chỉ layer `projection` phía sau nó được
train/lưu), việc load sai encoder không crash — nó âm thầm tạo ra embedding text hoàn toàn
khác với những gì `projection` đã học khớp, tương đương với điều kiện text ngẫu nhiên/nhiễu.

**Phát hiện bug qua chính log của lệnh generate**: dòng in ra
`"Loading pretrained XPhoneBERT phoneme encoder: xlm-roberta-base..."` — tên class mới
(XPhoneBERT) nhưng tên model cũ (xlm-roberta-base), một dấu hiệu không nhất quán đủ rõ để
dừng lại kiểm tra ngay trước khi tin số liệu.

**Fix tận gốc (không chỉ vá riêng lần đo này)**: `save_checkpoint`/`arch` giờ lưu thêm
`roberta_model` thật đã dùng lúc train (cả `train-distill` và `train-self`);
`load_checkpoint` đọc lại giá trị này từ chính checkpoint trước khi quyết định load encoder
nào, chỉ dùng `"xlm-roberta-base"` làm fallback cho checkpoint cũ chưa có field này (đúng
hành vi lịch sử của chúng, xem §4.17). `distill_training.py` cũng được sửa để truyền
`roberta_model` tường minh vào `MicroDiT` thay vì lệ thuộc default ngầm của
`dit_transformer.py`. Test suite (19 test, `test_cfm_conditioning.py`/
`test_self_diffusion.py`/`test_model_improvements.py`) chạy lại pass sau fix. Checkpoint
`exp14` được train TRƯỚC fix này nên không tự khai báo được — phải truyền
`roberta_model="vinai/xphonebert-base"` tường minh khi đo lại nó; mọi checkpoint train SAU
fix sẽ tự đúng không cần can thiệp tay.

**Đo lại đúng encoder** (`outputs/eval_multi_seqconcat_dim384_correctencoder/`):

| Chỉ số | Đo SAI encoder (rút lại) | Đo ĐÚNG encoder (thật) | exp06 (mốc §4.14) | Vocal thật |
|---|---|---|---|---|
| voiced_ratio (TB 6 bài) | 20.4% | **2.4%** (4/6 bài = 0%) | 92.4% | 74.3% |
| pitch_std_semitones (TB 6 bài) | 6.04 (giả) | **1.58** (chỉ tính được trên 2/6 bài có voiced frame) | 0.91 | 6.39 |
| spectral flatness (TB 6 bài) | 0.076 | 0.076 (không đổi — flatness là đặc trưng phổ thuần âm học, ít phụ thuộc encoder) | 0.011 | 0.056 |

**Kết luận đã đảo ngược hoàn toàn**: đây KHÔNG phải bước tiến lớn nhất của project — đúng
với giả thuyết "bi quan" đã nêu trong bản gốc §4.18 (nay xác nhận đúng): pitch dao động cao
đo được lúc trước là dao động hỗn loạn do input text bị nhiễu bởi encoder sai, không phải
giai điệu thật. Với đúng encoder, checkpoint này thực ra **kém hơn exp06 rõ rệt** trên cả
voiced_ratio và pitch_std — 4/6 bài không phát hiện được bất kỳ khung voiced nào (giống mức
độ thoái hoá của job 8/exp08, §4.13). File nghe `exp14_distill_seqconcat_dim384_alpha08_40ep.wav`
trong `outputs/listening_guide/` đã được **sinh lại đúng encoder** — bản trước đó dùng sai
encoder, không phản ánh checkpoint thật.

**Bài học phương pháp (thứ ba trong cùng nhánh mâu thuẫn chỉ số)**: sau mel-std (§4.13) và
voiced_ratio (§4.16), lần này lỗi không nằm ở chỉ số đo mà ở **quy trình đo** — một thay đổi
kiến trúc song song (đổi default text encoder, §4.17) tạo ra một cách load-sai-mà-không-crash
rất dễ bỏ sót vì text encoder đông cứng không nằm trong checkpoint. Sau mỗi lần đổi kiến trúc
có liên quan tới encoder/tokenizer, nên **in ra và kiểm tra tên model thật đang load** trước
khi tin bất kỳ số liệu inference nào — không chỉ tin rằng "load không lỗi" nghĩa là "load
đúng".

**Vẫn chưa biết kiến trúc mới (sequence-concatenation) có tốt hơn không** — điểm dữ liệu
duy nhất đo đúng encoder cho tới giờ (exp14) cho kết quả kém, nhưng cũng chưa cô lập được
biến nào (kiến trúc/size/epoch/encoder đều đổi cùng lúc, §4.17) là nguyên nhân. Cần ít nhất
một lần chạy nữa giữ `dim=256/25ep` (khớp exp06) trên kiến trúc mới, đo đúng encoder ngay từ
đầu, mới tách được hiệu ứng kiến trúc khỏi hiệu ứng size/epoch.

### 4.19 So sánh cô lập đúng biến kiến trúc: `dim=256/25ep/alpha=0.8` trên kiến trúc mới vs exp06

Chạy **exp15** (`ddvnam05/genmusic-distill-1784277292`) giữ chính xác cấu hình exp06
(`dim=256, depth=4, heads=4, alpha=0.8, 25 epoch, batch=4`, cùng dataset 250 bài) nhưng trên
kiến trúc MỚI (sequence-concatenation + XPhoneBERT + additive conditioning) — lần này đo
đúng encoder ngay từ đầu (checkpoint tự khai báo `roberta_model` qua fix ở §4.18, xác nhận
qua log: `arch used: {..., 'roberta_model': 'vinai/xphonebert-base'}`, không cần can thiệp
tay). Đây là lần so sánh đầu tiên chỉ đổi đúng 1 biến (kiến trúc) giữa hai đường code.

| Chỉ số | exp06 (kiến trúc cũ, mốc §4.14) | exp15 (kiến trúc mới, cùng size/epoch/alpha) | Vocal thật |
|---|---|---|---|
| loss_gt cuối | 0.605 | **0.600** (gần như không đổi) | — |
| wall-clock | ~1240s | **5481s (+342%)** | — |
| voiced_ratio (TB 6 bài) | 92.4% | **53.5%** (44-60%, đều trên cả 6 bài) | 74.3% |
| pitch_std_semitones (TB 6 bài) | 0.91 | **1.49** (1.39-1.70, đều — cả 6/6 bài đều có voiced frame, khác hẳn exp14's 2/6) | 6.39 |
| spectral flatness (TB 6 bài) | 0.011 | **0.034** | 0.056 |

**Kết luận cô lập được (khác hẳn exp14 — lần này sạch, đáng tin)**: ở đúng cùng ngân sách
step, kiến trúc mới không cải thiện `loss_gt` (gần như y hệt, 0.605→0.600) nhưng **đổi hẳn
đặc tính output** theo đúng hướng đã thấy ở §4.15-4.16 khi tăng size/epoch trên kiến trúc cũ:
`voiced_ratio` giảm (92.4%→53.5%) trong khi `pitch_std`/`flatness` đều tiến gần vocal thật
hơn (0.91→1.49 semitone, 0.011→0.034 flatness) — và **nhất quán trên cả 6/6 bài** (khác
exp14, nơi 4/6 bài không có voiced frame nào — dấu hiệu exp15 ổn định hơn exp14 nhiều, có
thể vì giữ đúng size nhỏ/epoch thấp thay vì scale nhiều biến cùng lúc).

**Chi phí thật đáng kể phải tính vào quyết định**: kiến trúc mới chậm hơn cũ **4.4 lần** ở
cùng cấu hình (5481s vs 1240s) — khả năng do sequence-concatenation làm attention chạy trên
chuỗi dài hơn (text + mel ghép chung thay vì cộng riêng), cộng thêm chi phí G2P
(`text2phonemesequence`) chạy trên CPU mỗi batch. Đây là đánh đổi thật cần cân nhắc: cùng một
lượng epoch giờ tốn gần gấp 5 lần thời gian GPU.

**Tổng hợp 2 điểm dữ liệu trên kiến trúc mới (exp14, exp15) cho tới giờ**: cả hai đều cho
`voiced_ratio` thấp hơn hẳn kiến trúc cũ ở cùng alpha=0.8, nhưng exp15 (cùng size/epoch với
exp06) ổn định và có pitch_std/flatness gần thật hơn rõ rệt, còn exp14 (size/epoch lớn hơn
nhiều, 40ep/dim=384) lại tệ hơn (4/6 bài mất hết voiced frame) — **gợi ý kiến trúc mới có
thể nhạy với size/epoch theo cách khác kiến trúc cũ**, không đơn giản là "cứ scale lên sẽ tốt
hơn" như đã quan sát ở §4.15-4.16 cho kiến trúc cũ. Chưa đủ dữ liệu để kết luận chắc — mới có
2 lần chạy, chưa lặp lại lần nào để loại trừ nhiễu ngẫu nhiên giữa các lần train (đúng bài
học N=1 đã nêu ở §4.14).

**Vẫn cần nghe thật để đưa ra khuyến nghị cuối cùng**: file
`outputs/listening_guide/exp15_distill_seqconcat_dim256_alpha08_25ep.wav` đã sẵn sàng, cùng
điều kiện chuẩn với mọi file khác trong bộ nghe thử — nên nghe cùng lúc với `exp06` và `exp14`
để so sánh cả 3 hướng (kiến trúc cũ tốt nhất đã biết, kiến trúc mới cùng size, kiến trúc mới
scale lớn hơn).

**Cập nhật quan trọng (xem §4.20 ngay dưới)**: exp06/exp14/exp15 ở trên đều chạy TRƯỚC một
phát hiện lớn hơn hẳn của đồng nghiệp — teacher thực ra vận hành ở tần số thời gian 5 Hz
(không phải mel thường), trong khi mọi lần distillation trong report này (kể cả exp15) đưa
vào teacher một chuỗi dài gấp **18.75 lần** phân phối huấn luyện thật của nó. So sánh
kiến trúc ở §4.19 vẫn hợp lệ nội tại (cả hai vế đều có cùng lỗi này, nên vẫn là so sánh công
bằng giữa "kiến trúc cũ" và "kiến trúc mới" dưới cùng một tín hiệu teacher bị lệch), nhưng
**không phản ánh tiềm năng thật của distillation** — cả hai phía đều bị giới hạn bởi cùng
một bug nghiêm trọng hơn nhiều so với bất kỳ biến số nào đã thử (alpha/size/kiến trúc). Xem
§4.20 để biết chi tiết và trạng thái fix.

### 4.20 Phát hiện sai lệch cốt lõi về bản chất dữ liệu (VAE Latent vs. Mel-spectrogram) và tần số thời gian (5 Hz vs. 93.75 Hz) giữa Teacher và Student

Trong quá trình rà soát tài liệu thiết kế và source code của DiffRhythm2, một sai lệch cấu trúc cực kỳ lớn giữa Teacher và Student đã được phát hiện:
1. **Bản chất dữ liệu ẩn (VAE Latents vs. Mel-spectrogram)**: Teacher DiffRhythm2 thực chất hoạt động trên không gian ẩn (latents) 64 chiều của một bộ **Music VAE** tự thiết kế (dựa trên Stable Audio 2 VAE), chứ không phải phổ Mel-spectrogram 64-kênh thông thường.
2. **Lệch pha tần số thời gian (5 Hz vs. 93.75 Hz)**: Bộ Music VAE của Teacher nén tín hiệu âm thanh cực mạnh xuống tần số thời gian chỉ **5 Hz** (5 khung/giây). Trong khi đó, Student hoạt động trực tiếp trên phổ Mel-spectrogram chuẩn (hop_length=256 ở 24kHz) với tần số thời gian **93.75 Hz** (gần 94 khung/giây).
3. **Cơ chế chưng cất bị lệch phân phối (Out-of-Distribution)**: 
   * Hàm `_teacher_velocity` trong `distill_training.py` hiện tại chỉ thực hiện nội suy tuyến tính kênh tần số từ 100 về 64 kênh (`_resize_mel_bins`), nhưng **giữ nguyên trục thời gian** (ví dụ giữ nguyên chuỗi 375 khung ứng với 4 giây thay vì nén về 20 khung tương thích với 5Hz).
   * Việc đẩy một chuỗi dài gấp 18.75 lần so với phân phối huấn luyện vào khối Llama của Teacher khiến bộ mã hóa vị trí quay (RoPE) và Attention mask của Teacher cho ra kết quả dự đoán vận tốc bị sai lệch nghiêm trọng, làm mất đi ý nghĩa hướng dẫn của chưng cất tri thức.
4. **Cơ chế Attention Mask theo khối**: Teacher được huấn luyện theo cơ chế ghép chuỗi clean-noisy tự hồi quy theo từng khối (`block-by-block autoregressive attention mask`), trong khi Student hiện đang dùng attention bidirectional không nhân quả hoàn toàn trên toàn chuỗi phổ Mel.

Phát hiện này giải thích lý do thực sự tại sao chưng cất (`train-distill`) trước đây cho kết quả thoái hóa hoặc không vượt trội hẳn so với tự học (`train-self`). Hướng sửa đổi bắt buộc phải bao gồm việc đồng bộ hóa tần số thời gian và không gian đặc trưng giữa hai mô hình trước khi chưng cất.

#### Trạng thái sửa đổi và Mức độ ưu tiên của các điểm lệch pha:

1. **Đồng bộ tần số thời gian và Lệch phân phối (Mục 2 & 3)**:
   * *Trạng thái*: **ĐÃ FIX**. Đã chèn các bước downsample và upsample trục thời gian (`_resample_time_dimension`) vào `_teacher_velocity`. Teacher hiện nhận đúng chuỗi 5 Hz theo phân phối huấn luyện và trả về vận tốc dự đoán chính xác, sau đó được upsample lại 93.75 Hz cho Student.
   * *Đánh giá*: Loại bỏ hoàn toàn lỗi nghiêm trọng Out-of-Distribution ở attention/RoPE của Teacher.

2. **Cơ chế Attention Mask theo khối (Mục 4)**:
   * *Trạng thái*: **ĐÃ FIX**. Đã hiện thực hóa hàm `_build_block_attn_mask` để tạo mặt nạ attention tự hồi quy theo từng block (mặc định size = 10 của Teacher) và gộp chuỗi đầu vào thành dạng `(S, L, Z, Zt)` gồm cả Clean Latent $Z$ và Noisy Latent $Zt$.
   * *Đánh giá*: Sửa đổi này giúp Teacher tính toán attention chính xác theo đúng cấu trúc tự hồi quy mà nó được huấn luyện, ngăn chặn việc rò rỉ thông tin tương lai trong lúc chưng cất tri thức.

3. **Bản chất dữ liệu ẩn VAE Latent vs. Mel-spectrogram (Mục 1)**:
   * *Trạng thái*: **TẠM HOÃN / GIỮ NGUYÊN THIẾT KẾ**. Mô hình Student vẫn sinh trực tiếp phổ Mel 100 chiều để giải mã qua Vocos (không dùng VAE riêng cho Student). Sự lệch pha kênh đặc trưng 64-latent vs 100-mel được giải quyết thông qua adapter tuyến tính có thể huấn luyện (`from_teacher_mel`).
   * *Mức độ cần fix*: **THẤP**. Do Vocos mang lại chất lượng tái tạo âm thanh vượt trội mà không cần huấn luyện lại bộ Music VAE cực kỳ tốn tài nguyên tính toán. Đây là một trade-off kiến trúc hợp lý và tối ưu trong điều kiện giới hạn phần cứng.

**Ý nghĩa đối với mọi kết luận trước đó trong report này**: fix này (đã merge vào `master`
sau exp06/14/15) làm lại đúng tín hiệu velocity của teacher lần đầu tiên trong toàn bộ
project — mọi so sánh alpha_feature (§4.14), kích thước model (§4.15-16), và kiến trúc
(§4.19) đều được đo dưới một teacher signal bị lệch phân phối nghiêm trọng (input dài gấp
18.75 lần bình thường). Đây có thể là lời giải thích thật cho "khoảng cách còn lại" mà §4.12
từng nêu là "giả thuyết chưa kiểm chứng đầy đủ" (teacher tự nhiên đưa dự đoán mượt hơn) —
khả năng cao đó không phải do bản chất teacher mà do chính bug này. **Chưa có bất kỳ lần
train nào trong report tính đến giờ dùng đúng fix này** — bước tiếp theo hợp lý nhất là chạy
lại `train-distill` ở quy mô nhỏ (khớp exp06, `dim=256/25ep/alpha=0.8`) với code đã fix, đo
lại đủ bộ 3 chỉ số (voiced_ratio/flatness/pitch_std_semitones) trước khi tin bất kỳ kết luận
cũ nào về alpha/size/kiến trúc.

### 4.21 Kết quả lần train đầu tiên với fix VAE-rate — kết quả mơ hồ nhất trong toàn bộ report, mâu thuẫn chỉ số ở mức cực đại

Chạy **exp16** (`ddvnam05/genmusic-distill-1784288826`) — đúng cấu hình exp06/exp15
(`dim=256/25ep/alpha=0.8`) nhưng với fix VAE-rate + block-attention-mask của §4.20 đã áp
dụng. `loss_gt` cuối **0.572** (gần exp06's 0.605, exp15's 0.600 — không đổi nhiều), nhưng
wall-clock **4029s** — nhanh hơn exp15 (5481s, chưa fix) đúng như dự đoán, vì teacher giờ chỉ
xử lý chuỗi ngắn 5Hz thay vì chuỗi dài 93.75Hz đầy đủ.

**Đo N=6 bài** (`outputs/eval_multi_vaefix_dim256/`):

| Chỉ số | exp06 (mốc §4.14) | exp15 (kiến trúc mới, chưa fix VAE-rate) | **exp16 (đã fix VAE-rate)** | Vocal thật |
|---|---|---|---|---|
| voiced_ratio (TB 6 bài) | 92.4% | 53.5% | **0.0%** (0/6 bài) | 74.3% |
| pitch_std_semitones | 0.91 | 1.49 | **None** (không có voiced frame) | 6.39 |
| spectral flatness | 0.011 | 0.034 | **0.0556** | 0.0558 |
| RMS | — | — | 0.058-0.060 (bình thường, không im lặng) | 0.08-0.10 |

**Kết quả cực đoan theo cả hai hướng cùng lúc**: flatness của exp16 (0.0556) là **gần khớp
tuyệt đối** với vocal thật (0.0558, sai biệt <0.5%) — chỉ số tốt nhất đo được trong TOÀN BỘ
report này trên trục này. Nhưng voiced_ratio rơi về **đúng 0%** trên **cả 6/6 bài** — không
tệ hơn một chút như exp14/exp15, mà **tuyệt đối không có** một khung nào được `pyin` nhận
diện là có pitch ổn định. RMS vẫn ở mức bình thường (không im lặng, không clip) — loại trừ
khả năng output chỉ là silence/degenerate đơn giản.

**Không thể diễn giải chỉ bằng số liệu — đây là trường hợp mơ hồ nhất trong report**:
1. **Khả dĩ 1 (tích cực)**: audio giờ có kết cấu phổ tần số rất giống giọng hát thật
   (flatness khớp gần hoàn hảo) — có thể là giọng hát/nói có thật nhưng theo kiểu không giữ
   một nốt ổn định lâu (hát nhanh, nhiều âm tiết, hoặc gần với nói hơn hát) — hợp lý một phần
   vì ngay cả vocal thật trong dataset này cũng chỉ có 74.3% voiced_ratio (không phải 100%),
   tức giọng hát thật cũng có nhiều đoạn `pyin` không track được pitch ổn định (phụ âm,
   chuyển âm, hơi thở) — có thể exp16 tạo ra kết cấu gần giống nhưng tỉ lệ đoạn "khó track"
   cao hơn hẳn.
2. **Khả dĩ 2 (tiêu cực)**: đây là một dạng thoái hoá hoàn toàn khác — âm thanh có kết cấu
   phổ giống giọng hát về mặt thống kê tổng thể (flatness là chỉ số trung bình toàn cục) nhưng
   không có bất kỳ cấu trúc pitch/giai điệu nào thật — tương tự nhiễu có màu (colored noise)
   được thiết kế tình cờ khớp phân phối phổ của giọng hát mà không mang nội dung âm nhạc nào.

**Bắt buộc phải nghe thật để phân xử — không có lựa chọn khác lần này**: cả 2 khả dĩ trên đều
hoàn toàn tương thích với số liệu đo được, và không giống các trường hợp mơ hồ trước (§4.16,
§4.18 cuối cùng đều giải quyết được nhờ có ít nhất một chỉ số đồng thuận rõ) — lần này 2 chỉ
số quan trọng nhất mâu thuẫn ở mức tuyệt đối (flatness ~hoàn hảo vs voiced_ratio ~hoàn toàn
sụp). File: `outputs/listening_guide/exp16_distill_vaefix_dim256_alpha08_25ep.wav`.

**Không nên vội kết luận fix VAE-rate ở §4.20 "không hiệu quả" hay "đã giải quyết vấn đề"**
chỉ từ bảng số liệu này — cần nghe thật trước. Nếu nghe thật xác nhận khả dĩ 1 (giọng hát/nói
thật, chỉ thiếu ổn định pitch), đây là bước tiến very đáng kể, và hướng tiếp theo là cải thiện
độ ổn định pitch (có thể qua nhiều epoch hơn, hoặc điều kiện F0 rõ ràng hơn — xem §5.2, mục
điều kiện pitch/F0 chưa khôi phục). Nếu xác nhận khả dĩ 2, fix VAE-rate dù đúng về mặt lý
thuyết vẫn chưa đủ để tạo tín hiệu distillation hữu ích ở quy mô 250 bài/25 epoch hiện tại.

**Xác nhận bằng nghe thật (2026-07-17)**: người dùng nghe trực tiếp `exp16` và xác nhận
**khả dĩ 2** — "vẫn khá nhiễu... những đoạn lời thì toàn nhiễu". Vậy flatness gần-hoàn-hảo đo
được không phản ánh chất lượng giọng hát thật; đó là một chỉ số toàn cục (trung bình phổ tần
số trên cả file) có thể trùng khớp thống kê với vocal thật mà không mang nội dung âm nhạc nào
— đúng dạng "colored noise" đã lo ngại, không phải "giọng hát thiếu ổn định pitch". Đây là
lần thứ 3 trong report này một chỉ số khách quan (mel-std §4.13, voiced_ratio §4.16, giờ
flatness §4.21) tạo cảm giác "gần đạt target" nhưng nghe thật lại phủ nhận — xác nhận thêm
quy tắc đã rút ra: **không một chỉ số đơn lẻ nào trong bộ 3 hiện có là đủ để tin cậy hoàn
toàn, kể cả khi nó khớp gần tuyệt đối với vocal thật**.

**Nhưng có một tín hiệu tích cực cụ thể, đáng ghi nhận riêng**: người dùng ghi nhận "có cảm
giác đã được căn chỉnh lời" — nghĩa là **cấu trúc thời gian** của các đoạn lời (khi nào có
tiếng hát, khi nào là nhạc đệm/im lặng, theo đúng nhịp của lyric timing) nghe có vẻ đúng vị
trí, dù *nội dung* âm thanh tại các đoạn đó vẫn là nhiễu. Đây là tách biệt quan trọng: vấn đề
còn lại nhiều khả năng nằm ở **chất lượng sinh audio tại từng đoạn** (velocity field/pitch/
timbre), không phải ở **cấu trúc thời gian tổng thể** (đoạn nào ứng với lời nào) — hai lớp
vấn đề khác nhau, và lớp thứ hai (cấu trúc thời gian) có vẻ đã được giải quyết tốt hơn qua
các fix tích lũy (lyric_timing, block-attention mask §4.20, v.v.), trong khi lớp thứ nhất
(chất lượng nội dung mỗi đoạn) vẫn là nút thắt chính.

**Kết luận thực tế cho §4.20-4.21**: fix VAE-rate là một fix đúng và cần thiết về lý thuyết
(loại bỏ một lỗi out-of-distribution có thật), nhưng **một mình nó không đủ** để giải quyết
vấn đề "toàn nhiễu" ở lời hát — vẫn cần điều tra thêm ở tầng chất lượng sinh audio (có thể là
quy mô dữ liệu/model vẫn còn nhỏ, §4.13's giả thuyết cũ; hoặc cần điều kiện F0/pitch rõ ràng
hơn thay vì chỉ dựa vào teacher signal; hoặc một hạn chế khác chưa phát hiện). Không nên coi
đây là "đã giải quyết được noise" — vẫn còn nguyên vấn đề cốt lõi ban đầu của cả project.

### 4.22 Truy tìm nguyên nhân gốc: có phải bug, hay đúng là do quy mô? Ba thử nghiệm rẻ, cùng ngày, cùng chỉ về một hướng

Sau §4.21, câu hỏi đặt ra: liệu còn một bug cụ thể chưa tìm ra, hay đây là hạn chế thật về
cách tiếp cận (ví dụ: student sinh mel-spectrogram + Vocos trong khi teacher "nói" bằng VAE
latent riêng — một giả thuyết hợp lý được đề xuất trong lúc thảo luận)? Ba thử nghiệm rẻ
(không tốn nhiều quota Kaggle, chủ yếu chạy local) được thực hiện cùng lúc để trả lời:

**1) Tăng số bước sampling (6/16/30/40/64) trên exp16 — loại trừ giả thuyết "chưa hội tụ"**:

| steps | 6 | 16 | 30 | 40 | 64 |
|---|---|---|---|---|---|
| voiced_ratio | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| flatness | 0.0554 | 0.0523 | 0.0513 | 0.0510 | 0.0507 |

voiced_ratio **giữ đúng 0% ở MỌI mức step** — tăng 10x số bước tích phân ODE không thay đổi
gì. Xác nhận lại đúng kết luận cũ của §4.9 (test 6 vs 64 bước trên checkpoint đời trước) —
vấn đề nằm ở chính velocity field đã học, không phải cách tích phân lúc sinh.

**2) train-self (KHÔNG dùng teacher) trên code hiện tại đầy đủ fix — kiểm tra trực tiếp giả
thuyết Vocos/mel vs VAE-latent**: nếu giả thuyết đó đúng, bỏ hẳn teacher đi (loại bỏ hoàn
toàn nguồn gây lệch pha Vocos/VAE-latent) phải cho kết quả tốt hơn rõ rệt. Chạy **exp17**
(`ddvnam05/genmusic-train-1784297100`, `dim=256/25ep`, cùng dataset, code hiện tại — mel
normalization + loss chống collapse + kiến trúc sequence-concat + XPhoneBERT, chỉ khác đúng
1 biến: không có teacher):

| Chỉ số | exp16 (train-distill, có fix VAE-rate) | **exp17 (train-self, KHÔNG teacher)** | Vocal thật |
|---|---|---|---|
| voiced_ratio (TB 6 bài) | 0.0% (0/6) | **0.0% (0/6)** | 74.3% |
| pitch_std_semitones | None | **None** | 6.39 |
| spectral flatness | 0.0556 | **0.0650** | 0.0558 |
| wall-clock | 4029s | **7415s** (!) | — |

**Kết quả giống hệt nhau về mặt định tính** — cả hai đều 0% voiced trên toàn bộ 6 bài. Bỏ
hẳn teacher đi (loại bỏ hoàn toàn khả năng lệch pha Vocos/VAE-latent) **không** thay đổi bản
chất kết quả. Đây là bằng chứng khá mạnh **bác bỏ** giả thuyết "vấn đề nằm ở tín hiệu teacher
bị lệch không gian đặc trưng" — vì thất bại xảy ra y hệt cả khi không có teacher. Một phát
hiện phụ đáng chú ý: **train-self giờ CHẬM HƠN train-distill** (7415s vs 4029s) — đảo ngược
hoàn toàn kinh tế học đã ghi nhận ở §4.8 (train-self rẻ hơn 49 lần) — vì text encoder mới
(XPhoneBERT + G2P neural mỗi batch) tạo overhead cố định mà trước đây bị chi phí teacher che
lấp; khi không có teacher, chi phí đó lộ ra và trở thành nút thắt chính.

**3) So sánh trực quan spectrogram — vocal thật vs. 4 checkpoint (exp06/15/16/17)**:
`docs/assets/spectrogram_comparison.png`. Vocal thật cho thấy **cấu trúc rõ ràng**: dải hài
âm sắc nét (harmonic banding) khi có tiếng hát, xen giữa các **khoảng lặng thật** (hơi thở,
ngắt câu). Cả 4 checkpoint sinh ra — **kể cả exp17 không dùng teacher** — đều cho năng lượng
**liên tục, không đứt đoạn, trải khắp mọi tần số** suốt cả file, không có khoảng lặng, không
có dải hài âm tách biệt. exp06 có một dải hẹp sáng gần 0Hz (khớp giả thuyết "hát một nốt đơn
điệu", §4.16); exp15/16/17 trải năng lượng rộng hơn (khớp flatness gần thật hơn) nhưng vẫn
không có cấu trúc thời gian nào giống giọng hát thật.

**Tổng hợp 3 thử nghiệm — cùng chỉ về một hướng**: không phải sampling chưa hội tụ (thử 1),
không phải riêng tín hiệu teacher/Vocos-VAE mismatch (thử 2 — train-self cũng fail y hệt),
và về mặt thị giác thì THIẾU CẤU TRÚC THỜI GIAN (không có khoảng lặng, không có onset/offset
rời rạc như hát thật) là đặc điểm chung của MỌI checkpoint, không phân biệt có/không teacher,
cũ/mới kiến trúc (thử 3). Điều duy nhất KHÔNG đổi giữa tất cả các thử nghiệm trong toàn bộ
report — từ đầu tới giờ — là **quy mô dữ liệu (250 bài) và số step huấn luyện (1575-3900
step)**. Đây là biến duy nhất chưa từng được thử thay đổi thật, và là ứng viên hợp lý nhất
cho nguyên nhân gốc còn lại, dựa trên loại trừ toàn bộ các giả thuyết khác đã kiểm chứng được
trong report này.

**4) Điểm dữ liệu quyết định: cho chính teacher DiffRhythm2 tự sinh nhạc bằng pipeline gốc
của nó** (`scripts/test_teacher_inference.py` — CFM `sample_block_cache`, 16 bước, decoder
BigVGAN riêng của teacher, không đi qua student/Vocos ở bất kỳ bước nào), cùng câu lyric
tiếng Việt, chạy trên Kaggle GPU (`ddvnam05/genmusic-teachertest-1784305867`, ~vài phút, chi
phí rất nhỏ vì chỉ inference không train). Xem hàng thứ 2 trong
`docs/assets/spectrogram_comparison.png`.

**Kết quả: teacher tự sinh cho spectrogram có cấu trúc rõ ràng** — dải hài âm sắc nét (đặc
biệt dưới 1024Hz) và các khoảng lặng phân biệt rõ giữa các câu (quanh giây 1, 3-3.5, 4.5-5,
6) — về hình dạng tổng quát **giống vocal thật hơn hẳn** so với bất kỳ checkpoint nào của
student (exp06/15/16/17), dù chỉ là bản kiểm tra nhanh (16 bước, không tinh chỉnh riêng cho
tiếng Việt). Đây là bằng chứng quyết định: **bản thân "công thức" (CFM + backbone kiểu Llama
+ điều kiện style MuLan) hoàn toàn có khả năng sinh ra âm thanh có cấu trúc, giống giọng hát
thật** — không phải một cách tiếp cận sai về nguyên lý. Sự khác biệt nằm ở quy mô: teacher có
~1.14 tỉ tham số và (theo giả định hợp lý) được huấn luyện ở quy mô dữ liệu/step gấp nhiều
bậc so với student (~vài triệu tham số, 250 bài, 1575-3900 step).

**Kết luận tổng hợp cho §4.22 (trả lời trực tiếp câu hỏi "sai bug hay sai hướng?")**: dựa
trên toàn bộ 4 thử nghiệm — không phải do sampling chưa hội tụ, không phải do riêng tín hiệu
teacher/Vocos-VAE mismatch (train-self cũng fail y hệt), không phải do cách tiếp cận CFM/
kiến trúc sai về nguyên lý (chính teacher dùng cùng công thức và sinh ra được audio có cấu
trúc) — nguyên nhân còn lại hợp lý nhất, sau khi loại trừ, là **quy mô dữ liệu + số step huấn
luyện của student quá nhỏ** so với độ phức tạp của tác vụ (sinh nhạc + lời hát tiếng Việt có
cấu trúc thời gian thật). Đây không phải một bug có thể fix bằng một dòng code — mà là giới
hạn tài nguyên (quota GPU miễn phí, §1) đã biết từ đầu project, giờ có bằng chứng thực nghiệm
cụ thể hơn hẳn so với suy đoán ban đầu.

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
  nhưng **§4.15-4.16 (loss mới + alpha=0.8) đảo ngược lại**: tăng epoch hoặc kích thước đều
  giảm `loss_gt` rõ rệt (bão hoà sau ~dim=384), đưa flatness gần vocal thật hơn, VÀ tăng
  pitch_std_semitones (độ dao động giai điệu) gần gấp 3 lần so với baseline dim=256. Ba chỉ
  số độc lập đều đồng thuận: cấu hình "tối ưu" ở §4.14 (dim=256, chọn theo voiced_ratio) hoá
  ra có pitch gần như đứng yên một nốt — một dạng regression-to-mean tinh vi hơn ở tầng giai
  điệu mà voiced_ratio không bắt được (§4.16).
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

- **Ưu tiên cao nhất (§4.22, kết luận sau khi loại trừ hệ thống): nguyên nhân gốc nhiều khả
  năng là QUY MÔ DỮ LIỆU/STEP của student, không phải một bug cụ thể hay cách tiếp cận sai.**
  4 thử nghiệm rẻ cùng ngày đều chỉ về hướng này: tăng bước sampling không giúp gì (loại trừ
  "chưa hội tụ"); train-self (không teacher) fail y hệt train-distill (loại trừ "tín hiệu
  teacher/Vocos-VAE mismatch"); so sánh spectrogram cho thấy MỌI checkpoint student (dù kiến
  trúc/teacher khác nhau) đều thiếu cấu trúc thời gian (không khoảng lặng, không dải hài âm)
  mà vocal thật có; và **quan trọng nhất — chính teacher DiffRhythm2 tự sinh (không qua
  student) lại cho ra spectrogram có cấu trúc rõ, giống vocal thật** — chứng minh "công thức"
  CFM+Llama-backbone+MuLan hoàn toàn khả thi, chỉ là student (~vài triệu tham số, 250 bài,
  1575-3900 step) chưa đủ quy mô để tái tạo nó. Không nên tiếp tục dò thêm hyperparameter ở
  quy mô dữ liệu hiện tại — bằng chứng cho thấy các lever đó đã cạn tác dụng. Hướng thật cần
  làm ở phiên tiếp theo (quota mới): **mở rộng dữ liệu qua toàn bộ ~1843 bài** (chưa thử ở bất
  kỳ hướng nào trong report này — luôn chỉ dùng 250 bài) kết hợp tăng số epoch/step thật sự,
  không phải thử thêm alpha/kiến trúc ở cùng 250 bài.
- Các hướng bổ sung (ưu tiên thấp hơn mở rộng dữ liệu): (a) khôi phục điều kiện pitch/F0 rõ
  ràng (§3.6, đã bỏ khi thêm Audio Style Anchor, chưa quay lại — đánh giá feasibility cho thấy
  cần 3-5+ giờ code + có thể phải chạy lại Demucs, không làm trong ngân sách quota hẹp), (b) đo
  Word Error Rate bằng Whisper để có con số cụ thể thay vì chỉ "nghe thấy nhiễu" — script đã
  viết (`scripts/measure_wer.py`-style, chưa hoàn tất do treo khi tranh CPU với việc khác,
  cần chạy lại độc lập không cạnh tranh tài nguyên).
- **Kiến trúc mới (sequence-concat + XPhoneBERT, §4.17-4.19) chưa chứng minh được lợi ích rõ
  ràng và tốn chi phí thật (chậm hơn 4.4 lần cùng cấu hình)** — chưa đủ dữ liệu (chỉ 2-3 lần
  chạy N=1/cấu hình) để kết luận chắc nó tốt hơn hay kém hơn kiến trúc cũ; không nên đầu tư
  thêm quota vào việc scale kiến trúc này trước khi giải quyết được vấn đề chất lượng nội dung
  cốt lõi ở trên, vì bất kỳ cải thiện nào cũng sẽ bị vấn đề nhiễu che lấp trong đánh giá nghe
  thật.
- **Ưu tiên cao nhất (§4.16): dùng `dim=384` hoặc `dim=512` (không phải `dim=256`) làm mặc
  định tiếp theo, cùng `alpha_feature=0.8`, và tăng epoch khi quota cho phép.** §4.14 từng
  chọn `dim=256/25ep` (exp06) làm "tối ưu" hoàn toàn dựa vào voiced_ratio (92.4%) — nhưng
  §4.16 đo thêm `pitch_std_semitones` (độ dao động giai điệu, không phải chỉ "có pitch hay
  không") và phát hiện exp06 gần như hát đứng yên một nốt (0.91 semitone, ~14% vocal thật),
  trong khi `dim=384`/`dim=512` dao động gần gấp 3 (2.6 semitones, ~41% vocal thật). Cả 3
  chỉ số độc lập (loss_gt, flatness-so-với-thật, pitch_std) đều đồng thuận theo hướng model
  to hơn — vẫn nên nghe thử `outputs/listening_guide/exp11-13_*.wav` để xác nhận cuối cùng
  trước khi coi đây là kết luận chắc chắn, nhưng không nên tiếp tục dùng `dim=256` làm mặc
  định trong lúc chờ.
- **Mở rộng dữ liệu + kết hợp scale-up**: dồn quota vào (a) mở rộng dữ liệu qua toàn bộ
  ~1843 bài (không chỉ 250) kết hợp `dim=384`/`dim=512` + `alpha=0.8`, (b) đo Word Error Rate
  bằng Whisper để đo trực tiếp "nghe ra từ" thay vì suy luận qua voiced_ratio/pitch_std, (c)
  điều tra vì sao job 8 (LR=2e-4, alpha=0.5) lại 0% voiced dù mel-std bình thường.
- **Bài học phương pháp cần giữ cho mọi ablation tiếp theo**: đừng chọn "cấu hình tốt nhất"
  chỉ dựa một chỉ số duy nhất — mel-std (đã sai ở §4.13), rồi voiced_ratio (giờ có dấu hiệu
  sai lệch tinh vi hơn ở §4.16) đều lần lượt là "chỉ số tin tưởng nhất" tại một thời điểm rồi
  sau đó lộ ra góc mù riêng. Nên luôn đo tối thiểu bộ ba (flatness, voiced_ratio,
  pitch_std_semitones — cả ba đã có sẵn trong `scripts/evaluate_generation_quality.py`) và
  chỉ tin một kết luận khi cả ba đồng thuận, thay vì tối ưu theo đúng 1 con số.
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

---

## Tài liệu tham khảo

- Lipman, Y. et al. (2022). *Flow Matching for Generative Modeling*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747). — Cơ sở CFM/rectified flow dùng bởi cả teacher và student (§2.4).
- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531). — Khuôn mẫu knowledge distillation gốc mà `alpha_feature`/velocity-matching của project dựa theo (§2.5).
- Siuzdak, H. (2023). *Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis*. [arXiv:2306.00814](https://arxiv.org/abs/2306.00814). — Vocoder dùng để giải mã mel ra waveform (§2.3, §4.1).
- Dieleman, S. (2024). *The paradox of diffusion distillation*. [sander.ai blog](https://sander.ai/2024/02/28/paradox.html). — Giải thích cơ chế "distributional averaging" dùng để chẩn đoán regression-to-mean ở §4.9/§4.11/§4.16.
- Distribution/Adversarial Distribution Matching (DMD/ADM), 2024-2025 — nhóm kỹ thuật (không phải một paper đơn lẻ được chốt cụ thể trong quá trình research của report này) khớp phân phối thay vì MSE thuần cho diffusion distillation; tham khảo cùng lúc với Dieleman (2024) ở §4.11 làm cơ sở lý thuyết cho việc đổi `loss_velocity` sang L1.
- Chen, Z., Badrinarayanan, V., Lee, C.-Y., Rabinovich, A. (2018). *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*. ICLR 2018 / [arXiv:1711.02257](https://arxiv.org/abs/1711.02257). — Cơ sở cho giả thuyết mất cân bằng gradient giữa `loss_gt`/`loss_velocity` (§5.2), test bằng fake teacher chưa xác nhận rõ với teacher thật.
- MeanFlow (2025). [arXiv:2505.13447](https://arxiv.org/abs/2505.13447). — Reformulation vận tốc trung bình `u(x_t,r,t)` cho sinh một-bước; đã thử prototype toy, chưa thành công (§5.2). *(Tên tác giả chưa được xác minh lại trong quá trình viết report này — chỉ dùng arXiv ID đã kiểm chứng qua tra cứu ban đầu.)*
- MusFlow (2025) — flow matching cho sinh nhạc, cùng domain với project; chỉ được ghi nhận qua tên gọi lúc research literature (§5.2), chưa có trích dẫn chính xác paper/arXiv ID cụ thể — cần tra cứu lại nếu muốn dùng làm cơ sở kỹ thuật thật.
- ASLP-lab. *DiffRhythm2*. [github.com/ASLP-lab/DiffRhythm2](https://github.com/ASLP-lab/DiffRhythm2). — Model teacher mà project distill từ (§2.1).
- OpenMuQ. *MuQ-MuLan* (`OpenMuQ/MuQ-MuLan-large`, HuggingFace). — Model style embedding audio-text dùng chung bởi teacher và student (§2.1, §3.1-3.2).
- VinAI Research. *XPhoneBERT* (`vinai/xphonebert-base`, HuggingFace) kết hợp `text2phonemesequence` — text encoder phoneme-level tiếng Việt, thay thế `xlm-roberta-base` từ 2026-07-17 (§3.2, §4.17). *(Tên paper/venue chính xác chưa được xác minh lại trong report này — chỉ dùng tên model HuggingFace đã kiểm chứng chạy được, xem §4.17.)*
