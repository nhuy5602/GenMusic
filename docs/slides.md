---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
---

# Xây dựng mô hình sinh ca khúc tiếng Việt
## từ lời bài hát và gợi ý phong cách

**Bài tập lớn — Nền tảng AI tạo sinh**

Đàm Việt Anh · Vũ Thùy Linh · Nguyễn Lê Hùng · Vũ Hồng Quang · Đặng Ngọc Huy
GVHD: TS. Dương Quang Huy — Đại học Bách khoa Hà Nội, 07/2026

---

## Đặt vấn đề

- Sinh ca khúc tự động (Song Generation) cần kiểm soát đồng thời: nội dung lời, phát âm, trường độ, cao độ, nhịp điệu — hài hòa với nhạc đệm.
- Tiếng Việt: đơn âm tiết, giàu thanh điệu — cao độ vừa quyết định **ngữ nghĩa** vừa bị chi phối bởi **giai điệu**.
- Dữ liệu thực tế: bản thu đã trộn giọng+nhạc, lời không có mốc thời gian → cần pipeline tiền xử lý riêng.
- Model sinh nhạc quy mô lớn (DiffRhythm2, ~1,1 tỷ tham số) tốn tài nguyên → cần **chưng cất tri thức (Knowledge Distillation)** sang model nhỏ gọn.

---

## Mục tiêu & đóng góp

**Mục tiêu**: sinh phổ âm thanh giọng hát chất lượng cao từ (ca từ, nhạc đệm, phong cách), tối ưu hiệu năng qua chưng cất tri thức từ teacher DiffRhythm2.

**Đóng góp chính**:
1. Pipeline tự động tiền xử lý dữ liệu âm thanh thô tiếng Việt.
2. Huấn luyện mạng dự đoán phổ giọng hát bằng chưng cất tri thức (CFM).
3. Đánh giá thực nghiệm qua chỉ số kỹ thuật + chất lượng cảm tính.

---

## Cơ sở lý thuyết — các khối chính

| Thành phần | Vai trò |
|---|---|
| **Mel-spectrogram** + Vocos | Biểu diễn trung gian nén, giải mã lại waveform (iSTFT, không cần GAN nặng) |
| **XPhoneBERT** + G2P | Âm vị hoá lời — giải quyết bất đối xứng chữ viết/âm vị + thanh điệu tiếng Việt |
| **MuQ-MuLan** | Style embedding 512-chiều — "neo phong cách" toàn cục từ audio tham chiếu |
| **Conditional Flow Matching (CFM)** | Học trường vận tốc nối thẳng $Z_0\to Z_1$, ít bước sampling hơn DDPM |
| **DiffRhythm2** | Teacher ~1,1 tỷ tham số, sinh bài hát đầy đủ qua Block Flow Matching |

---

## Vì sao Flow Matching thay vì DDPM?

DDPM: khử nhiễu qua $K$ bước rời rạc, tốn chi phí khi $K$ lớn.

Flow Matching: học trực tiếp trường vận tốc của một ODE liên tục

$$\frac{dZ_t}{dt} = u_t(Z_t, c), \quad Z_0\sim\mathcal{N}(0,I),\; Z_1\sim p_{\text{data}}$$

Với nội suy thẳng $Z_t=(1-t)Z_0+tZ_1$, mục tiêu vận tốc đơn giản là $Z_1-Z_0$:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}\left[\|v_\theta(Z_t,t,c) - (Z_1-Z_0)\|_2^2\right]$$

→ quỹ đạo gần thẳng ⇒ **ít bước Euler sampling** vẫn giữ chất lượng.

---

## Kiến trúc tổng thể

**Tiền xử lý**: WAV/MP3 thô → Demucs (tách vocal/backing) → Whisper (transcript+timestamp) + Mel 100-bin Vocos-native; song song, MuQ-MuLan trích style 512d từ bản mix gốc → tất cả gộp vào `records.jsonl`.

**Huấn luyện** (cùng dataset, hai nhánh không đổi biểu diễn đầu vào):
- `train-self` — CFM thuần, không teacher.
- `train-distill` — chưng cất thêm từ DiffRhythm2 teacher.

**Suy luận**: Gaussian noise → Euler ODE sampling qua MicroDiT → giải mã Vocos (hoặc BigVGAN thật ở latent-space) → WAV/MP3.

---

## MicroDiT — backbone student duy nhất

- Nhận **noisy mel** $x_t$ + điều kiện: lời (qua **cross-attention**) + style (cộng trực tiếp).
- **Không** nhận backing làm input — backing chỉ dùng để dựng target $x_1$ = hỗn hợp cả bài:

$$x_1 = \log\big(\exp(x_1^{\text{voc}}) + \exp(x_1^{\text{acc}})\big)$$

- Lời → G2P → XPhoneBERT (đóng băng) → `TextSelfAttentionLayer` (trainable) → key/value cho cross-attention.
- Hai đầu ra: velocity cả bài (chính) + velocity vocal (phụ trợ, chống "lãng quên" giọng hát yếu hơn nhạc đệm).

---

## Một backbone đã bị gộp/loại bỏ

Từng có 2 backbone song song: **MicroDiT** (cross-attention) vs **NativeDiTStudent** (self-attention nối chuỗi, nhúng âm vị từ đầu, port từ mã nguồn DiffRhythm2 gốc).

So sánh cùng cấu hình/epoch:

| | loss_gt | Tốc độ/step |
|---|---|---|
| MicroDiT | tương đương | **1×** |
| NativeDiTStudent | không tốt hơn | chậm hơn **4,4×** |

→ Gộp: giữ lại ý tưởng self-attention riêng cho lời (`TextSelfAttentionLayer`) gắn vào MicroDiT, xoá NativeDiTStudent khỏi mã nguồn.

---

## Chưng cất tri thức từ DiffRhythm2

$$\mathcal{L} = (1-\alpha)\,\text{L1}(\hat u_t, \tilde v_T) + \alpha\,\mathcal{L}_{\text{gt}}$$

- Teacher đóng băng, khôi phục đúng hợp đồng: tokenizer thật, layout `[Text, Clean, Noisy]`, attention mask theo khối.
- Cầu nối lệch tần số: **93,75 Hz (student) ↔ 5 Hz (teacher)**, resample + adapter học được $64\to100$ mel bins.
- $\alpha=0{,}8$ xác nhận là điểm tối ưu thật qua đo đa-bài (N=6) — không phải $\alpha=0{,}5$ mặc định ban đầu.
- L1 (không MSE) cho loss teacher-matching — tránh hiệu ứng "làm mượt" của MSE trong distillation sinh.

---

## Không gian latent thật của teacher

DiffRhythm2 **không** chạy trên mel — teacher thật dùng **Music VAE latent 64-chiều, 5 Hz** (~19× thấp hơn mel 93,75 Hz của student).

- DiffRhythm2 chỉ công bố **decoder** (BigVGAN), không công bố encoder.
- Giải pháp: đóng băng decoder thật, tự train **`LatentAudioEncoder`** (11M tham số) bằng loss reconstruction thuần (không GAN).

**Lỗi thật gặp phải**: LR cố định, không warmup/clipping → encoder **collapse** (latent ground-truth giải mã ra gần như một nốt đứng yên, `pitch_std_semitones`≈0,9).

**Fix**: warmup 200 step + cosine decay + gradient clipping → `pitch_std_semitones` tăng lên 6,16–12,38.

---

## Thí nghiệm 4: Mel-space vs Latent-space

| Điều kiện | flatness | voiced_ratio | pitch_std (semitone) |
|---|---|---|---|
| Nhiễu trắng (mốc) | 0,562 | 0,040 | 0,49 |
| Vocal thật (mốc) | 0,056 | 0,743 | **6,39** |
| (A) Mel-space (exp06, đã hội tụ) | 0,011 | 0,924 | 0,91 |
| (B) Latent-space, encoder chưa fix | ~0,0003 | 0,95–0,99 | 0,83–0,97 |
| (C) Latent-space, encoder đã fix (13/300 epoch) | 0,00095 | 0,21 | **5,70** |

→ (C) đạt ~89% biên độ pitch của vocal thật, **dù train ít hơn hẳn** (A) → latent-space là bài toán hồi quy dễ hơn cho CFM.
→ Xác nhận bằng nghe thật: (C) là mẫu **đầu tiên nghe ra nhạc thật**, không còn nhiễu hoàn toàn.

---

## Cập nhật mới nhất: loại bỏ vòng Vocos khỏi encoder

Lần train encoder trước (§4.24) vẫn phải: mel → **giải mã ngược qua Vocos** → waveform → encoder — một xấp xỉ có mất pha, không phải bản ghi gốc.

**Đã làm trong phiên này**:
- `preprocess-raw --raw-audio`: lưu thẳng waveform 24kHz, bỏ hẳn mel.
- `train-latent-encoder`/`precompute-latent-dataset`: đọc trực tiếp `raw_audio_mode`, cộng vocal+backing waveform — **không còn Vocos** trong pipeline train encoder.
- Mở rộng nhận **nhiều dataset cùng lúc** → gộp cả 6 phần corpus gốc: **249 → 1843 bài**.

**Đã kiểm chứng trên Kaggle GPU thật**: truy cập đủ 6 phần (1843 record, không lỗi), train thử 1 epoch với 1 bài/phần — chạy đúng.

---

## Kết quả: encoder huấn luyện lại trên toàn bộ 1843 bài

- **1839 bài** (sau lọc), 10 epoch, batch 4, LR 1e-4, 4600 step, ~98 phút thuần train.
- Loss giảm mượt: 2,105 → 1,734 → 1,612 → ... → **1,239** (epoch 10) — **tốt hơn** kết quả cũ trên 249 bài (1,28), dù dữ liệu lớn hơn 7,4×.
- `peak_vram_gb=15,30` trên T4 (16GB) — batch=4 đã gần ngưỡng an toàn.
- Phát hiện phụ: file waveform thô nặng hơn mel đúng **2,56 lần** (256/100) → I/O chậm hơn hẳn nếu không prefetch; thêm `num_workers=4` đưa tốc độ về gần bình thường.

**Nhưng sanity-check phát hiện encoder collapse**: `pitch_std_semitones` giải mã chỉ **0,44** trung bình (so với **9,46** của audio thật) — cùng dấu hiệu như lỗi gốc §4.24, dù loss trông tốt. Xác nhận lại: **loss thấp không đảm bảo encoder khoẻ**.

**Thử lại với warmup dài hơn (400 step) + LR thấp hơn (5e-5) — vẫn collapse** (0,78, vẫn xa mức khoẻ 6-12). Hai lần thất bại liên tiếp → **quyết định**: dừng thử hyperparameter ở quy mô lớn, quay lại checkpoint encoder 249-bài đã xác nhận khoẻ, dùng nó để huấn luyện MicroDiT bằng CFM trên latent-space — bước chưa từng thực hiện trước đây.

---

## Kết quả kiểm chứng hạ tầng (RQ1–RQ2)

| Thiết lập | Kết quả |
|---|---|
| Vocoder loopback (Vocos-native) | correlation **0,997**, RMSE 1,78 — tốt nhất trong mọi cấu hình thử |
| Kaggle end-to-end, 12 bài | 12/12 tiền xử lý; 120 step; loss cuối 5,34; không NaN/Inf |
| Cục bộ, 2 bài | RMS/silence hợp lý; loopback Vocos 0,986 |

→ Trả lời RQ1/RQ2: hợp đồng mel-Vocos đúng, pipeline đầu-cuối chạy được thật trên dữ liệu tiếng Việt — nền tảng để mở rộng, chưa phải đánh giá chất lượng âm nhạc quy mô lớn.

---

## Chưng cất: kiểm chứng cơ chế (RQ3)

- Teacher thật: **1.136.249.664** tham số; student nhỏ: **745.188** tham số trainable (~1525×).
- `distillation_active=true` xác nhận qua log — không phải baseline đổi tên.
- Kết quả 2 bài/30 epoch: chưa đủ bằng chứng thống kê cải thiện chất lượng (mẫu quá nhỏ, dao động loss 3,5–229 do 1 step/epoch).
- **Chưa bác bỏ, chưa xác nhận** — cần chạy lại ma trận baseline/distilled cùng seed, nhiều update hơn.

---

## Hạn chế & rủi ro hợp lệ

1. **Demucs** có thể rò giọng vào backing → model dựa vào tín hiệu lời "rò rỉ".
2. **Whisper timestamp** không chính xác tới âm vị → giới hạn điều khiển lời theo frame.
3. Tập dữ liệu chưa đủ đa dạng ca sĩ/thể loại/cấu trúc bài hát.
4. Sanity metrics (flatness, voiced_ratio) **không thay thế** đánh giá người nghe — đã tự phát hiện: 2/3 chỉ số này không phân biệt được CFM output với nhiễu ngẫu nhiên thuần (§4.26), chỉ `pitch_std_semitones` còn tín hiệu thật.

---

## Kết luận

- Đã xây dựng **pipeline nghiên cứu khả dụng, minh bạch giới hạn** — chưa phải hệ thống sinh bài hát hoàn chỉnh.
- Nhánh mel-space: hợp đồng Vocos đúng là cải tiến tác động rõ nhất tới độ trung thực audio.
- Nhánh latent-space (không gian thật của teacher): kết quả sơ bộ **vượt hẳn** mel-space dù train ít hơn — hướng ưu tiên hiện tại.
- Vừa loại bỏ được một nguồn xấp xỉ (Vocos detour) và mở rộng dữ liệu encoder 7,4× — đang huấn luyện lại trên quy mô đầy đủ.

---

## Hướng phát triển

1. **Đang thực hiện**: encoder trên 1843 bài → sanity-check → CFM training tới hội tụ thật (không bị cắt do hết quota) → điền lại bảng so sánh mel/latent.
2. Nâng cao số lượng & chất lượng dữ liệu nhạc Việt Nam.
3. Đánh giá chủ quan MOS/CMOS với người nghe thật + độ rõ lời tiếng Việt.
4. Mở rộng model: thanh điệu, alignment, pitch/F0/melody explicit conditioning.

---

# Cảm ơn!

Câu hỏi & thảo luận
