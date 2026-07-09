# GenMusic VN

Project bài tập lớn môn AI tạo sinh.

Mục tiêu: nhập văn bản tiếng Việt, dùng model đã train để phân tích cảm xúc/style, tạo lời/harmony/melody, sinh backing bằng custom composer của project, synth vocal tiếng Việt bằng TTS rồi mix thành file `.mp3`.

## Kiến Trúc

```text
Local web/CLI
  Nhận input: văn bản tiếng Việt, từ 1-2 câu đến vài chục câu
  Trả output: link nghe/tải file MP3
  Ưu tiên load trained text model từ models/current hoặc datasets/trained_models

Trained text model
  Tự train bằng dataset tiếng Việt do project sinh ra
  Dự đoán emotion + genre/style
  Xuất artifact genmusic_text_model.json
  Local/Kaggle đều dùng artifact này, không gọi LLM API để phân tích text

Kaggle GPU / CPU
  Train lại text model khi cần cải thiện
  Lập kế hoạch cho text dài
  Phân tích cảm xúc tiếng Việt
  Tra cứu dataset kiến thức âm nhạc Việt Nam
  Viết lại nội dung thành lời bài hát hoàn chỉnh
  Chọn key, scale, BPM, hợp âm, nhạc cụ và hướng giai điệu
  Custom Music Model sinh chord + bass + drum + melody guide + arrangement
  Render MIDI/symbolic arrangement thành backing.wav/backing.mp3
  Chạy F5-TTS Vietnamese để synth vocal tiếng Việt
  Nếu F5-TTS lỗi thì fallback sang MMS Vietnamese TTS
  Mix vocal với backing track
  Chuyển WAV sang MP3
```

Luồng audio chính:

```text
Text tiếng Việt
  -> TextPlan / Emotion / Harmony / Lyrics
  -> Custom Music Model
  -> chord + bass + drum + melody guide + arrangement
  -> backing.wav + song.mid
  -> TTS vocal tiếng Việt
  -> mix vocal + backing
  -> final.mp3
```

Project không còn phụ thuộc Meta/Facebook MusicGen cho bước sinh nhạc nền. Một số field legacy `musicgen_*` có thể còn xuất hiện trong state để đọc job cũ, nhưng pipeline mới dùng custom composer trong repo.

TTS mặc định dùng model `hynt/F5-TTS-Vietnamese-ViVoice` trên Kaggle. Model này dùng một đoạn giọng tham chiếu tiếng Việt ngắn để tạo vocal; trường male/female trong project vẫn là khuyến nghị phối giọng, cao độ và profile hậu kỳ chứ không bảo đảm đổi hoàn toàn timbre ca sĩ. `facebook/mms-tts-vie` được giữ làm fallback để demo vẫn trả file MP3 có vocal khi F5-TTS lỗi.

## Model Train Của Project

Project có model train thật cho bước hiểu văn bản tiếng Việt:

```text
datasets/trained_models/genmusic_text_model.json
models/current/genmusic_text_model.json
```

Thứ tự load:

1. `models/current/genmusic_text_model.json`: artifact mới nhất tải từ Kaggle.
2. `datasets/trained_models/genmusic_text_model.json`: bootstrap artifact đi kèm repo.
3. fallback rule-based nếu chưa có artifact.

Model hiện tại là Multinomial Naive Bayes tự triển khai, train trên dataset supervised tiếng Việt do project tự sinh. Output gồm:

- emotion: `joy`, `sadness`, `anger`, `fear`, `calm`, `romantic`, `hope`, `nostalgic`
- genre/style: `pop_ballad`, `trap`, `edm`, `folk`, `rock`, `rnb`, `bolero`, `ambient`, `orchestral`, `horror`, `lofi`
- style prompt tương ứng để chọn harmony, arrangement và custom composer

Kiểm tra artifact đang được load:

```powershell
python -m genmusic_vn.cli model-status
```

Tự tạo training dataset:

```powershell
python -m genmusic_vn.cli make-train-dataset --count 800 --seed 5602 --out datasets/training/generated_text_model_train.jsonl
```

Train local nhanh để kiểm thử thuật toán:

```powershell
python -m genmusic_vn.cli train-text-model --local --samples 800 --seed 5602 --model-out datasets/trained_models/genmusic_text_model.json
```

Train trên Kaggle và tải artifact về `models/current`:

```powershell
python -m genmusic_vn.cli train-text-model --samples 1200 --seed 5602 --wait --model-out models/current/genmusic_text_model.json
```

Chỉ stage job Kaggle, chưa submit:

```powershell
python -m genmusic_vn.cli train-text-model --samples 1200 --seed 5602 --no-submit
```

## Dataset Kiến Thức Âm Nhạc Việt Nam

Project có dataset tri thức dạng có cấu trúc:

```text
datasets/vn_music_stylebank/
  emotion_to_music.json
  vietnamese_instruments.json
  genre_templates.json
  chord_presets.json
  lyric_patterns.json
```

Dataset này không phải audio training set. Nó được dùng để hướng dẫn pipeline trước khi custom composer sinh nhạc:

- ánh xạ cảm xúc sang BPM, key, scale, hợp âm
- chọn màu nhạc cụ Việt Nam như đàn tranh, đàn bầu, sáo trúc, trống cơm
- chọn template thể loại như V-pop ballad, cinematic pop, lo-fi memory
- chọn hình ảnh lời hát, chorus, bridge
- bổ sung keyword/style theo ngữ cảnh tiếng Việt

Khi submit job, dataset này được đóng gói vào `genmusic_vn_source.zip` và upload lên Kaggle.

## Xử Lý Text Dài

Input có thể là một câu ngắn, vài câu hoặc đoạn văn dài vài chục câu.

Với text dài, Kaggle tạo `TextPlan`:

- đếm số câu và số từ
- rút keyword/motif
- chọn câu đại diện ở mở đầu, diễn biến và kết thúc
- tạo bản cô đọng để dùng cho lyric, melody và prompt
- viết lại nội dung thành cấu trúc bài hát:
  `Verse 1 -> Pre-Chorus -> Chorus -> Verse 2 -> Bridge -> Final Chorus -> Outro`

Text gốc vẫn được giữ nguyên trong `request.json`; phần đưa vào planner/composer được cô đọng để harmony, lyrics và melody không bị loãng.

### Input Là Lời Bài Hát Dài

Nếu người dùng paste sẵn nhiều dòng lyrics, project tự nhận diện `input_kind = lyrics`.

- với lyrics ngắn: giữ cấu trúc dòng gốc để TTS hát trực tiếp
- với lyrics dài hơn duration cho phép: tự chọn excerpt gồm verse mở đầu, chorus/refrain lặp lại và đoạn kết
- với lyrics chưa có vần rõ: sửa nhẹ từng section để dễ bám melody
- với lyrics đã có vần: ưu tiên giữ nguyên dòng gốc, không ép thêm đuôi vần
- bộ đánh giá vần hỗ trợ nhiều kiểu vần tiếng Việt: vần cuối câu, vần móc đầu-cuối và vần lưng kiểu lục bát

Ví dụ duration 60s không cố hát toàn bộ một bài dài vài chục dòng; hệ thống chọn khoảng 10-12 dòng hát được để tránh TTS bị quá tải hoặc rơi về file nhạc nền không lời.

Khi làm báo cáo hoặc demo công khai, nên dùng lyrics tự viết, lyrics sinh tổng hợp hoặc dữ liệu có quyền sử dụng; nếu dùng lời bài hát có bản quyền thì cần ghi rõ nguồn và phạm vi sử dụng học thuật.

## Cài Đặt Kaggle API Token

Tạo token trên Kaggle:

1. Vào Kaggle -> Account Settings.
2. Chọn `Create New Token`.
3. Kaggle tải về file `kaggle.json` có `username` và `key`.

Cài Kaggle CLI:

```powershell
pip install -U kaggle
```

Cách khuyến nghị cho project này: tạo file `.env` hoặc `.env.local` ở thư mục gốc project:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

Project sẽ tự đọc token từ `.env`, `.env.local` hoặc environment variables rồi truyền vào Kaggle CLI. Không commit `.env` hoặc `.env.local` lên GitHub.

Có thể dùng cách chuẩn của Kaggle nếu muốn:

```powershell
mkdir $HOME\.kaggle
# đặt kaggle.json vào $HOME\.kaggle\kaggle.json
```

## Chạy Web Demo

```powershell
python -m genmusic_vn.server --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

Nhập text tiếng Việt và bấm `Generate MP3`. Khi Kaggle job hoàn tất, giao diện sẽ hiện audio player và link tải MP3.

## Chạy CLI

Chỉ stage job, chưa submit lên Kaggle:

```powershell
python -m genmusic_vn.cli generate --text "Một chiều mưa, tôi nhớ về những con phố cũ." --duration 30 --no-submit
```

Submit lên Kaggle và đợi tải MP3:

```powershell
python -m genmusic_vn.cli generate --text "Một chiều mưa, tôi nhớ về những con phố cũ." --duration 30 --wait
```

MP3 sau khi tải về nằm ở:

```text
outputs/<run_id>/kaggle_job/downloaded_output/
```

Chạy local bằng custom composer:

```powershell
python -m genmusic_vn.cli generate-local --text "Một chiều mưa, tôi nhớ về những con phố cũ." --duration 30
```

Output local nằm ở:

```text
outputs/local/<run_id>/backing.wav
outputs/local/<run_id>/song.mid
outputs/local/<run_id>/final.mp3  # có nếu máy có ffmpeg hoặc imageio-ffmpeg
```

## Dữ Liệu Upload Lên Kaggle

Mỗi request local tạo:

```text
outputs/<run_id>/
  request.json
  kaggle_job/
    dataset/
      request.json
      genmusic_vn_source.zip
      dataset-metadata.json
    kernel/
      run_genmusic_vn.py
      kernel-metadata.json
    run_commands.ps1
```

Kaggle Kernel sẽ giải nén `genmusic_vn_source.zip`, chạy pipeline AI tiếng Việt, sinh backing bằng custom composer, chạy TTS/mix và ghi:

```text
/kaggle/working/genmusic_vn/<run_id>.mp3
/kaggle/working/genmusic_vn/kaggle_result.json
```

## Cấu Trúc Project

```text
genmusic_vn/
  server.py           # web/API local
  cli.py              # CLI local
  kaggle_auto.py      # tự động tạo Kaggle Dataset/Kernel bằng API token
  training_dataset.py # tự sinh supervised dataset tiếng Việt để train model
  trained_text_model.py # train/load/predict emotion + genre/style model
  training_auto.py    # tạo Kaggle job train text model
  emotion.py          # phân tích cảm xúc tiếng Việt bằng trained model + fallback
  text_planner.py     # xử lý input dài, chạy trên Kaggle
  music_theory.py     # chọn key, scale, chord, melody, chạy trên Kaggle
  lyric_writer.py     # viết lại thành bài hát hoàn chỉnh, chạy trên Kaggle
  prompt_builder.py   # tạo mô tả style/arrangement cho custom composer và báo cáo
  pipeline.py         # điều phối pipeline phía Kaggle
datasets/
  vn_music_stylebank/ # dataset kiến thức âm nhạc Việt Nam
web/
  index.html
  app.css
  app.js
tests/
  test_pipeline.py
```

## Chạy Đánh Giá

Đánh giá pipeline text -> emotion -> lyric -> prompt trên dataset mặc định:

```powershell
python -m genmusic_vn.cli evaluate --out outputs/evaluation
```

Đánh giá bằng file Excel benchmark tiếng Việt:

```powershell
python -m genmusic_vn.cli evaluate-xlsx --xlsx "C:\Users\ADMIN\Documents\GenMusic\vietnamese_musicgen_input_dataset.xlsx" --out outputs/evaluation_xlsx
```

Đánh giá ablation chorus theo 2 nhánh `không truyền style` và `truyền style đúng`:

```powershell
python -m genmusic_vn.cli chorus-ablation --dataset datasets/evaluation/chorus_ablation_safe.jsonl --out outputs/chorus_ablation --duration 45
```

Dataset ablation dùng JSONL, mỗi dòng có dạng:

```json
{"id":"CASE01","chorus":"...","style":"Vietnamese pop ballad, piano, strings","expected_style_terms":["pop ballad","piano","strings"],"duration_seconds":45}
```

Lưu ý: không tự crawl nguyên lời chorus bài hit trên mạng. Nếu dùng lời bài hát có bản quyền trong báo cáo/demo, hãy chỉ dùng phần bạn có quyền sử dụng hoặc tự chuẩn bị file local.

Report sẽ được ghi vào:

```text
outputs/evaluation/evaluation_report.json
outputs/evaluation_xlsx/evaluation_report.json
outputs/chorus_ablation/chorus_ablation_report.json
```

Các metric chính gồm `emotion_match`, `keyword_recall`, `prompt_keyword_recall`, `scene_cue_density`, `diacritic_line_rate`, `vietnamese_rhyme_rate`, `rhyme_pair_rate`, `head_tail_rhyme_rate`, `luc_bat_rhyme_rate`, `melody_line_rate` và `overall_score`.

Nếu muốn vừa submit các dòng trong Excel lên Kaggle để sinh MP3 thật, dùng:

```powershell
python -m genmusic_vn.cli batch-generate-xlsx --xlsx "C:\Users\ADMIN\Documents\GenMusic\vietnamese_musicgen_input_dataset.xlsx" --out outputs/xlsx_batch --limit 3 --wait
```

## Kiểm Thử

```powershell
python -m unittest discover -s tests -v
```

## Tài Liệu Tham Khảo

- Kaggle API docs: https://www.kaggle.com/docs/api
- Kaggle API docs: https://www.kaggle.com/docs/api
