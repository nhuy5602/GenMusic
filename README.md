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

Sinh dataset tong hop lon voi nhieu to hop boi canh/style, khong lap lai prompt:

```powershell
python -m genmusic_vn.cli make-train-dataset --count 30000 --profile diverse --seed 5602 --out datasets/training/diverse_30000_train.jsonl
```

`--profile diverse` tao input goc moi cho tung record va phan bo theo 11 genre, 8 mood, 3 nhom do dai. Dataset nay la du lieu tu tao, khong phai loi bai hat crawl tu web.

Neu can dataset co size hang GB, generator ghi streaming thanh shard, khong giu toan bo trong RAM:

```powershell
python -m genmusic_vn.cli make-large-dataset --target-gb 5 --out datasets/training/diverse_5gb --shard-mb 128 --batch-size 4000 --seed 5602
```

Khi train voi thu muc shard, loader mac dinh reservoir-sample toi da 60000 record (`--dataset-limit` hoac `--extra-dataset-limit`). Toan bo 5 GB van duoc giu lai de doi seed/sample cho cac lan sau.

Project không nhúng reference lyric/MP3 hoặc danh sách bài hát mẫu trong code. Reference dataset phải được cung cấp từ bên ngoài, có nhãn rõ ràng, rồi truyền bằng `--dataset` khi train hoặc `--reference-dataset` khi chạy self-improve. Mỗi dòng nên có:

```json
{"id":"my_case_001","input_text":"...","emotion":"sadness","genre_label":"pop_ballad","style_prompt":"Vietnamese melancholic pop ballad, soft piano, warm strings","expected_keywords":["mưa","piano"],"expected_vocal_gender":"female","source":"user_licensed_lyrics"}
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
  cli.py              # điểm vào CLI local
  server.py           # điểm vào web/API local
  core/               # logic cốt lõi của pipeline sinh nhạc
    controls.py       # đọc các điều khiển mood, BPM, duration
    emotion.py        # phân tích cảm xúc và thể loại
    lyric_writer.py   # lập cấu trúc verse/chorus và viết lời
    music_theory.py   # key, scale, chord và melody
    pipeline.py       # điều phối toàn bộ pipeline
    generators/       # các backend sinh backing track local/Kaggle
  data/               # tạo, nhập và quản lý dataset
    training_dataset.py # sinh record huấn luyện tiếng Việt
    dataset_scale.py  # tạo dataset streaming/shard tới quy mô GB
    reference_dataset.py # loader reference dataset do người dùng/crawler cung cấp
    licensed_lyric_crawler.py # nhập section có license được phê duyệt
  evaluation/         # chấm điểm, self-improve và báo cáo
    evaluation.py     # đánh giá text -> emotion -> lyric -> prompt
    quality_checks.py  # đủ lời, vần, beat/mood, vocal, clarity, flow/style
    self_improve.py   # train -> giả lập user -> đánh giá -> thêm case yếu
    report_plots.py   # tạo biểu đồ matplotlib cho model và project
  integrations/       # kết nối dịch vụ ngoài và artifact model
    kaggle_auto.py    # tạo dataset/kernel và tải kết quả Kaggle
    training_auto.py  # submit job train text model trên Kaggle
    trained_text_model.py # train/load/predict model text local
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

## Tự Cải Thiện Sau Mỗi Lần Train

Chạy vòng local để tự tạo thêm data, train model, giả lập user nhập prompt, chấm output và sinh thêm record nhắm vào các case yếu:

```powershell
python -m genmusic_vn.cli self-improve --iterations 3 --samples 640 --eval-count 24 --seed 5602 --out outputs/self_improve --duration 30
```

Chay voi dataset vai chuc nghin mau:

```powershell
 python -m genmusic_vn.cli self-improve --iterations 2 --samples 8000 --eval-count 32 --seed 5602 --extra-dataset datasets/training/diverse_30000_train.jsonl --out outputs/self_improve_30000 --duration 30 --stop-score 0.96
```

Moi iteration luu model rieng, report danh gia va cac record nham vao loi yeu. Sau cung model duoc chon la model cua iteration co `combined_score` cao nhat.

Report tu `evaluate` va `self-improve` co dashboard matplotlib (PNG + `plot_data.json`):

- `duration_input_vs_processing_time.png`: duration input vs thoi gian xu ly.
- `emotion_vs_bpm.png`: phan bo emotion vs BPM sinh ra.
- `user_rating.png`: rating proxy 1-5 tu quality score; neu co rating that se ghi ro nguon.
- `success_error_rate.png`: ty le thanh cong/loi cua tung evaluation run.

Vi du voi dataset 5 GB:

```powershell
python -m genmusic_vn.cli self-improve --iterations 1 --samples 4000 --eval-count 24 --seed 5602 --extra-dataset datasets/training/diverse_5gb --extra-dataset-limit 60000 --out outputs/self_improve_5gb --duration 30 --stop-score 0.90
```

Artifact nam trong `outputs/self_improve_5gb/plots/` va duoc lien ket trong `self_improve_report.json`/`.md`.

Nếu có dataset lyrics/text local bạn có quyền dùng:

```powershell
python -m genmusic_vn.cli self-improve --iterations 3 --samples 640 --eval-count 24 --extra-dataset datasets/training/my_licensed_lyrics.jsonl
```

Nếu dataset có đủ nhãn reference để vừa train vừa đánh giá:

```powershell
python -m genmusic_vn.cli self-improve --iterations 3 --samples 640 --eval-count 24 --reference-dataset datasets/training/licensed_lyric_sections.jsonl
```

Report chính:

```text
outputs/self_improve/self_improve_report.json
outputs/self_improve/self_improve_report.md
outputs/self_improve/iteration_*/evaluation_report.json
outputs/self_improve/iteration_*/quality/quality_report.json
```

Quality report chấm các điểm gần với checklist nghe thử: đủ lời, lời có vần, beat/BPM hợp mood, có kế hoạch vocal hoặc vocal artifact, WAV có bị quá nhỏ/clipping không, flow có hợp style không. Local composer chỉ xác minh backing và planning; vocal hát thật vẫn cần Kaggle F5-TTS/MMS TTS.

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

## Section Lời Có License

Crawler tùy chọn chỉ thu thập nguyên một section có nhãn, chẳng hạn một verse hoặc một chorus, từ nguồn được phê duyệt rõ ràng trong manifest và có đánh dấu public domain, Creative Commons hoặc quyền sở hữu của người dùng. Crawler tôn trọng `robots.txt`, lưu URL và license trong từng record, giới hạn mỗi section ở 24 dòng/2.400 ký tự, và không tái tạo thành một bài hoàn chỉnh.

Bắt đầu từ manifest mẫu và thay placeholder bằng nguồn mà bạn có quyền sử dụng:

```powershell
python -m genmusic_vn.cli crawl-licensed-lyrics --sources datasets/sources/licensed_lyrics_sources.example.json --out datasets/training/licensed_lyric_sections.jsonl --max-sections 0
python -m genmusic_vn.cli train-rhyme-profile --dataset datasets/training/licensed_lyric_sections.jsonl --out models/rhyme_profile.json
```

Với `--max-sections 0`, crawler lấy toàn bộ verse/chorus/bridge hợp lệ mà nguồn trả về trong response giới hạn 2 MB; có thể truyền số dương để giới hạn mỗi nguồn. Output crawler có thể truyền vào `--dataset` hoặc `--reference-dataset` khi nhãn emotion/style hợp lệ. Các `rhyme_features` được dùng để học vần cuối và điệp âm ở cấp verse/chorus. Profile đã học chỉ là gợi ý, không ép vần khi câu trở nên thiếu tự nhiên.

## Telemetry Của Project

Các plot đánh giá mô tả vòng lặp model/planning. Để xem hiệu năng thật của app và Kaggle, quét các file `job_state.json` sau khi job kết thúc:

```powershell
python -m genmusic_vn.cli project-report --source outputs --out outputs/project_report
```

Lệnh này tạo `project_report.json`/`.md` và các plot về độ trễ từ input tới MP3, tỷ lệ retry/error/success của Kaggle, thời gian từng stage Kaggle và số lượng kết quả theo trạng thái. Web app cũng cung cấp báo cáo tại `/api/project/report`.

Bộ lập kế hoạch lời dùng chorus anchor ngắn lặp lại, chorus đầy đủ cho request 24-30 giây, điệp âm có chọn lọc và nhịp câu verse đa dạng. Các cặp dòng 1-2, 3-4, 5-6 không bị bắt buộc phải có cùng một vần cuối tuyệt đối.

Project không phụ thuộc vào bất kỳ bài hát, MP3 hay file lyric tham chiếu cụ thể nào. Bộ lyric/MP3 có license trong tương lai có thể được đưa vào qua manifest dataset hoặc quy trình import ZIP riêng.
