# GenMusic VN

Project bài tập lớn môn AI tạo sinh.

Mục tiêu: nhập văn bản tiếng Việt, chạy toàn bộ pipeline AI trên Kaggle GPU, sau đó trả về file nhạc nền `.mp3`.

## Chạy Nhanh Trên macOS

Các bước này đã được kiểm tra trên máy macOS hiện tại. Project dùng `python3`, không dùng lệnh `python`.

1. Đi tới thư mục project:

```bash
cd /Users/user/IdeaProjects/GenMusic
```

2. Kiểm tra Python và Kaggle CLI:

```bash
python3 --version
which kaggle
```

Nếu chưa có Kaggle CLI:

```bash
python3 -m pip install --user -U kaggle
```

3. Tạo hoặc kiểm tra file `.env` ở thư mục gốc project:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_token
```

Với Kaggle CLI 2.x, dùng `KAGGLE_API_TOKEN` hoặc file `~/.kaggle/access_token`. Cặp `KAGGLE_USERNAME`/`KAGGLE_KEY` kiểu cũ có thể làm project nhận nhầm là đã sẵn sàng nhưng Kaggle CLI vẫn fail khi upload dataset.

4. Chạy test nhanh:

```bash
python3 -m unittest discover -s tests -v
```

5. Chạy web local:

```bash
python3 -m genmusic_vn.server --port 8000
```

Khi thấy dòng này là server đã chạy:

```text
GenMusic VN running at http://127.0.0.1:8000
```

6. Mở trình duyệt:

```text
http://127.0.0.1:8000
```

7. Nhập một đoạn tiếng Việt, ví dụ:

```text
Một chiều mưa, tôi nhớ về những con phố cũ và ánh đèn vàng bên hiên nhà.
```

Sau đó bấm `Generate MP3`. Web sẽ tạo Kaggle job, gửi text lên Kaggle GPU, theo dõi trạng thái và hiện audio player/link tải khi MP3 tải xong.

Nếu chỉ muốn kiểm tra local có tạo job đúng mà chưa submit lên Kaggle:

```bash
python3 -m genmusic_vn.cli generate --text "Một chiều mưa, tôi nhớ về những con phố cũ." --duration 30 --no-submit
```

Nếu port `8000` đang bận, đổi sang port khác:

```bash
python3 -m genmusic_vn.server --port 8001
```

## Kiến Trúc

```text
Local web/CLI
  Nhận input: văn bản tiếng Việt, từ 1-2 câu đến vài chục câu
  Trả output: link nghe/tải file MP3

Kaggle GPU
  Lập kế hoạch cho text dài
  Phân tích cảm xúc tiếng Việt
  Tra cứu dataset kiến thức âm nhạc Việt Nam
  Viết lại nội dung thành lời bài hát hoàn chỉnh
  Chọn key, scale, BPM, hợp âm, nhạc cụ và hướng giai điệu
  Sinh prompt cho MusicGen
  Chạy MusicGen
  Chuyển WAV sang MP3
```

Local không chạy mô hình AI nặng và không sinh nhạc. Local chỉ tạo job Kaggle từ raw text, upload dataset job, push Kaggle Kernel, theo dõi trạng thái, tải file `.mp3` về và phục vụ file cho giao diện web.

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

Dataset này không phải audio training set. Nó được dùng để hướng dẫn pipeline trước khi gọi MusicGen:

- ánh xạ cảm xúc sang BPM, key, scale, hợp âm
- chọn màu nhạc cụ Việt Nam như đàn tranh, đàn bầu, sáo trúc, trống cơm
- chọn template thể loại như V-pop ballad, cinematic pop, lo-fi memory
- chọn hình ảnh lời hát, chorus, bridge
- bổ sung prompt keyword cho MusicGen theo ngữ cảnh tiếng Việt

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

Text gốc vẫn được giữ nguyên trong `request.json`; chỉ prompt sinh nhạc được cô đọng để MusicGen không bị loãng.

## Vì Sao Chỉ Dùng MusicGen

Vì đây là bài tập lớn phi thương mại, MusicGen phù hợp để demo:

- mô hình text-to-music dễ giải thích
- chạy được trên Kaggle GPU
- local app nhẹ
- tránh phức tạp khi duy trì nhiều backend sinh nhạc

## Chạy Trên macOS

Máy macOS thường có lệnh `python3`, không có sẵn lệnh `python`. Các lệnh dưới đây dùng `python3` để chạy đúng trên máy này.

Kiểm tra Python:

```bash
python3 --version
```

Nếu cần cài Kaggle CLI:

```bash
python3 -m pip install --user -U kaggle
```

Nếu terminal báo không thấy `kaggle` sau khi cài bằng `pip`, mở shell mới hoặc thêm Python user bin vào `PATH`. Trên macOS thường là một trong các thư mục dạng:

```bash
export PATH="$HOME/Library/Python/3.12/bin:$PATH"
```

Trên máy này Kaggle CLI đang có ở:

```text
/opt/homebrew/bin/kaggle
```

## Cài Đặt Kaggle API Token

Tạo token trên Kaggle:

1. Vào Kaggle -> Account Settings.
2. Chọn `Create New Token`.
3. Kaggle tải về file `kaggle.json` có `username` và `key`.

Cài Kaggle CLI:

```bash
python3 -m pip install --user -U kaggle
```

Cách khuyến nghị cho project này: tạo file `.env` hoặc `.env.local` ở thư mục gốc project:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_token
```

Project sẽ tự đọc token từ `.env`, `.env.local`, environment variables hoặc `~/.kaggle/access_token` rồi truyền vào Kaggle CLI. Không commit `.env` hoặc `.env.local` lên GitHub.

Có thể dùng cách chuẩn của Kaggle nếu muốn:

```bash
mkdir -p ~/.kaggle
# đặt token mới vào ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

## Chạy Web Demo

```bash
python3 -m genmusic_vn.server --port 8000
```

Mở:

```text
http://127.0.0.1:8000
```

Nhập text tiếng Việt và bấm `Generate MP3`. Khi Kaggle job hoàn tất, giao diện sẽ hiện audio player và link tải MP3.

## Chạy CLI

Chỉ stage job, chưa submit lên Kaggle:

```bash
python3 -m genmusic_vn.cli generate --text "Một chiều mưa, tôi nhớ về những con phố cũ." --duration 30 --no-submit
```

Submit lên Kaggle và đợi tải MP3:

```bash
python3 -m genmusic_vn.cli generate --text "Một chiều mưa, tôi nhớ về những con phố cũ." --duration 30 --wait
```

MP3 sau khi tải về nằm ở:

```text
outputs/<run_id>/kaggle_job/downloaded_output/
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
    run_commands.sh
    run_commands.ps1
```

Kaggle Kernel sẽ giải nén `genmusic_vn_source.zip`, chạy pipeline AI tiếng Việt, tạo prompt MusicGen, sinh audio, chuyển sang MP3 và ghi:

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
  emotion.py          # phân tích cảm xúc tiếng Việt, chạy trên Kaggle
  text_planner.py     # xử lý input dài, chạy trên Kaggle
  music_theory.py     # chọn key, scale, chord, melody, chạy trên Kaggle
  lyric_writer.py     # viết lại thành bài hát hoàn chỉnh, chạy trên Kaggle
  prompt_builder.py   # tạo prompt MusicGen, chạy trên Kaggle
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

## Kiểm Thử

```bash
python3 -m unittest discover -s tests -v
```

## Tài Liệu Tham Khảo

- AudioCraft / MusicGen: https://github.com/facebookresearch/audiocraft
- MusicGen docs: https://raw.githubusercontent.com/facebookresearch/audiocraft/main/docs/MUSICGEN.md
- Kaggle API docs: https://www.kaggle.com/docs/api
