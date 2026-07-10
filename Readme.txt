GenMusic VN - Hướng dẫn nhanh

1. Cài Kaggle CLI:
   pip install -U kaggle

2. Tạo file .env hoặc .env.local tại thư mục gốc project:
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key

3. Chạy web local:
   python -m genmusic_vn.server --port 8000

4. Mở trình duyệt tại:
   http://127.0.0.1:8000

5. Nhập văn bản tiếng Việt và bấm Tạo MP3.
   Local dùng text model đã train để phân tích cảm xúc/phong cách nếu có artifact.
   Kaggle chạy custom composer, F5-TTS tiếng Việt và trả về MP3.
   MMS tiếng Việt được giữ làm TTS dự phòng nếu F5-TTS lỗi.

6. Train text model:
   python -m genmusic_vn.cli train-text-model --samples 1200 --seed 5602 --wait

7. Chạy local bằng custom composer:
   python -m genmusic_vn.cli generate-local --text "Một đoạn văn tiếng Việt..." --duration 30

8. Chạy test:
   python -m unittest discover -s tests -v

9. Tạo dataset tham chiếu an toàn:
   python -m genmusic_vn.cli make-reference-dataset --count 24 --seed 5602

10. Tự train, giả lập user, đánh giá và thêm case yếu:
    python -m genmusic_vn.cli self-improve --iterations 3 --samples 640 --eval-count 24 --seed 5602

11. Tạo dataset tổng hợp vài chục nghìn mẫu:
    python -m genmusic_vn.cli make-train-dataset --count 30000 --profile diverse --seed 5602 --out datasets/training/diverse_30000_train.jsonl
    python -m genmusic_vn.cli self-improve --iterations 2 --samples 8000 --eval-count 32 --extra-dataset datasets/training/diverse_30000_train.jsonl --out outputs/self_improve_30000

12. Tạo dataset quy mô GB:
    python -m genmusic_vn.cli make-large-dataset --target-gb 5 --out datasets/training/diverse_5gb --shard-mb 128 --batch-size 4000 --seed 5602
    python -m genmusic_vn.cli self-improve --iterations 1 --samples 4000 --eval-count 24 --extra-dataset datasets/training/diverse_5gb --extra-dataset-limit 60000 --out outputs/self_improve_5gb

Báo cáo self-improve có 4 plot: thời lượng input và thời gian xử lý, cảm xúc và BPM, rating người dùng, tỷ lệ success/error. Dữ liệu plot nằm trong thư mục plots/.

Nếu có lyric thật, chỉ đưa vào bằng file JSONL local mà bạn có quyền sử dụng, rồi truyền qua --dataset hoặc --extra-dataset. Project không tự crawl nguyên lời bài hát có bản quyền.

Xem README.md để biết đầy đủ kiến trúc và cách hoạt động.

Crawler section lời có license:
   python -m genmusic_vn.cli crawl-licensed-lyrics --sources datasets/sources/licensed_lyrics_sources.example.json --out datasets/training/licensed_lyric_sections.jsonl --max-sections 12
   python -m genmusic_vn.cli train-rhyme-profile --dataset datasets/training/licensed_lyric_sections.jsonl --out models/rhyme_profile.json

Mỗi record là một verse/chorus đầy đủ, có nhãn section, URL và license; giới hạn 24 dòng/2.400 ký tự, không ghép thành cả bài.

Báo cáo telemetry của toàn project:
   python -m genmusic_vn.cli project-report --source outputs --out outputs/project_report
   http://127.0.0.1:8000/api/project/report

Dataset tổng hợp mặc định 5 GB:
   python -m genmusic_vn.cli make-large-dataset --target-gb 5 --out datasets/training/diverse_5gb --shard-mb 128 --batch-size 4000 --seed 5602
