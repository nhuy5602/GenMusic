GenMusic VN - Huong dan nhanh

1. Cai Kaggle CLI:
   pip install -U kaggle

2. Tao file .env hoac .env.local tai thu muc goc project:
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_api_key

3. Chay web local:
   python -m genmusic_vn.server --port 8000

4. Mo trinh duyet:
   http://127.0.0.1:8000

5. Nhap van ban tieng Viet va bam Generate MP3.
   Local dung trained text model de phan tich emotion/style neu da co artifact.
   Kaggle chay custom composer + F5-TTS Vietnamese va tra ve file MP3.
   MMS Vietnamese TTS duoc giu lam fallback neu F5-TTS loi.

6. Train text model:
   python -m genmusic_vn.cli train-text-model --samples 1200 --seed 5602 --wait

7. Chay local bang custom composer:
   python -m genmusic_vn.cli generate-local --text "Mot doan van tieng Viet..." --duration 30

8. Chay test:
   python -m unittest discover -s tests -v

9. Tao dataset tham chieu an toan:
   python -m genmusic_vn.cli make-reference-dataset --count 24 --seed 5602

10. Tu train, gia lap user, danh gia va them case yeu:
   python -m genmusic_vn.cli self-improve --iterations 3 --samples 640 --eval-count 24 --seed 5602

11. Neu can dataset tong hop vai chuc nghin mau:
   python -m genmusic_vn.cli make-train-dataset --count 30000 --profile diverse --seed 5602 --out datasets/training/diverse_30000_train.jsonl
   python -m genmusic_vn.cli self-improve --iterations 2 --samples 8000 --eval-count 32 --extra-dataset datasets/training/diverse_30000_train.jsonl --out outputs/self_improve_30000

12. Neu can dataset co size hang GB:
   python -m genmusic_vn.cli make-large-dataset --target-gb 5 --out datasets/training/diverse_5gb --shard-mb 128 --batch-size 4000 --seed 5602
   python -m genmusic_vn.cli self-improve --iterations 1 --samples 4000 --eval-count 24 --extra-dataset datasets/training/diverse_5gb --extra-dataset-limit 60000 --out outputs/self_improve_5gb

Report self-improve co 4 plot PNG: duration vs processing time, emotion vs BPM, user rating proxy va success/error rate. Du lieu plot nam trong thu muc plots/.

Neu co lyrics that, chi dua vao bang file JSONL local ma ban co quyen su dung, roi truyen qua --dataset hoac --extra-dataset. Project khong tu crawl nguyen loi bai hat co ban quyen.

Xem README.md de biet day du kien truc va cach hoat dong.

Crawler section loi co license (khong tu y dung nguon ban quyen):
   python -m genmusic_vn.cli crawl-licensed-lyrics --sources datasets/sources/licensed_lyrics_sources.example.json --out datasets/training/licensed_lyric_sections.jsonl --max-sections 12
   python -m genmusic_vn.cli train-rhyme-profile --dataset datasets/training/licensed_lyric_sections.jsonl --out models/rhyme_profile.json

Moi record la mot verse/chorus day du co nhan section, URL va license; gioi han 24 dong/2.400 ky tu, khong ghep thanh ca bai.

Bao cao telemetry cua ca project:
   python -m genmusic_vn.cli project-report --source outputs --out outputs/project_report
   http://127.0.0.1:8000/api/project/report

Dataset tong hop mac dinh 5 GB:
   python -m genmusic_vn.cli make-large-dataset --target-gb 5 --out datasets/training/diverse_5gb --shard-mb 128 --batch-size 4000 --seed 5602
