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

Xem README.md de biet day du kien truc va cach hoat dong.
