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
   Local chi gui text len Kaggle.
   Kaggle chay pipeline AI + MusicGen + F5-TTS Vietnamese va tra ve file MP3.
   MMS Vietnamese TTS duoc giu lam fallback neu F5-TTS loi.

6. Chay test:
   python -m unittest discover -s tests -v

Xem README.md de biet day du kien truc va cach hoat dong.
