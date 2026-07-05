GenMusic VN - Huong dan nhanh

1. Cai Kaggle CLI:
   python3 -m pip install --user -U kaggle

2. Tao file .env hoac .env.local tai thu muc goc project:
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_API_TOKEN=your_kaggle_api_token

3. Chay web local:
   python3 -m genmusic_vn.server --port 8000

4. Mo trinh duyet:
   http://127.0.0.1:8000

5. Nhap van ban tieng Viet va bam Generate MP3.
   Local chi gui text len Kaggle.
   Kaggle chay pipeline AI + MusicGen va tra ve file MP3.

6. Chay test:
   python3 -m unittest discover -s tests -v

Xem README.md de biet day du kien truc va cach hoat dong.
