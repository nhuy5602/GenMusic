GenMusic VN dùng trực tiếp ASLP-lab/DiffRhythm.

Cài đặt:
  pip install -e ".[diffrhythm]"

Chạy web:
  python -m genmusic_vn.server --port 8000
  http://127.0.0.1:8000

Stage job DiffRhythm:
  python -m genmusic_vn.cli generate --text "Một đêm mưa, em còn nhớ con đường cũ." --duration 95 --no-submit

Tạo random dataset đúng format upstream:
  python -m genmusic_vn.cli make-random-diffrhythm-dataset --out datasets/random_diffrhythm --count 4 --max-frames 64
  python -m genmusic_vn.cli validate-diffrhythm-dataset --dataset datasets/random_diffrhythm

Train official train/train.py:
  python -m genmusic_vn.cli train-diffrhythm --dataset datasets/random_diffrhythm --epochs 1 --batch-size 1

Inference local:
  python -m genmusic_vn.cli generate-local-diffrhythm --lyrics data/song.txt --style "Vietnamese pop ballad, piano" --duration 95 --out outputs/local_diffrhythm

Đánh giá khách quan, bỏ qua MOS:
  python -m genmusic_vn.cli evaluate-diffrhythm --generated outputs/generated.wav --out outputs/diffrhythm_evaluation

Nếu máy thiếu torch/torchaudio thì chạy random train trên Kaggle GPU. Project không dùng custom Transformer, classifier, symbolic composer hoặc TTS add-on làm pipeline chính.
