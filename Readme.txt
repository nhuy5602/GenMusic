GenMusic VN là model text-to-music conditional diffusion tự code.

Cài đặt:
  pip install -e ".[self]"

Tạo dataset và train:
  python -m genmusic_vn.cli make-random-dataset --out datasets/random_self_diffusion --count 16 --frames 128
  python -m genmusic_vn.cli train-self --dataset datasets/random_self_diffusion --checkpoint outputs/self_music.pt --epochs 1 --batch-size 4

Sinh local:
  python -m genmusic_vn.cli generate-local --text "Mưa rơi nhẹ nhàng, em còn nhớ con đường xưa." --style "Vietnamese pop ballad, warm piano" --duration 4 --checkpoint outputs/self_music.pt --out outputs/local_self_music

Stage Kaggle:
  python -m genmusic_vn.cli generate --text "Một ngày mới bắt đầu." --duration 12 --no-submit

Web:
  python -m genmusic_vn.server --port 8000
  http://127.0.0.1:8000

Đánh giá:
  python -m genmusic_vn.cli evaluate-self --generated outputs/local_self_music/final.wav --out outputs/self_evaluation

Model random chỉ dùng smoke test. Muốn cải thiện chất lượng cần train trên dataset audio/mel và lyric có quyền sử dụng.
