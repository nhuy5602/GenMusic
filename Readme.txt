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

Tạo và upload dataset khoảng 1 GB:
  python -m genmusic_vn.cli make-and-upload-dataset --out datasets/random_self_diffusion_training --target-gb 1
  (Có thể đổi dung lượng bằng --target-gb 5 hoặc đổi đường dẫn bằng --dataset-ref owner/slug.)

Web:
  python -m genmusic_vn.server --port 8000
  http://127.0.0.1:8000

Đánh giá:
  python -m genmusic_vn.cli evaluate-self --generated outputs/local_self_music/final.wav --out outputs/self_evaluation

Tự improve 10 input:
  python -m genmusic_vn.cli self-improve --dataset datasets/random_self_diffusion_1gb --checkpoint outputs/self_music_1gb_subset.pt --out outputs/self_improve_10 --rounds 10 --duration 4 --steps 4 --max-records 64
  (Mặc định chỉ giữ checkpoint cuối; thêm --keep-artifacts nếu cần giữ chẩn đoán.)

Model random chỉ dùng smoke test. Muốn cải thiện chất lượng cần train trên dataset audio/mel và lyric có quyền sử dụng.
