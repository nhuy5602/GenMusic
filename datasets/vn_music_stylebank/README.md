# Dataset Kiến Thức Âm Nhạc Việt Nam

Đây là dataset tri thức dùng cho pipeline GenMusic VN trên Kaggle.

Dataset này không phải tập audio để train model. Nó là stylebank có cấu trúc, dùng để hướng dẫn các bước trước khi gọi MusicGen:

- ánh xạ cảm xúc sang màu âm nhạc
- chọn BPM, key, scale và vòng hợp âm
- chọn nhạc cụ Việt Nam như đàn tranh, đàn bầu, sáo trúc, đàn nhị, trống cơm
- chọn template thể loại
- chọn hình ảnh lời hát, chorus và bridge
- bổ sung keyword cho prompt MusicGen

Các file chính:

```text
emotion_to_music.json       # cảm xúc -> BPM/key/scale/chord/instrument
vietnamese_instruments.json # mô tả nhạc cụ Việt và prompt token
genre_templates.json        # template thể loại
chord_presets.json          # preset hợp âm
lyric_patterns.json         # pattern lời hát theo cảm xúc
```

Khi submit job, folder này được đóng gói vào `genmusic_vn_source.zip` và upload lên Kaggle cùng request.

