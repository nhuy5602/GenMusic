# Bố Cục Dataset

## Các Thư Mục

- `evaluation/`: benchmark JSONL và các case đánh giá an toàn.
- `training/`: record huấn luyện tự sinh hoặc có license. Dataset shard lớn chỉ lưu local và được Git bỏ qua.
- `trained_models/`: artifact text model bootstrap được commit, dùng khi chưa có model local mới hơn.
- `vn_music_stylebank/`: tài nguyên gọn về nhạc cụ, thể loại, âm nhạc và mẫu lời được pipeline sử dụng.
- `sources/`: chỉ chứa manifest nguồn. Manifest phải ghi URL, license và quyền phê duyệt rõ ràng trước khi crawler fetch.
- `incoming/`: vùng nhận ZIP dataset do người dùng cung cấp, chỉ dùng local và được Git bỏ qua.

## ZIP Lyric Và MP3 Trong Tương Lai

Khi có bộ dữ liệu được cấp phép, đặt ZIP dưới `datasets/incoming/` và giữ license/readme cạnh dữ liệu. Bố cục khuyến nghị:

```text
collection.zip
  metadata.jsonl
  lyrics/<song_id>.json
  audio/<song_id>.mp3
```

Mỗi dòng metadata nên có `song_id`, `section_type` (`verse` hoặc `chorus`), `license`, `source` và mối liên hệ giữa section lyric với MP3. Không trộn bộ dữ liệu vào thư mục gốc và không commit file audio.
