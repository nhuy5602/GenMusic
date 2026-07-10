# Bố Cục Dataset

## Các Thư Mục

- `evaluation/`: benchmark JSONL và các case đánh giá an toàn.
- `training/`: record huấn luyện tự sinh hoặc có license. Dataset shard lớn chỉ lưu local và được Git bỏ qua.
- `trained_models/`: artifact text model bootstrap được commit, dùng khi chưa có model local mới hơn.
- `vn_music_stylebank/`: tài nguyên gọn về nhạc cụ, thể loại, âm nhạc và mẫu lời được pipeline sử dụng.
- `sources/`: chỉ chứa manifest nguồn. Manifest phải ghi URL, license và quyền phê duyệt rõ ràng trước khi crawler fetch.

