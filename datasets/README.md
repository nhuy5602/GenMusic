# Dataset Layout

## Directories

- `evaluation/`: JSONL benchmarks and safe evaluation cases.
- `training/`: generated or licensed training records. Large shard datasets stay local and are ignored by Git.
- `trained_models/`: the committed bootstrap text-model artifact used when no newer local model exists.
- `vn_music_stylebank/`: compact music, instrument, genre, and lyric-pattern resources shipped with the pipeline.
- `sources/`: source manifests only. A manifest must record URL, license, and explicit approval before the crawler can fetch anything.
- `incoming/`: local-only drop zone for future user-provided ZIP datasets. It is ignored by Git.

## Future Lyric + MP3 ZIP

When a licensed collection is available, place the ZIP under `datasets/incoming/` and keep its license/readme alongside the data. The preferred layout is:

```text
collection.zip
  metadata.jsonl
  lyrics/<song_id>.json
  audio/<song_id>.mp3
```

Each metadata row should include `song_id`, `section_type` (`verse` or `chorus`), `license`, `source`, and the relationship between the lyric section and MP3. Do not mix the collection into the repository root or commit the audio files.
