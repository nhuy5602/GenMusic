from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


EMOTION_SCENES: dict[str, dict[str, Any]] = {
    "joy": {
        "vocal": "female",
        "keywords": ["tiếng cười", "nắng", "ngày mới", "con đường", "mở cửa"],
        "phrases": ["tiếng cười", "ngày mới"],
        "sentences": [
            "Tiếng cười vang lên dưới nắng sớm.",
            "Mọi người mở cửa đón một ngày mới rất trong.",
            "Con đường phía trước sáng lên như vừa được gọi tên.",
            "Tôi muốn giữ lại khoảnh khắc rộn ràng ấy trong tim.",
        ],
    },
    "sadness": {
        "vocal": "female",
        "keywords": ["mưa", "đêm", "nỗi nhớ", "lời chưa nói", "rời xa"],
        "phrases": ["mưa", "nỗi nhớ"],
        "sentences": [
            "Đêm xuống chậm, tiếng mưa rơi lên khung cửa.",
            "Có một lời chưa nói cứ nằm lại trong tim.",
            "Người rời xa để căn phòng im lặng hơn mọi ngày.",
            "Tôi nghe nỗi nhớ đi qua rất khẽ.",
        ],
    },
    "anger": {
        "vocal": "male",
        "keywords": ["bất công", "cúi đầu", "siết chặt", "nhịp tim", "trống trận"],
        "phrases": ["bất công", "nhịp tim"],
        "sentences": [
            "Tôi không muốn cúi đầu trước bất công nữa.",
            "Bàn tay siết chặt, nhịp tim dồn lên như trống trận.",
            "Những vết thương cũ hóa thành một lời đáp mạnh mẽ.",
            "Ta bước thẳng qua tiếng ồn và đứng về phía mình.",
        ],
    },
    "fear": {
        "vocal": "female",
        "keywords": ["bóng tối", "căn phòng", "run rẩy", "đường vắng", "ánh sáng"],
        "phrases": ["bóng tối", "ánh sáng"],
        "sentences": [
            "Bóng tối phủ xuống căn phòng rất xa.",
            "Tôi nghe hơi thở mình run rẩy giữa đường vắng.",
            "Một đốm ánh sáng nhỏ vẫn còn ở cuối hành lang.",
            "Ta bước chậm để không đánh rơi lòng can đảm.",
        ],
    },
    "calm": {
        "vocal": "female",
        "keywords": ["bờ sông", "gió", "hàng cây", "tiếng nước", "bình yên"],
        "phrases": ["bờ sông", "gió"],
        "sentences": [
            "Buổi sáng bên bờ sông thật yên.",
            "Gió đi qua hàng cây, mọi lo âu dần tan trong tiếng nước.",
            "Tôi ngồi lại nghe bình yên chạm vào vai áo.",
            "Ngày trôi chậm như một hơi thở nhẹ.",
        ],
    },
    "romantic": {
        "vocal": "duet",
        "keywords": ["anh", "em", "đèn vàng", "lời chào", "thành phố"],
        "phrases": ["đèn vàng", "thành phố"],
        "sentences": [
            "Anh gặp em dưới hàng đèn vàng.",
            "Một lời chào rất khẽ cũng đủ làm thành phố mềm lại.",
            "Bàn tay chưa chạm mà mùa hẹn đã lên tiếng.",
            "Ta giữ yêu thương ở lại trong ánh mắt.",
        ],
    },
    "hope": {
        "vocal": "female",
        "keywords": ["thất bại", "đứng dậy", "ánh sáng", "khởi đầu", "niềm tin"],
        "phrases": ["ánh sáng", "khởi đầu"],
        "sentences": [
            "Sau nhiều ngày thất bại, chúng tôi vẫn đứng dậy.",
            "Cửa sổ mở ra để ánh sáng đi vào căn phòng.",
            "Một khởi đầu mới đang chờ ở phía xa.",
            "Niềm tin nhỏ thôi cũng đủ đưa bước chân đi tiếp.",
        ],
    },
    "nostalgic": {
        "vocal": "female",
        "keywords": ["mưa", "phố cũ", "lời hứa", "ánh đèn", "ngày xưa"],
        "phrases": ["chiều mưa", "ánh đèn"],
        "sentences": [
            "Một chiều mưa, tôi nhớ về những con phố cũ.",
            "Có lời hứa chưa kịp nói, có ánh đèn vẫn chờ trong tim.",
            "Ngày xưa nghiêng lại trong màu nắng cũ.",
            "Ta trở về bằng một câu ca rất khẽ.",
        ],
    },
}

LENGTH_SENTENCE_COUNTS = {
    "short": (1, 2),
    "medium": (4, 6),
    "long": (10, 16),
}


def generate_synthetic_records(
    count: int,
    *,
    seed: int = 42,
    emotions: list[str] | None = None,
    lengths: list[str] | None = None,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    emotion_pool = emotions or list(EMOTION_SCENES)
    length_pool = lengths or list(LENGTH_SENTENCE_COUNTS)
    records: list[dict[str, Any]] = []

    for index in range(start_index, start_index + max(0, count)):
        emotion = emotion_pool[(index - start_index) % len(emotion_pool)]
        length = length_pool[((index - start_index) // max(1, len(emotion_pool))) % len(length_pool)]
        scene = EMOTION_SCENES[emotion]
        sentence_count = rng.randint(*LENGTH_SENTENCE_COUNTS[length])
        sentences = _expand_sentences(scene["sentences"], sentence_count, rng)
        records.append(
            {
                "id": f"synthetic_{index:04d}_{emotion}_{length}",
                "input_text": " ".join(sentences),
                "expected_emotions": [emotion],
                "expected_keywords": scene["keywords"],
                "expected_lyric_phrases": scene["phrases"],
                "expected_vocal_gender": scene["vocal"],
                "length_bucket": length,
                "source": "synthetic_rule_generated",
            }
        )
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return output_path


def _expand_sentences(base_sentences: list[str], sentence_count: int, rng: random.Random) -> list[str]:
    sentences = list(base_sentences)
    while len(sentences) < sentence_count:
        pivot = rng.choice(base_sentences)
        sentences.append(_variation(pivot, rng))
    rng.shuffle(sentences)
    return sentences[:sentence_count]


def _variation(sentence: str, rng: random.Random) -> str:
    prefixes = ["Rồi", "Sau đó", "Trong khoảnh khắc ấy", "Ở cuối câu chuyện"]
    suffixes = ["và mọi thứ ngân lại rất lâu.", "như một nhịp thở mới.", "để lòng người nhớ mãi.", "trong một màu rất riêng."]
    core = sentence.rstrip(".")
    return f"{rng.choice(prefixes)}, {core.lower()} {rng.choice(suffixes)}"
