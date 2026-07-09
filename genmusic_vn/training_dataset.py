from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .synthetic_dataset import EMOTION_SCENES


GENRE_SCENES: dict[str, dict[str, Any]] = {
    "pop_ballad": {
        "style_prompt": "Vietnamese melancholic pop ballad, soft piano, warm strings, slow tempo, intimate vocal",
        "emotions": ["sadness", "romantic", "nostalgic", "calm"],
        "keywords": ["mưa", "phố cũ", "piano", "dây đàn", "lời hứa"],
        "sentences": [
            "Tiếng piano rơi chậm trên con phố mưa.",
            "Một lời hứa cũ vẫn còn sáng trong tim.",
            "Giọng hát cần mềm, gần và nhiều khoảng thở.",
            "Dây đàn phía sau nâng câu chuyện buồn lên rất khẽ.",
        ],
    },
    "trap": {
        "style_prompt": "Vietnamese melodic trap, 808 bass, crisp hi-hats, dark synth, confident rap hook",
        "emotions": ["hope", "anger", "joy"],
        "keywords": ["808", "hi-hat", "rap", "đường phố", "đứng lên"],
        "sentences": [
            "Nhịp 808 nảy dưới ánh đèn thành phố.",
            "Tôi đứng dậy sau từng lần bị kéo xuống.",
            "Flow cần chắc, ít ngân dài và có hook mạnh.",
            "Hi-hat chạy nhanh như bước chân qua đêm.",
        ],
    },
    "edm": {
        "style_prompt": "Vietnamese festival EDM, bright synth lead, four-on-the-floor kick, energetic drop, wide stereo",
        "emotions": ["joy", "hope"],
        "keywords": ["drop", "synth", "lễ hội", "đám đông", "ánh sáng"],
        "sentences": [
            "Đám đông bật lên khi drop mở rộng bầu trời.",
            "Synth lead sáng như pháo hoa trong đêm.",
            "Nhịp kick đều kéo mọi người cùng hát.",
            "Câu hook cần ngắn, vui và dễ nhớ.",
        ],
    },
    "folk": {
        "style_prompt": "Vietnamese folk ballad, dan bau, bamboo flute, acoustic guitar, gentle percussion, nostalgic countryside",
        "emotions": ["nostalgic", "calm", "romantic"],
        "keywords": ["đàn bầu", "sáo trúc", "quê nhà", "bờ tre", "dòng sông"],
        "sentences": [
            "Sáo trúc đưa hương lúa qua bờ sông cũ.",
            "Đàn bầu ngân một nét quê rất xa.",
            "Mẹ ngồi bên hiên nghe chiều xuống thật chậm.",
            "Tiết tấu dân gian cần mềm và nhiều luyến láy.",
        ],
    },
    "rock": {
        "style_prompt": "Vietnamese arena rock, electric guitar, live drums, bass guitar, powerful chorus, high energy",
        "emotions": ["anger", "hope", "joy"],
        "keywords": ["guitar điện", "trống live", "bùng nổ", "không lùi", "sân khấu"],
        "sentences": [
            "Guitar điện mở riff như một lời tuyên bố.",
            "Trống live đẩy câu hát đi thẳng về phía trước.",
            "Tôi không lùi bước giữa tiếng ồn của đám đông.",
            "Chorus cần lớn, mạnh và có cảm giác sân khấu.",
        ],
    },
    "rnb": {
        "style_prompt": "Vietnamese smooth R&B, electric piano, soft bass, snap drums, late-night romantic groove",
        "emotions": ["romantic", "calm", "nostalgic"],
        "keywords": ["R&B", "electric piano", "snap", "đêm muộn", "groove"],
        "sentences": [
            "Electric piano tan trong ly cà phê đêm.",
            "Snap drums giữ nhịp nhẹ như một lời thì thầm.",
            "Giọng hát nên mượt, gần và hơi lả lơi.",
            "Bass mềm ôm lấy khoảng lặng giữa hai người.",
        ],
    },
    "bolero": {
        "style_prompt": "Vietnamese bolero, tremolo acoustic guitar, slow sentimental rhythm, dan bau ornament",
        "emotions": ["sadness", "nostalgic", "romantic"],
        "keywords": ["bolero", "tremolo", "mưa đêm", "tình lỡ", "quán cũ"],
        "sentences": [
            "Guitar tremolo rơi đều trong quán nhỏ.",
            "Tình lỡ đi qua một đêm mưa rất dài.",
            "Câu hát cần chậm, rõ chữ và nhiều tiếc nuối.",
            "Đàn bầu điểm nhẹ ở cuối mỗi câu ngân.",
        ],
    },
    "ambient": {
        "style_prompt": "Vietnamese ambient soundscape, warm drone pad, soft texture, distant bell, minimal pulse",
        "emotions": ["calm", "fear", "nostalgic"],
        "keywords": ["ambient", "drone", "chuông xa", "không gian", "tĩnh"],
        "sentences": [
            "Một lớp drone mở ra như sương trên mặt hồ.",
            "Tiếng chuông xa giữ không gian rộng và tĩnh.",
            "Nhạc không cần nhiều nhịp, chỉ cần hơi thở.",
            "Âm nền nên trôi chậm và không che lời hát.",
        ],
    },
    "orchestral": {
        "style_prompt": "Vietnamese orchestral cinematic trailer, strings, brass, choir pad, heroic percussion",
        "emotions": ["hope", "fear", "anger"],
        "keywords": ["orchestral", "strings", "brass", "trailer", "anh hùng"],
        "sentences": [
            "Dàn strings kéo bầu trời mở ra trước mắt.",
            "Brass nâng cao khoảnh khắc chiến thắng.",
            "Trống điện ảnh dồn nhịp như một đoàn quân.",
            "Câu chuyện cần cảm giác lớn và có cao trào.",
        ],
    },
    "horror": {
        "style_prompt": "Vietnamese horror score, low strings, dark pads, distant hit, sub drone, no cheerful melody",
        "emotions": ["fear"],
        "keywords": ["bóng tối", "horror", "low strings", "bất an", "tiếng động xa"],
        "sentences": [
            "Low strings bò dưới sàn nhà tối.",
            "Một tiếng động xa làm căn phòng lạnh đi.",
            "Không khí cần căng, ít giai điệu vui.",
            "Sub drone giữ cảm giác bất an đến cuối đoạn.",
        ],
    },
    "lofi": {
        "style_prompt": "Vietnamese lo-fi chillhop, dusty electric piano, vinyl tape noise, muted guitar, soft lo-fi drums",
        "emotions": ["calm", "nostalgic", "sadness"],
        "keywords": ["lo-fi", "vinyl", "chill", "đêm học", "mưa nhẹ"],
        "sentences": [
            "Vinyl noise phủ nhẹ lên căn phòng học đêm.",
            "Electric piano bụi và ấm giữ nhịp chậm.",
            "Mưa ngoài cửa làm câu hát mềm hơn.",
            "Beat cần chill, không quá dày và dễ lặp.",
        ],
    },
}


MOOD_ALIASES = {
    "happy": "joy",
    "playful": "joy",
    "sad": "sadness",
    "melancholic": "sadness",
    "angry": "anger",
    "scary": "fear",
    "mysterious": "fear",
    "peaceful": "calm",
    "professional": "calm",
    "romantic": "romantic",
    "hopeful": "hope",
    "inspiring": "hope",
    "nostalgic": "nostalgic",
}

ANCHOR_TRAINING_RECORDS: list[dict[str, Any]] = [
    {
        "id": "anchor_pop_ballad_rain_old_street",
        "input_text": "Một chiều mưa, tôi nhớ về những con phố cũ. Có lời hứa chưa kịp nói, có ánh đèn vẫn chờ trong tim.",
        "emotion": "nostalgic",
        "genre_label": "pop_ballad",
        "style_prompt": GENRE_SCENES["pop_ballad"]["style_prompt"],
        "expected_keywords": ["mưa", "phố cũ", "lời hứa", "piano", "strings"],
        "expected_vocal_gender": "female",
        "source": "anchor_training_sample",
    },
    {
        "id": "anchor_pop_ballad_unspoken_promise",
        "input_text": "Mưa rơi qua phố quen, lời chưa nói còn nằm lại trong tim. Tiếng piano cần chậm và buồn.",
        "emotion": "sadness",
        "genre_label": "pop_ballad",
        "style_prompt": GENRE_SCENES["pop_ballad"]["style_prompt"],
        "expected_keywords": ["mưa", "phố quen", "piano", "buồn"],
        "expected_vocal_gender": "female",
        "source": "anchor_training_sample",
    },
    {
        "id": "anchor_rnb_late_night_cafe",
        "input_text": "Ly cà phê còn hơi ấm, giọng em trôi qua rất chậm. Snap drums và electric piano giữ groove đêm muộn.",
        "emotion": "romantic",
        "genre_label": "rnb",
        "style_prompt": GENRE_SCENES["rnb"]["style_prompt"],
        "expected_keywords": ["cà phê", "snap", "electric piano", "groove"],
        "expected_vocal_gender": "duet",
        "source": "anchor_training_sample",
    },
    {
        "id": "anchor_folk_countryside",
        "input_text": "Sáo chiều đưa hương lúa xa, mẹ ngồi bên hiên chờ ta. Đàn bầu ngân như dòng sông quê.",
        "emotion": "nostalgic",
        "genre_label": "folk",
        "style_prompt": GENRE_SCENES["folk"]["style_prompt"],
        "expected_keywords": ["sáo", "hương lúa", "đàn bầu", "quê"],
        "expected_vocal_gender": "female",
        "source": "anchor_training_sample",
    },
    {
        "id": "anchor_rock_no_retreat",
        "input_text": "Đập tan im lặng trong tim, đứng lên qua từng vết xước. Guitar điện và trống live làm chorus bùng nổ.",
        "emotion": "hope",
        "genre_label": "rock",
        "style_prompt": GENRE_SCENES["rock"]["style_prompt"],
        "expected_keywords": ["guitar điện", "trống live", "chorus", "bùng nổ"],
        "expected_vocal_gender": "male",
        "source": "anchor_training_sample",
    },
    {
        "id": "anchor_edm_festival",
        "input_text": "Nắng lên trên môi cười, đêm nay không ai đứng yên. Synth lead sáng và drop lễ hội mở rộng stereo.",
        "emotion": "joy",
        "genre_label": "edm",
        "style_prompt": GENRE_SCENES["edm"]["style_prompt"],
        "expected_keywords": ["synth", "drop", "lễ hội", "stereo"],
        "expected_vocal_gender": "female",
        "source": "anchor_training_sample",
    },
    {
        "id": "anchor_trap_city",
        "input_text": "Bật lên giữa thành phố tối, tim không ngủ trên con đường mới. 808 bass và hi-hat giữ rap hook chắc.",
        "emotion": "hope",
        "genre_label": "trap",
        "style_prompt": GENRE_SCENES["trap"]["style_prompt"],
        "expected_keywords": ["808", "hi-hat", "rap", "hook"],
        "expected_vocal_gender": "male",
        "source": "anchor_training_sample",
    },
]


# These pools are deliberately independent from the small hand-written scene banks above.
# Their combinations give the local classifier more linguistic coverage without copying
# lyrics from the web.
DIVERSE_CONTEXTS = [
    "một sân thượng nhìn xuống thành phố lúc nửa đêm",
    "ga tàu vắng sau chuyến cuối cùng",
    "con hẻm nhỏ vừa tắt cơn mưa",
    "bờ biển trước khi trời đổi sáng",
    "căn phòng có ánh đèn xanh từ cửa sổ",
    "một chuyến xe đi qua những cánh đồng",
    "quán cà phê còn mở sau giờ đóng cửa",
    "sân khấu ngoài trời trước lúc khán giả vào",
    "con đường núi phủ sương đầu ngày",
    "khu phố mới thức dậy sau một đêm dài",
    "mái hiên cũ nhìn ra hàng cây",
    "lối đi ven sông có tiếng chuông xa",
    "phòng tập đầy dây đàn và micro",
    "bến phà chờ chuyến nước rút",
    "ngã tư đầy biển hiệu và xe đêm",
    "khoảng sân sau một buổi lễ mùa hè",
    "căn nhà quê nghe gió qua hàng tre",
    "một căn phòng im lặng trước cơn bão",
    "đường hầm có tiếng bước chân vọng lại",
    "ban công nhìn xuống dòng người tan ca",
    "một buổi chiều trên chuyến tàu trở về",
    "khu vườn còn đọng sương sau bình minh",
    "mặt hồ phẳng dưới mây xám",
    "sàn nhảy sáng đèn giữa đêm lễ hội",
]

DIVERSE_ACTIONS = [
    "nhặt lại một lời chưa kịp nói",
    "bước tiếp sau một lần hụt chân",
    "giữ nhịp thở giữa đám đông",
    "gọi tên người đã đi xa",
    "mở cánh cửa cho ngày mới",
    "để ký ức trôi theo mặt nước",
    "đứng trước nỗi sợ mà không lùi",
    "chạm vào bàn tay đang chờ",
    "gom ánh sáng từ những điều nhỏ bé",
    "thả một câu hát vào khoảng không",
    "đổi im lặng thành nhịp trống",
    "đi qua thành phố bằng đôi chân mỏi",
    "giữ một mùa cũ trong ánh mắt",
    "lắng nghe tiếng nhà ở phía xa",
    "đợi một dấu hiệu sau màn mưa",
    "bật cười khi bình minh vừa đến",
]

DIVERSE_OPENERS = [
    "Hãy tạo một bản nhạc kể về {context}, trong đó nhân vật muốn {action}.",
    "Bản demo cần bắt đầu từ {context}; câu chuyện xoay quanh việc {action}.",
    "Người dùng mô tả {context} và muốn lời hát dẫn đến khoảnh khắc {action}.",
    "Viết ý tưởng ca khúc lấy {context} làm nền, với một nhân vật đang {action}.",
    "Tạo một màn trình diễn có hình ảnh {context}, rồi đẩy cảm xúc về phía {action}.",
    "Ca khúc nên mở ra từ {context}; phần hook cần gợi cảm giác {action}.",
    "Hãy biến {context} thành một câu chuyện âm nhạc về việc {action}.",
    "Prompt này cần một bài hát có điểm nhìn ở {context}, không khí dẫn tới việc {action}.",
]

DIVERSE_VOCAL_HINTS = [
    "vocal chính phát âm rõ từng chữ",
    "giọng hát gần microphone, không bị chìm trong beat",
    "hook dễ hát lại và có khoảng thở hợp lý",
    "vocal có cao trào nhưng vẫn giữ lời rõ",
    "phần bè chỉ hỗ trợ, không che lead vocal",
    "giữ phụ âm cuối đủ rõ để nghe được câu",
    "để khoảng trống giữa các câu cho giọng hát",
    "không để bản phối biến thành nhạc không lời",
]

DIVERSE_FLOW_HINTS = [
    "flow đều và tự nhiên theo nhịp",
    "flow nói gần, nhấn mạnh âm cuối câu",
    "melody có câu ngắn nối với câu dài",
    "nhịp chữ linh hoạt nhưng không dồn tiếng",
    "hook lặp vừa đủ, verse có chuyển động",
    "tránh nhồi quá nhiều âm tiết vào một ô nhịp",
    "để câu rap bám kick và hi-hat rõ ràng",
    "đường giai điệu đi lên ở cuối phần điệp khúc",
]

DIVERSE_MIX_HINTS = [
    "âm lượng cân bằng, không clipping",
    "giữ vocal sạch và tách khỏi nền",
    "tránh tiếng rè, méo hoặc dội phòng",
    "bass gọn để không lấn lời",
    "treble đủ sáng nhưng không chói",
    "để stereo rộng vừa phải và trung tâm rõ",
    "đoạn chuyển không làm mất nhịp",
    "kết bài không bị cắt đột ngột",
]

DIVERSE_RHYME_HINTS = [
    "các câu cuối có vần gần nhau",
    "ưu tiên vần lưng nhẹ và vần cuối rõ",
    "giữ một mô-típ âm cuối cho hook",
    "vần tự nhiên, không ép từ cho đủ vần",
    "để hai câu liên tiếp có âm vang tương đồng",
    "đổi vần ở verse nhưng giữ hook dễ nhớ",
]


def generate_diverse_training_records(
    count: int,
    *,
    seed: int = 42,
    genres: list[str] | None = None,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Generate high-cardinality original prompts with broad combinatorial variety."""
    rng = random.Random(seed)
    genre_pool = [label for label in (genres or list(GENRE_SCENES)) if label in GENRE_SCENES]
    if not genre_pool:
        genre_pool = list(GENRE_SCENES)
    records: list[dict[str, Any]] = []

    for offset in range(max(0, count)):
        index = start_index + offset
        genre_label = rng.choice(genre_pool)
        genre = GENRE_SCENES[genre_label]
        compatible_emotions = list(genre.get("emotions") or EMOTION_SCENES)
        emotion = rng.choice(compatible_emotions)
        emotion_scene = EMOTION_SCENES[emotion]
        length_bucket = rng.choices(
            ["short", "medium", "long"],
            weights=[0.24, 0.51, 0.25],
            k=1,
        )[0]
        clause_count = {
            "short": rng.randint(2, 4),
            "medium": rng.randint(4, 7),
            "long": rng.randint(7, 11),
        }[length_bucket]
        context = rng.choice(DIVERSE_CONTEXTS)
        action = rng.choice(DIVERSE_ACTIONS)
        parts = [rng.choice(DIVERSE_OPENERS).format(context=context, action=action)]
        parts.append(rng.choice(genre["sentences"]))
        parts.append(rng.choice(emotion_scene["sentences"]))
        while len(parts) < clause_count:
            choice = rng.random()
            if choice < 0.28:
                parts.append(rng.choice(DIVERSE_ACTIONS).capitalize() + ".")
            elif choice < 0.52:
                parts.append(rng.choice(genre["sentences"]))
            elif choice < 0.76:
                parts.append(rng.choice(emotion_scene["sentences"]))
            else:
                parts.append(_variation(rng.choice(genre["sentences"] + emotion_scene["sentences"]), rng))

        if rng.random() < 0.32:
            parts.insert(1, rng.choice(["Mở đầu thật tiết chế rồi mở rộng dần.", "Verse đầu nên để nhiều khoảng thở.", "Đừng đẩy cao trào quá sớm."]))
        parts.append(rng.choice(DIVERSE_VOCAL_HINTS) + ".")
        parts.append(rng.choice(DIVERSE_FLOW_HINTS) + ".")
        parts.append(rng.choice(DIVERSE_RHYME_HINTS) + ".")
        parts.append(rng.choice(DIVERSE_MIX_HINTS) + ".")

        genre_keywords = rng.sample(genre["keywords"], k=min(len(genre["keywords"]), rng.randint(2, 4)))
        emotion_keywords = rng.sample(emotion_scene["keywords"], k=min(len(emotion_scene["keywords"]), rng.randint(2, 4)))
        expected_keywords = _dedupe(genre_keywords + emotion_keywords + ["vocal", "flow", "vần"])
        style_prompt = f"{genre['style_prompt']}, clear Vietnamese lead vocal, singable phrasing, controlled mix"
        records.append(
            {
                "id": f"diverse_{index:06d}_{genre_label}_{emotion}_{length_bucket}",
                "input_text": " ".join(parts),
                "emotion": emotion,
                "genre_label": genre_label,
                "style_prompt": style_prompt,
                "expected_keywords": expected_keywords,
                "expected_vocal_gender": emotion_scene["vocal"],
                "length_bucket": length_bucket,
                "source": "generated_diverse_training_dataset",
            }
        )
    return records


def generate_training_records(
    count: int,
    *,
    seed: int = 42,
    start_index: int = 1,
    genres: list[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    genre_pool = genres or list(GENRE_SCENES)
    records: list[dict[str, Any]] = [dict(record) for record in ANCHOR_TRAINING_RECORDS[: max(0, min(count, len(ANCHOR_TRAINING_RECORDS)))]]
    generated_count = max(0, count - len(records))
    for offset in range(generated_count):
        index = start_index + offset
        genre_label = genre_pool[offset % len(genre_pool)]
        genre = GENRE_SCENES[genre_label]
        emotion = rng.choice(genre["emotions"])
        emotion_scene = EMOTION_SCENES[emotion]
        sentence_count = rng.randint(2, 6)
        text_parts = _compose_training_text(genre["sentences"], emotion_scene["sentences"], sentence_count, rng)
        records.append(
            {
                "id": f"train_{index:05d}_{genre_label}_{emotion}",
                "input_text": " ".join(text_parts),
                "emotion": emotion,
                "genre_label": genre_label,
                "style_prompt": genre["style_prompt"],
                "expected_keywords": _dedupe(list(genre["keywords"]) + list(emotion_scene["keywords"])[:3]),
                "expected_vocal_gender": emotion_scene["vocal"],
                "source": "generated_training_dataset",
            }
        )
    return records


def load_training_records(
    paths: list[str | Path],
    *,
    max_records: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    seen = 0
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        source_paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for source_path in source_paths:
            with source_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    normalized = normalize_training_record(raw)
                    if not normalized:
                        continue
                    if max_records is None or max_records <= 0:
                        records.append(normalized)
                        continue
                    seen += 1
                    if len(records) < max_records:
                        records.append(normalized)
                    else:
                        replacement = rng.randrange(seen)
                        if replacement < max_records:
                            records[replacement] = normalized
    return records


def normalize_training_record(record: dict[str, Any]) -> dict[str, Any] | None:
    text = str(record.get("input_text") or record.get("text") or record.get("chorus") or "").strip()
    if not text:
        return None
    emotion = _emotion_from_record(record)
    genre_label = _genre_from_record(record)
    if not emotion or not genre_label:
        return None
    style_prompt = str(record.get("style_prompt") or record.get("genre") or GENRE_SCENES[genre_label]["style_prompt"]).strip()
    return {
        "id": str(record.get("id") or f"record_{abs(hash(text))}"),
        "input_text": text,
        "emotion": emotion,
        "genre_label": genre_label,
        "style_prompt": style_prompt,
        "expected_keywords": list(record.get("expected_keywords") or GENRE_SCENES[genre_label]["keywords"]),
        "expected_vocal_gender": record.get("expected_vocal_gender", ""),
        "source": str(record.get("source") or "external_training_record"),
    }


def write_training_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return output_path


def style_prompt_for_genre(genre_label: str) -> str:
    return str(GENRE_SCENES.get(genre_label, GENRE_SCENES["pop_ballad"])["style_prompt"])


def _compose_training_text(
    genre_sentences: list[str],
    emotion_sentences: list[str],
    sentence_count: int,
    rng: random.Random,
) -> list[str]:
    pool = list(genre_sentences) + list(emotion_sentences)
    rng.shuffle(pool)
    chosen = pool[:sentence_count]
    while len(chosen) < sentence_count:
        chosen.append(_variation(rng.choice(pool), rng))
    return chosen


def _variation(sentence: str, rng: random.Random) -> str:
    prefixes = ["Rồi", "Trong đoạn hook", "Ở cuối bài", "Giữa cao trào"]
    suffixes = ["để cảm xúc ngân lại.", "và nhịp tim sáng lên.", "như một câu hát dễ nhớ.", "trong màu âm rất riêng."]
    return f"{rng.choice(prefixes)}, {sentence.rstrip('.').lower()} {rng.choice(suffixes)}"


def _emotion_from_record(record: dict[str, Any]) -> str:
    direct = str(record.get("emotion") or "").strip()
    if direct in EMOTION_SCENES:
        return direct
    expected = record.get("expected_emotions")
    if isinstance(expected, list) and expected:
        first = str(expected[0]).strip()
        if first in EMOTION_SCENES:
            return first
    mood_text = f"{record.get('expected_mood_text', '')} {record.get('mood', '')}".lower()
    for marker, label in MOOD_ALIASES.items():
        if marker in mood_text:
            return label
    return ""


def _genre_from_record(record: dict[str, Any]) -> str:
    direct = str(record.get("genre_label") or "").strip()
    if direct in GENRE_SCENES:
        return direct
    blob = f"{record.get('style_prompt', '')} {record.get('genre', '')} {record.get('input_text', '')}".lower()
    checks = [
        ("r&b", "rnb"),
        ("rnb", "rnb"),
        ("trap", "trap"),
        ("rap", "trap"),
        ("edm", "edm"),
        ("dance", "edm"),
        ("folk", "folk"),
        ("dân gian", "folk"),
        ("dan bau", "folk"),
        ("đàn bầu", "folk"),
        ("rock", "rock"),
        ("bolero", "bolero"),
        ("ambient", "ambient"),
        ("orchestral", "orchestral"),
        ("trailer", "orchestral"),
        ("horror", "horror"),
        ("scary", "horror"),
        ("lo-fi", "lofi"),
        ("lofi", "lofi"),
        ("ballad", "pop_ballad"),
        ("piano", "pop_ballad"),
    ]
    for marker, label in checks:
        if marker in blob:
            return label
    return "pop_ballad"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
