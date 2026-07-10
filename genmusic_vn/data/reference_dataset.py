from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .training_dataset import GENRE_SCENES, normalize_training_record, style_prompt_for_genre


REFERENCE_SONG_CASES: list[dict[str, Any]] = [
    {
        "id": "ref_ballad_rain_window",
        "title": "Khung cửa mưa",
        "genre_label": "pop_ballad",
        "emotion": "sadness",
        "expected_vocal_gender": "female",
        "input_text": "Một người ngồi bên khung cửa mưa, nhớ lời hẹn chưa nói hết và muốn một bản pop ballad piano thật rõ vocal.",
        "reference_lyrics": [
            "mưa nghiêng qua khung cửa nhỏ",
            "lời chưa nói ngủ trong tim",
            "phố quen xa dần trong gió",
            "em gọi tên anh lặng im",
            "xin giữ câu ca ở lại",
            "cho đêm thôi rơi thật dài",
        ],
        "expected_keywords": ["mưa", "khung cửa", "lời hẹn", "piano", "vocal"],
        "expected_lyric_phrases": ["mưa", "lời chưa nói", "trong tim"],
    },
    {
        "id": "ref_folk_river_home",
        "title": "Bến sông nhà",
        "genre_label": "folk",
        "emotion": "nostalgic",
        "expected_vocal_gender": "female",
        "input_text": "Ký ức bến sông quê, tiếng sáo trúc và đàn bầu đưa người xa nhà nhớ mẹ, nhớ hàng tre.",
        "reference_lyrics": [
            "sáo đưa chiều qua bến cũ",
            "đàn bầu ngân phía sau nhà",
            "mẹ ngồi hong mùi rơm mới",
            "con nghe thương nhớ đi xa",
            "hàng tre nghiêng lời ru nhẹ",
            "dòng sông giữ bóng quê nhà",
        ],
        "expected_keywords": ["sáo trúc", "đàn bầu", "quê", "mẹ", "sông"],
        "expected_lyric_phrases": ["đàn bầu", "quê nhà", "dòng sông"],
    },
    {
        "id": "ref_trap_city_rise",
        "title": "Đứng dậy giữa phố",
        "genre_label": "trap",
        "emotion": "hope",
        "expected_vocal_gender": "male",
        "input_text": "Một bạn trẻ vượt qua thất bại giữa thành phố đêm, cần melodic trap, 808 chắc, rap hook rõ chữ và flow tự tin.",
        "reference_lyrics": [
            "đèn phố rọi lên vai áo",
            "ta đứng dậy sau cơn đau",
            "tám không tám đập trong ngực",
            "hi hat dẫn bước qua cầu",
            "hook vang lên không lùi lại",
            "ngày mai mở cửa thật sâu",
        ],
        "expected_keywords": ["808", "hi-hat", "rap", "hook", "đứng dậy"],
        "expected_lyric_phrases": ["đứng dậy", "hook", "ngày mai"],
    },
    {
        "id": "ref_edm_summer_light",
        "title": "Mùa sáng",
        "genre_label": "edm",
        "emotion": "joy",
        "expected_vocal_gender": "female",
        "input_text": "Một đêm lễ hội mùa hè nhiều ánh sáng, mọi người cùng hát, cần EDM festival có drop vui và vocal hook dễ nhớ.",
        "reference_lyrics": [
            "nắng bật lên trong mắt ai",
            "đêm nay ta hát không ngừng",
            "drop rơi xuống như pháo sáng",
            "đám đông nghiêng giữa không trung",
            "câu hook bay qua môi cười",
            "ngày mới reo vang từng vùng",
        ],
        "expected_keywords": ["edm", "drop", "lễ hội", "ánh sáng", "hook"],
        "expected_lyric_phrases": ["drop", "câu hook", "môi cười"],
    },
    {
        "id": "ref_rnb_late_cafe",
        "title": "Sau quán khuya",
        "genre_label": "rnb",
        "emotion": "romantic",
        "expected_vocal_gender": "duet",
        "input_text": "Hai người gặp lại trong quán cà phê đêm, cần R&B mượt, electric piano, snap drums và vocal duet ấm.",
        "reference_lyrics": [
            "ly cà phê còn hơi ấm",
            "mắt em nghiêng phía đèn vàng",
            "snap rơi nhẹ trên nhịp thở",
            "anh nghe đêm bỗng dịu dàng",
            "hai giọng chạm nhau thật khẽ",
            "tình yêu đi giữa mơ màng",
        ],
        "expected_keywords": ["cà phê", "R&B", "electric piano", "snap", "duet"],
        "expected_lyric_phrases": ["cà phê", "đèn vàng", "hai giọng"],
    },
    {
        "id": "ref_rock_fire_stage",
        "title": "Không lùi",
        "genre_label": "rock",
        "emotion": "anger",
        "expected_vocal_gender": "male",
        "input_text": "Một ca khúc rock sân khấu về không cúi đầu trước bất công, guitar điện mạnh, trống live và chorus bùng nổ.",
        "reference_lyrics": [
            "guitar xé ngang màn tối",
            "trống gọi tim đứng thẳng lên",
            "ta không cúi đầu lần nữa",
            "lửa trong ngực vẫn chưa quên",
            "chorus nổ tung sân khấu",
            "vết thương hóa tiếng gọi tên",
        ],
        "expected_keywords": ["rock", "guitar điện", "trống live", "bất công", "chorus"],
        "expected_lyric_phrases": ["không cúi đầu", "chorus", "sân khấu"],
    },
    {
        "id": "ref_bolero_night_station",
        "title": "Ga mưa đêm",
        "genre_label": "bolero",
        "emotion": "nostalgic",
        "expected_vocal_gender": "female",
        "input_text": "Một chuyện tình lỡ ở sân ga mưa đêm, cần bolero chậm, guitar tremolo, vocal rõ từng chữ và nhiều tiếc nuối.",
        "reference_lyrics": [
            "ga khuya nghiêng trong mưa nhỏ",
            "guitar run tiếng mong manh",
            "người đi qua miền thương nhớ",
            "bỏ tôi đứng giữa ga xanh",
            "tremolo rơi đều rất khẽ",
            "tình xưa còn gọi tên anh",
        ],
        "expected_keywords": ["bolero", "tremolo", "mưa đêm", "tình lỡ", "vocal"],
        "expected_lyric_phrases": ["ga khuya", "tremolo", "tình xưa"],
    },
    {
        "id": "ref_horror_dark_room",
        "title": "Phòng cuối hành lang",
        "genre_label": "horror",
        "emotion": "fear",
        "expected_vocal_gender": "female",
        "input_text": "Một căn phòng tối cuối hành lang, tiếng động xa và hơi thở run, cần horror score lạnh, vocal mỏng, không giai điệu vui.",
        "reference_lyrics": [
            "cửa cuối hành lang khép lại",
            "bóng đêm bò dưới chân tường",
            "tiếng ai rơi ngoài ô cửa",
            "hơi thở run giữa màn sương",
            "xin giữ một đốm sáng nhỏ",
            "dẫn tôi qua khỏi đêm trường",
        ],
        "expected_keywords": ["horror", "bóng tối", "hành lang", "run", "không vui"],
        "expected_lyric_phrases": ["hành lang", "bóng đêm", "đốm sáng"],
    },
]


def generate_reference_training_records(
    count: int | None = None,
    *,
    seed: int = 42,
    include_reference_lyrics: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases = list(REFERENCE_SONG_CASES)
    records: list[dict[str, Any]] = []
    target = len(cases) if count is None else max(0, count)
    for index in range(target):
        case = cases[index % len(cases)]
        text = _training_text_from_case(case, include_reference_lyrics=include_reference_lyrics)
        if index >= len(cases):
            text = _vary_text(text, rng)
        records.append(
            {
                "id": f"{case['id']}_train_{index + 1:03d}",
                "input_text": text,
                "emotion": case["emotion"],
                "genre_label": case["genre_label"],
                "style_prompt": style_prompt_for_genre(case["genre_label"]),
                "expected_keywords": list(case["expected_keywords"]),
                "expected_vocal_gender": case["expected_vocal_gender"],
                "source": "curated_original_reference_song_case",
            }
        )
    return records


def generate_reference_eval_records(count: int | None = None, *, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases = list(REFERENCE_SONG_CASES)
    records: list[dict[str, Any]] = []
    target = len(cases) if count is None else max(0, count)
    for index in range(target):
        case = cases[index % len(cases)]
        text = str(case["input_text"])
        if index >= len(cases):
            text = _vary_text(text, rng)
        records.append(
            {
                "id": f"{case['id']}_eval_{index + 1:03d}",
                "input_text": text,
                "expected_emotions": [case["emotion"]],
                "expected_keywords": list(case["expected_keywords"]),
                "expected_lyric_phrases": list(case["expected_lyric_phrases"]),
                "expected_vocal_gender": case["expected_vocal_gender"],
                "genre": GENRE_SCENES[case["genre_label"]]["style_prompt"],
                "genre_label": case["genre_label"],
                "duration_seconds": 30,
                "length_bucket": "reference",
                "source": "curated_original_reference_song_case",
            }
        )
    return records


def load_user_licensed_lyrics_jsonl(paths: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            normalized = normalize_training_record(raw)
            if normalized:
                normalized["source"] = str(raw.get("source") or "user_licensed_lyrics_jsonl")
                records.append(normalized)
    return records


def write_reference_datasets(
    *,
    train_out: str | Path,
    eval_out: str | Path,
    count: int | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    train_path = _write_jsonl(generate_reference_training_records(count, seed=seed), train_out)
    eval_path = _write_jsonl(generate_reference_eval_records(count, seed=seed), eval_out)
    return train_path, eval_path


def _training_text_from_case(case: dict[str, Any], *, include_reference_lyrics: bool) -> str:
    parts = [str(case["input_text"])]
    if include_reference_lyrics:
        parts.append("Lời tham chiếu:\n" + "\n".join(case["reference_lyrics"]))
    return "\n".join(parts)


def _vary_text(text: str, rng: random.Random) -> str:
    prefixes = [
        "Người dùng muốn bản nhạc rõ vocal:",
        "Prompt thử nghiệm cần beat hợp mood:",
        "Ca khúc cần lời đủ câu và có vần:",
        "Bản demo cần flow đúng style:",
    ]
    suffixes = [
        "Ưu tiên hát rõ chữ và tránh rè.",
        "Giữ hook ngắn, dễ nhớ và đúng cảm xúc.",
        "Đừng để output chỉ có nhạc nền.",
        "Beat phải bám mood của câu chuyện.",
    ]
    return f"{rng.choice(prefixes)} {text} {rng.choice(suffixes)}"


def _write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return output_path
