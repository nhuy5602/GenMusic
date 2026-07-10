from __future__ import annotations

import unicodedata
import re

from .schemas import EmotionProfile, ScenePlan


SCENE_RULES = [
    {
        "label": "rain",
        "raw_phrases": ["mưa", "giọt mưa", "cơn mưa", "trời mưa", "bão"],
        "phrases": ["con mua", "giot mua", "troi mua", "mua roi", "rain", "storm"],
        "prompt_cues": ["rainy atmosphere", "soft rain texture", "wet reflections"],
        "arrangement_cues": ["felt piano droplets", "warm strings under the melody", "gentle reverb"],
        "ambience_layers": ["rain"],
        "mix_cues": ["low rain ambience behind the vocal"],
    },
    {
        "label": "old_street",
        "phrases": ["pho cu", "con pho", "duong cu", "ngo cu", "hem nho", "street"],
        "prompt_cues": ["nostalgic old streets", "distant city ambience", "memory haze"],
        "arrangement_cues": ["soft piano motif", "muted guitar pulse", "wide background pads"],
        "ambience_layers": ["street"],
        "mix_cues": ["wide stereo city depth"],
    },
    {
        "label": "city_lights",
        "phrases": ["anh den", "den vang", "den duong", "thanh pho", "pho dem", "city light"],
        "prompt_cues": ["lonely city lights", "warm street lamps", "night urban glow"],
        "arrangement_cues": ["electric piano shimmer", "soft sustained strings", "slow cinematic pulse"],
        "ambience_layers": ["street"],
        "mix_cues": ["gentle stereo reflections"],
    },
    {
        "label": "night",
        "raw_phrases": ["đêm", "khuya", "bóng tối", "trăng", "ngôi sao"],
        "phrases": ["dem khuya", "bong toi", "anh trang", "night"],
        "prompt_cues": ["quiet night mood", "deep blue atmosphere", "intimate late-night space"],
        "arrangement_cues": ["low warm pad", "sparse piano", "slow breathing pauses"],
        "ambience_layers": ["night"],
        "mix_cues": ["dark but clean ambience"],
    },
    {
        "label": "morning_sun",
        "raw_phrases": ["nắng", "bình minh", "sáng sớm", "mặt trời", "ngày mới"],
        "phrases": ["binh minh", "sang som", "mat troi", "ngay moi", "morning", "sun"],
        "prompt_cues": ["soft morning light", "open hopeful air", "sunlit warmth"],
        "arrangement_cues": ["bright piano", "light acoustic guitar", "gentle rising strings"],
        "ambience_layers": ["air"],
        "mix_cues": ["clear open stereo image"],
    },
    {
        "label": "river_sea",
        "raw_phrases": ["sông", "biển", "sóng", "bờ sông", "con sông"],
        "phrases": ["song nuoc", "bo song", "con song", "song bien", "wave", "river", "sea"],
        "prompt_cues": ["flowing water atmosphere", "wide horizon", "gentle wave motion"],
        "arrangement_cues": ["arpeggiated piano flow", "soft strings swell", "airy flute line"],
        "ambience_layers": ["water"],
        "mix_cues": ["smooth wide ambience"],
    },
    {
        "label": "nature_wind",
        "raw_phrases": ["gió", "hàng cây", "cánh đồng", "rừng", "đồi", "núi"],
        "phrases": ["hang cay", "canh dong", "rung cay", "doi nui", "forest", "wind"],
        "prompt_cues": ["natural wind and open air", "pastoral Vietnamese color", "organic calm space"],
        "arrangement_cues": ["sao truc flute air", "dan tranh texture", "nylon guitar"],
        "ambience_layers": ["air"],
        "mix_cues": ["airy natural reverb"],
    },
    {
        "label": "home_room",
        "raw_phrases": ["nhà", "căn phòng", "mái hiên", "cửa sổ", "bếp"],
        "phrases": ["can phong", "mai hien", "cua so", "room", "home"],
        "prompt_cues": ["close indoor intimacy", "quiet room", "personal memory"],
        "arrangement_cues": ["felt piano close mic", "soft pad", "minimal percussion"],
        "ambience_layers": ["room"],
        "mix_cues": ["warm intimate vocal space"],
    },
    {
        "label": "love_promise",
        "raw_phrases": ["yêu", "thương", "lời hứa", "nhớ em", "nhớ anh", "trái tim"],
        "phrases": ["loi hua", "nho em", "nho anh", "trai tim", "heart", "love"],
        "prompt_cues": ["unspoken promise", "gentle longing", "romantic restraint"],
        "arrangement_cues": ["heartfelt piano voicing", "warm strings", "voice-like dan bau accent"],
        "ambience_layers": [],
        "mix_cues": ["vocal-forward but not dry"],
    },
    {
        "label": "hope_rise",
        "phrases": ["hy vong", "dung day", "vuot qua", "ngay mai", "niem tin", "mo ra", "hope"],
        "prompt_cues": ["hopeful lift", "small light growing", "forward motion"],
        "arrangement_cues": ["gradual build", "rising strings", "open chorus lift"],
        "ambience_layers": ["air"],
        "mix_cues": ["bright clean lift"],
    },
    {
        "label": "epic_victory",
        "raw_phrases": ["chiến thắng", "hào hùng", "sân vận động", "lá cờ", "chúng ta thắng", "đội"],
        "phrases": ["chien thang", "hao hung", "san van dong", "la co", "chung ta thang", "team", "victory", "epic", "heroic", "anthem"],
        "prompt_cues": ["heroic victory anthem", "team spirit", "sports highlight", "triumphant lift"],
        "arrangement_cues": ["big drums", "heroic brass", "uplifting strings", "powerful percussion", "wide chorus impact"],
        "ambience_layers": ["air"],
        "mix_cues": ["wide anthem scale without clipping"],
    },
    {
        "label": "brand_jingle",
        "raw_phrases": ["thương hiệu", "khẩu hiệu", "đồng hành cùng bạn", "nhanh hơn", "thông minh hơn"],
        "phrases": ["brand_jingle", "brand jingle", "jingle", "tagline"],
        "prompt_cues": ["short catchy upbeat brand jingle", "memorable product hook", "bright logo ending"],
        "arrangement_cues": ["catchy two-bar motif", "claps on the hook", "clean final logo sting"],
        "ambience_layers": ["air"],
        "mix_cues": ["clear commercial polish"],
    },
    {
        "label": "product_launch",
        "raw_phrases": ["sneaker", "sản phẩm", "ra mắt", "thiết kế", "táo bạo", "cá tính"],
        "phrases": ["advertising", "product launch", "product", "launch"],
        "prompt_cues": ["bold modern product launch", "stylish advertising energy", "confident commercial beat"],
        "arrangement_cues": ["tight trap beat", "modern synth hook", "short product reveal drop"],
        "ambience_layers": ["air"],
        "mix_cues": ["punchy modern ad mix"],
    },
    {
        "label": "finance_banking",
        "raw_phrases": ["ngân hàng", "tài chính", "giao dịch", "phê duyệt", "minh bạch", "quy trình"],
        "phrases": ["banking", "finance", "financial", "banking product"],
        "prompt_cues": ["professional banking product", "trustworthy finance presentation", "clean confident business tone"],
        "arrangement_cues": ["steady corporate pulse", "soft synth bed", "precise light percussion"],
        "ambience_layers": ["air"],
        "mix_cues": ["clean professional stereo"],
    },
    {
        "label": "action_chase",
        "raw_phrases": ["chiếc xe", "lao qua", "con hẻm", "đếm ngược", "mười giây"],
        "phrases": ["action chase", "chase", "countdown"],
        "prompt_cues": ["tense action chase", "urgent rock momentum", "countdown pressure"],
        "arrangement_cues": ["driving rock riff", "fast tom fills", "tight stop-start accents"],
        "ambience_layers": ["street"],
        "mix_cues": ["punchy chase sequence without clipping"],
    },
    {
        "label": "station_farewell",
        "raw_phrases": ["sân ga", "ga tàu", "chia tay", "lời tạm biệt", "chuyến tàu"],
        "phrases": ["station farewell", "station", "farewell", "regenerate"],
        "prompt_cues": ["station farewell feeling", "regenerate variation cue", "bittersweet departure scene"],
        "arrangement_cues": ["soft piano departure motif", "distant train-like pulse", "slow string farewell"],
        "ambience_layers": ["street"],
        "mix_cues": ["gentle emotional depth"],
    },
    {
        "label": "fashion_runway",
        "raw_phrases": ["người mẫu", "thiết kế đen", "ánh kim", "sân khấu", "sàn diễn"],
        "phrases": ["fashion runway", "fashion", "runway"],
        "prompt_cues": ["luxury fashion runway", "sleek electronic beat", "high-gloss catwalk energy"],
        "arrangement_cues": ["deep bass pulse", "minimal synth hook", "four-on-the-floor runway kick"],
        "ambience_layers": ["air"],
        "mix_cues": ["wide glossy club-fashion mix"],
    },
    {
        "label": "festival",
        "raw_phrases": ["lễ hội", "pháo hoa", "quảng trường", "tiếng chuông", "cùng hát"],
        "phrases": ["le hoi", "phao hoa", "quang truong", "tieng chuong", "cung hat", "festival", "celebration"],
        "prompt_cues": ["festive community celebration", "bright crowd energy", "grand joyful lights"],
        "arrangement_cues": ["light percussion", "bright brass accents", "big chorus lift"],
        "ambience_layers": ["air"],
        "mix_cues": ["wide celebratory stereo"],
    },
    {
        "label": "fantasy_mystery",
        "raw_phrases": ["huyền bí", "khu rừng cổ", "cổ thụ", "ma thuật", "ánh sáng xanh", "màn sương", "thế giới này", "đồng hồ chạy ngược"],
        "phrases": ["huyen bi", "khu rung co", "co thu", "ma thuat", "anh sang xanh", "man suong", "the gioi nay", "dong ho chay nguoc", "dark fantasy", "fantasy", "mystery", "magical", "surreal"],
        "prompt_cues": ["dark fantasy orchestral cue", "magical mist", "surreal mystery", "quiet wonder"],
        "arrangement_cues": ["low orchestral strings", "harp-like arpeggios", "soft flute", "misty choir pad"],
        "ambience_layers": ["air"],
        "mix_cues": ["spacious mysterious shimmer"],
    },
    {
        "label": "travel_freedom",
        "raw_phrases": ["tự do", "ga tàu", "vali", "thành phố xa lạ", "biển hiệu", "chạy qua cánh đồng"],
        "phrases": ["tu do", "ga tau", "vali", "thanh pho xa la", "bien hieu", "chay qua canh dong", "travel", "freedom", "adventure"],
        "prompt_cues": ["open travel freedom", "forward motion", "curious new city"],
        "arrangement_cues": ["driving acoustic rhythm", "light modern pulse", "open chorus"],
        "ambience_layers": ["air", "street"],
        "mix_cues": ["open stereo movement"],
    },
    {
        "label": "focus_technology",
        "raw_phrases": ["dòng code", "màn hình", "lỗi cuối cùng", "công nghệ", "mạch điện"],
        "phrases": ["dong code", "man hinh", "loi cuoi cung", "cong nghe", "mach dien", "code", "technology", "focus", "electronic"],
        "prompt_cues": ["focused modern technology", "precise nighttime concentration", "electronic city grid"],
        "arrangement_cues": ["clean synth pulse", "minimal beat", "soft digital arpeggio"],
        "ambience_layers": ["night"],
        "mix_cues": ["clean modern stereo image"],
    },
    {
        "label": "healing_survival",
        "raw_phrases": ["chữa lành", "tha thứ", "cơn bão đã qua", "ngọn nến", "vẫn còn ở đây", "nghỉ ngơi"],
        "phrases": ["chua lanh", "tha thu", "con bao da qua", "ngon nen", "van con o day", "nghi ngoi", "healing", "survival"],
        "prompt_cues": ["gentle healing after hardship", "small warm light", "survival hope"],
        "arrangement_cues": ["soft piano", "warm pad", "slow rising strings"],
        "ambience_layers": ["room", "air"],
        "mix_cues": ["warm supportive space"],
    },
    {
        "label": "conflict_fire",
        "phrases": ["gian", "bat cong", "lua", "chien", "dau tranh", "vo tan", "anger", "fight"],
        "prompt_cues": ["controlled tension", "dark urgent emotion", "restless pulse"],
        "arrangement_cues": ["tight low drums", "dark strings", "short minor motif"],
        "ambience_layers": [],
        "mix_cues": ["punchy but unclipped"],
    },
    {
        "label": "fear_shadow",
        "raw_phrases": ["sợ", "lo lắng", "run rẩy", "bóng tối", "linh cảm", "không lành"],
        "phrases": ["lo lang", "run ray", "duong vang", "bong toi", "linh cam", "khong lanh", "mat hut", "fear"],
        "prompt_cues": ["shadowy suspense", "uncertain pulse", "cold distant space"],
        "arrangement_cues": ["low drone", "prepared piano", "thin strings"],
        "ambience_layers": ["night"],
        "mix_cues": ["dark spacious tension"],
    },
]


EMOTION_FALLBACKS = {
    "joy": {
        "prompt_cues": ["bright Vietnamese pop mood", "fresh optimistic motion"],
        "arrangement_cues": ["light drums", "bright piano", "acoustic guitar"],
        "mix_cues": ["clean bright stereo"],
    },
    "sadness": {
        "prompt_cues": ["Vietnamese melancholic ballad", "tender restrained emotion"],
        "arrangement_cues": ["soft piano", "warm strings", "slow tempo"],
        "mix_cues": ["gentle reverb without distortion"],
    },
    "anger": {
        "prompt_cues": ["dark intense cinematic pop", "controlled aggression"],
        "arrangement_cues": ["tight percussion", "low strings", "minor motif"],
        "mix_cues": ["clear punch without clipping"],
    },
    "fear": {
        "prompt_cues": ["cold suspenseful atmosphere", "unresolved tension"],
        "arrangement_cues": ["low drone", "thin strings", "sparse piano"],
        "mix_cues": ["wide dark space"],
    },
    "calm": {
        "prompt_cues": ["peaceful Vietnamese acoustic mood", "slow breathing warmth"],
        "arrangement_cues": ["felt piano", "nylon guitar", "soft pad"],
        "mix_cues": ["warm natural stereo"],
    },
    "romantic": {
        "prompt_cues": ["romantic Vietnamese ballad", "soft heartfelt longing"],
        "arrangement_cues": ["warm strings", "felt piano", "gentle guitar"],
        "mix_cues": ["intimate vocal space"],
    },
    "hope": {
        "prompt_cues": ["hopeful cinematic lift", "open emotional horizon"],
        "arrangement_cues": ["rising strings", "piano pulse", "light drums"],
        "mix_cues": ["wide clean lift"],
    },
    "nostalgic": {
        "prompt_cues": ["nostalgic Vietnamese memory", "bittersweet warmth"],
        "arrangement_cues": ["upright piano", "muted guitar", "soft strings"],
        "mix_cues": ["warm tape-like depth"],
    },
}


def build_scene_plan(text: str, emotion: EmotionProfile) -> ScenePlan:
    lowered = text.lower()
    folded = _fold(text)
    labels: list[str] = []
    prompt_cues: list[str] = []
    arrangement_cues: list[str] = []
    ambience_layers: list[str] = []
    mix_cues: list[str] = []

    for rule in SCENE_RULES:
        raw_phrases = rule.get("raw_phrases", [])
        if any(_contains_phrase(lowered, phrase) for phrase in raw_phrases) or any(_contains_phrase(folded, phrase) for phrase in rule["phrases"]):
            labels.append(str(rule["label"]))
            prompt_cues.extend(rule["prompt_cues"])
            arrangement_cues.extend(rule["arrangement_cues"])
            ambience_layers.extend(rule["ambience_layers"])
            mix_cues.extend(rule["mix_cues"])

    fallback = EMOTION_FALLBACKS.get(emotion.label, EMOTION_FALLBACKS["calm"])
    prompt_cues.extend(fallback["prompt_cues"])
    arrangement_cues.extend(fallback["arrangement_cues"])
    mix_cues.extend(fallback["mix_cues"])

    if not labels:
        labels.append(f"emotion_{emotion.label}")

    return ScenePlan(
        labels=_dedupe(labels),
        prompt_cues=_dedupe(prompt_cues)[:10],
        arrangement_cues=_dedupe(arrangement_cues)[:10],
        ambience_layers=_dedupe(ambience_layers)[:4],
        mix_cues=_dedupe(mix_cues)[:6],
    )


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


def _contains_phrase(text: str, phrase: str) -> bool:
    cleaned = phrase.strip().lower()
    if not cleaned:
        return False
    if " " in cleaned:
        return cleaned in text
    return re.search(rf"(?<![^\W\d_]){re.escape(cleaned)}(?![^\W\d_])", text, flags=re.UNICODE) is not None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
