from __future__ import annotations

import re


CONTROL_KEYS = {
    "style",
    "mood",
    "secondary mood",
    "genre",
    "instruments",
    "energy",
    "use case",
    "negative prompt",
    "tempo",
    "bpm",
}
CONTROL_RE = re.compile(
    r"(?P<key>style|mood|secondary mood|genre|instruments|energy|use case|negative prompt|tempo|bpm)\s*:\s*(?P<value>[^;]+)",
    re.IGNORECASE,
)


def parse_control_context(text: str | None) -> dict[str, str]:
    controls: dict[str, str] = {}
    for match in CONTROL_RE.finditer(text or ""):
        key = " ".join(match.group("key").lower().split())
        value = " ".join(match.group("value").strip().split())
        if value:
            controls[key] = value
    return controls


def positive_control_text(text: str | None) -> str:
    if not text:
        return ""
    controls = parse_control_context(text)
    if not controls:
        return text.strip()
    parts: list[str] = []
    for key, value in controls.items():
        if key == "negative prompt":
            continue
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def negative_control_text(text: str | None) -> str:
    controls = parse_control_context(text)
    return controls.get("negative prompt", "")


def target_bpm_from_text(text: str | None) -> int | None:
    controls = parse_control_context(text)
    for key in ("tempo", "bpm"):
        value = controls.get(key, "")
        match = re.search(r"\b(\d{2,3})\b", value)
        if match:
            return _clamp_bpm(int(match.group(1)))
    match = re.search(r"\b(\d{2,3})\s*bpm\b", text or "", flags=re.IGNORECASE)
    if match:
        return _clamp_bpm(int(match.group(1)))
    return None


def _clamp_bpm(value: int) -> int:
    return max(45, min(180, value))
