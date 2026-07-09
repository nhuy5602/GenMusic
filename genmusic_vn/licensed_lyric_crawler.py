from __future__ import annotations

import hashlib
import json
import re
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .rhyme import assonance_key_word, end_pair_assonance_rate, end_pair_rhyme_rate, end_rhyme_key
from .training_dataset import GENRE_SCENES
from .text_utils import tokenize_words


MAX_SECTION_CHARS = 2_400
MAX_SECTION_LINES = 24
MAX_RESPONSE_BYTES = 2_000_000
ALLOWED_LICENSE_TERMS = (
    "public domain",
    "public-domain",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
    "creative commons attribution",
    "creative commons attribution-sharealike",
    "user-owned",
    "user_owned",
    "permission granted",
)


class LicensedCrawlError(RuntimeError):
    pass


def load_source_specs(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    text = source_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [dict(item) for item in data if isinstance(item, dict)]
    specs: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                specs.append(item)
    return specs


def crawl_licensed_sources(
    source_specs: list[dict[str, Any]],
    output_path: str | Path,
    *,
    max_sources: int | None = None,
    max_sections_per_source: int = 12,
    max_snippets_per_source: int | None = None,
    user_agent: str = "GenMusicVN-LicensedLyricResearch/1.0 (+local evaluation)",
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    selected_specs = source_specs[: max_sources or len(source_specs)]
    if max_snippets_per_source is not None:
        max_sections_per_source = max_snippets_per_source
    for spec in selected_specs:
        result = _crawl_one_source(
            spec,
            max_sections=max(1, min(12, int(max_sections_per_source))),
            user_agent=user_agent,
        )
        source_results.append(result["summary"])
        records.extend(result["records"])

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "status": "complete",
        "source_count": len(selected_specs),
        "section_count": len(records),
        "snippet_count": len(records),
        "output_path": str(output),
        "source_results": source_results,
        "policy": {
            "robots_txt_required": True,
            "license_required": True,
            "max_section_chars": MAX_SECTION_CHARS,
            "max_section_lines": MAX_SECTION_LINES,
            "allowed_section_types": ["verse", "chorus", "bridge", "pre_chorus", "outro"],
            "full_song_reconstruction": False,
        },
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_rhyme_profile(dataset_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Learn a compact, provenance-preserving rhyme profile from crawled snippets."""
    records: list[dict[str, Any]] = []
    path = Path(dataset_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("rhyme_features"):
                records.append(item)
    ending_counts: dict[str, int] = {}
    assonance_counts: dict[str, int] = {}
    for record in records:
        features = record.get("rhyme_features") or {}
        for key in features.get("end_rhyme_keys", []):
            ending_counts[str(key)] = ending_counts.get(str(key), 0) + 1
        for key in features.get("assonance_keys", []):
            assonance_counts[str(key)] = assonance_counts.get(str(key), 0) + 1
    profile = {
        "status": "complete",
        "source_dataset": str(path),
        "record_count": len(records),
        "top_exact_endings": _top_counts(ending_counts),
        "top_assonance_families": _top_counts(assonance_counts),
        "training_note": "Use as a style prior; never force a learned ending when the lyric meaning becomes unnatural.",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    profile["output_path"] = str(target)
    return profile


def extract_lyric_sections(
    html: str,
    *,
    max_sections: int = 20,
    max_section_chars: int = MAX_SECTION_CHARS,
) -> list[dict[str, str]]:
    parser = _SectionParser()
    parser.feed(html)
    parser.close()
    parser._flush()
    sections: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed_types = {"verse", "chorus", "bridge", "pre_chorus", "outro", "unknown"}
    for section in parser.sections:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in section["text"].split("\n")]
        lines = [line for line in lines if line]
        if not lines:
            continue
        section_type = section["section_type"]
        if section_type not in allowed_types:
            continue
        clipped = lines[:MAX_SECTION_LINES]
        text = "\n".join(clipped)[: min(MAX_SECTION_CHARS, max(200, int(max_section_chars)))].strip()
        if len(tokenize_words(text)) < 6:
            continue
        key = _normalize_dedupe(text)
        if key and key not in seen:
            seen.add(key)
            sections.append({"section_type": section_type, "text": text})
        if len(sections) >= max_sections:
            break
    return sections


def extract_lyric_snippets(html: str, *, max_snippets: int = 20) -> list[str]:
    """Backward-compatible alias; each returned item is now a whole section."""
    return [
        section["text"]
        for section in extract_lyric_sections(html, max_sections=max_snippets)
    ]


def is_allowed_license(license_text: Any) -> bool:
    normalized = str(license_text or "").strip().lower()
    return bool(normalized) and any(term in normalized for term in ALLOWED_LICENSE_TERMS)


def _crawl_one_source(
    spec: dict[str, Any], *, max_sections: int, user_agent: str
) -> dict[str, Any]:
    url = str(spec.get("url") or "").strip()
    license_text = str(spec.get("license") or "").strip()
    summary: dict[str, Any] = {"url": url, "license": license_text, "snippet_count": 0}
    if not spec.get("approved", False):
        summary["status"] = "skipped"
        summary["reason"] = "Source is not explicitly approved in the manifest."
        return {"summary": summary, "records": []}
    if not is_allowed_license(license_text):
        summary["status"] = "skipped"
        summary["reason"] = "No accepted public/open/user-owned license marker was provided."
        return {"summary": summary, "records": []}
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        summary["status"] = "skipped"
        summary["reason"] = "Only HTTP(S) sources are allowed."
        return {"summary": summary, "records": []}
    try:
        if not _robots_allowed(url, user_agent):
            summary["status"] = "skipped"
            summary["reason"] = "robots.txt disallows this user agent."
            return {"summary": summary, "records": []}
        html = _fetch_html(url, user_agent)
        sections = extract_lyric_sections(
            html,
            max_sections=max_sections,
            max_section_chars=min(
                MAX_SECTION_CHARS,
                int(spec.get("max_section_chars") or MAX_SECTION_CHARS),
            ),
        )
    except Exception as exc:  # network/source failures are recorded per source
        summary["status"] = "error"
        summary["reason"] = f"{type(exc).__name__}: {exc}"
        return {"summary": summary, "records": []}

    genre_label = str(spec.get("genre_label") or "pop_ballad")
    if genre_label not in GENRE_SCENES:
        genre_label = "pop_ballad"
    emotion = str(spec.get("emotion") or "nostalgic")
    records = [
        _snippet_record(
            section["text"],
            url=url,
            license_text=license_text,
            source_title=str(spec.get("title") or "licensed lyric source"),
            emotion=emotion,
            genre_label=genre_label,
            index=index,
            section_type=section["section_type"],
        )
        for index, section in enumerate(sections, start=1)
    ]
    summary["status"] = "complete"
    summary["section_count"] = len(records)
    summary["snippet_count"] = len(records)
    return {"summary": summary, "records": records}


def _snippet_record(
    snippet: str,
    *,
    url: str,
    license_text: str,
    source_title: str,
    emotion: str,
    genre_label: str,
    index: int,
    section_type: str,
) -> dict[str, Any]:
    lines = [line for line in snippet.splitlines() if line.strip()]
    end_keys = [end_rhyme_key(line) for line in lines]
    assonance_keys = [assonance_key_word(line.split()[-1]) for line in lines if line.split()]
    return {
        "id": f"licensed_snippet_{hashlib.sha1(f'{url}|{index}|{snippet}'.encode('utf-8')).hexdigest()[:12]}",
        "input_text": snippet,
        "emotion": emotion,
        "genre_label": genre_label,
        "style_prompt": GENRE_SCENES[genre_label]["style_prompt"],
        "source": "licensed_public_lyric_section",
        "source_title": source_title,
        "source_url": url,
        "license": license_text,
        "section_only": True,
        "section_type": section_type,
        "rhyme_features": {
            "line_count": len(lines),
            "end_rhyme_keys": end_keys,
            "assonance_keys": assonance_keys,
            "exact_pair_rate": round(end_pair_rhyme_rate(lines), 4),
            "assonance_pair_rate": round(end_pair_assonance_rate(lines), 4),
        },
    }


def _robots_allowed(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    parser.read()
    return parser.can_fetch(user_agent, url)


def _fetch_html(url: str, user_agent: str) -> str:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "text/html"})
    with urlopen(request, timeout=20) as response:
        content = response.read(MAX_RESPONSE_BYTES + 1)
    if len(content) > MAX_RESPONSE_BYTES:
        raise LicensedCrawlError("Source response exceeds the 2 MB safety limit.")
    return content.decode("utf-8", errors="replace")


class _SectionParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[dict[str, str]] = []
        self._current: list[str] = []
        self._section_type = "unknown"
        self._in_heading = False
        self._heading: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._flush()
            self._heading = []
            self._in_heading = True
        elif tag == "br":
            self._current.append("\n")
        elif tag in {"p", "div", "li"}:
            if self._section_type == "unknown" and self._current:
                self._flush()
            elif self._current and not self._current[-1].endswith("\n"):
                self._current.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self._in_heading:
            title = " ".join("".join(self._heading).split())
            self._section_type = _classify_section_title(title)
            self._in_heading = False
        elif tag in {"p", "div", "li"}:
            if self._current and not self._current[-1].endswith("\n"):
                self._current.append("\n")
            if self._section_type == "unknown":
                self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading.append(data)
        else:
            self._current.append(data)

    def _flush(self) -> None:
        text = "".join(self._current).strip()
        if text:
            self.sections.append({"section_type": self._section_type, "text": text})
        self._current = []


def _classify_section_title(title: str) -> str:
    normalized = title.strip().lower()
    if any(token in normalized for token in ("pre-chorus", "pre chorus", "tiền điệp khúc")):
        return "pre_chorus"
    if any(token in normalized for token in ("chorus", "điệp khúc", "diep khuc", "refrain")):
        return "chorus"
    if any(token in normalized for token in ("verse", "khổ", "kho ", "đoạn 1", "đoạn 2")):
        return "verse"
    if any(token in normalized for token in ("bridge", "chuyển", "cầu")):
        return "bridge"
    if any(token in normalized for token in ("outro", "kết", "ending")):
        return "outro"
    return "unknown"


def _normalize_dedupe(text: str) -> str:
    return " ".join(text.lower().split())


def _top_counts(values: dict[str, int], limit: int = 24) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": count}
        for key, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]
