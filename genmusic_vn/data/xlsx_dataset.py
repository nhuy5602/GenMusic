from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from ..core.text_utils import extract_keywords


INPUT_SHEET = "Input_Dataset"
NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

MOOD_TO_EMOTIONS = {
    "Buồn/Hoài niệm": ["sadness", "nostalgic"],
    "Vui tươi": ["joy"],
    "Cô đơn": ["sadness", "nostalgic"],
    "Tự do": ["joy", "hope"],
    "Bình yên": ["calm"],
    "Hào hùng": ["hope", "joy"],
    "Căng thẳng": ["fear"],
    "Huyền bí": ["fear"],
    "Hy vọng": ["hope"],
    "Kịch tính": ["fear"],
    "Giận dữ": ["anger"],
    "Chill": ["calm", "nostalgic"],
    "Trưởng thành": ["hope", "nostalgic"],
    "Ấm áp": ["calm", "romantic"],
    "Kỳ vọng": ["hope"],
    "U tối": ["fear", "sadness"],
    "Hân hoan": ["joy", "hope"],
    "Mất mát": ["sadness"],
    "Hoài niệm vui": ["nostalgic", "joy"],
    "Chiến thắng": ["hope", "joy"],
    "Phiêu lưu": ["hope", "joy"],
    "Thiền định": ["calm"],
    "Quyết tâm": ["hope", "anger"],
    "Lãng mạn": ["romantic"],
    "Sống sót": ["fear", "hope"],
    "Chữa lành": ["calm", "hope"],
    "Tập trung": ["calm", "hope"],
    "Lễ hội": ["joy"],
    "Bí ẩn": ["fear"],
    "Xúc động": ["romantic", "hope", "calm"],
    "Hiện đại": ["hope", "joy", "calm"],
}


def load_xlsx_input_rows(path: str | Path, *, sheet_name: str = INPUT_SHEET) -> list[dict[str, Any]]:
    workbook_path = Path(path)
    with zipfile.ZipFile(workbook_path) as archive:
        shared_strings = _read_shared_strings(archive)
        worksheet_path = _worksheet_path_for_sheet(archive, sheet_name)
        rows = _read_sheet_rows(archive, worksheet_path, shared_strings)
    if not rows:
        return []
    headers = [str(item).strip() for item in rows[0]]
    records: list[dict[str, Any]] = []
    for raw_row in rows[1:]:
        record = {
            headers[index]: raw_row[index] if index < len(raw_row) else ""
            for index in range(len(headers))
            if headers[index]
        }
        if str(record.get("ID", "")).strip() and str(record.get("Input tiếng Việt", "")).strip():
            records.append(record)
    return records


def records_from_xlsx(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in load_xlsx_input_rows(path):
        input_text = str(row.get("Input tiếng Việt", "")).strip()
        mood = str(row.get("Mood chính", "")).strip()
        prompt_hint = str(row.get("Prompt nhạc gợi ý cho MusicGen", "")).strip()
        genre = prompt_hint or str(row.get("Genre/Style đề xuất", "")).strip()
        duration = _as_int(row.get("Duration target (s)"), 30)
        expected_keywords = extract_keywords(
            " ".join(
                [
                    input_text,
                    str(row.get("Expected output", "")),
                    str(row.get("Tiêu chí đánh giá chính", "")),
                ]
            ),
            limit=10,
        )
        records.append(
            {
                "id": str(row.get("ID", "")).strip(),
                "input_text": input_text,
                "duration_seconds": duration,
                "genre": genre,
                "expected_emotions": MOOD_TO_EMOTIONS.get(mood, []),
                "expected_mood_text": mood,
                "expected_secondary_mood": str(row.get("Mood phụ", "")).strip(),
                "expected_keywords": expected_keywords,
                "expected_output": str(row.get("Expected output", "")).strip(),
                "rubric": str(row.get("Tiêu chí đánh giá chính", "")).strip(),
                "length_bucket": str(row.get("Độ dài text", "")).strip() or "unknown",
                "source": str(Path(path)),
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


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall("a:si", NS):
        strings.append("".join(text.text or "" for text in item.findall(".//a:t", NS)))
    return strings


def _worksheet_path_for_sheet(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("rel:Relationship", NS)
    }
    for sheet in workbook.findall(".//a:sheet", NS):
        if sheet.attrib.get("name") == sheet_name:
            rel_id = sheet.attrib.get(f"{{{NS['r']}}}id", "")
            target = rel_map.get(rel_id)
            if not target:
                break
            target = target.lstrip("/")
            if target.startswith("xl/"):
                return target
            return "xl/" + target
    raise ValueError(f"Không tìm thấy sheet '{sheet_name}' trong workbook.")


def _read_sheet_rows(archive: zipfile.ZipFile, worksheet_path: str, shared_strings: list[str]) -> list[list[Any]]:
    root = ET.fromstring(archive.read(worksheet_path))
    rows: list[list[Any]] = []
    for row in root.findall(".//a:sheetData/a:row", NS):
        values: list[Any] = []
        for cell in row.findall("a:c", NS):
            col_index = _column_index(cell.attrib.get("r", ""))
            while len(values) < col_index:
                values.append("")
            values.append(_cell_value(cell, shared_strings))
        rows.append(values)
    return rows


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//a:t", NS))
    raw_value = cell.findtext("a:v", default="", namespaces=NS)
    if cell_type == "s":
        return shared_strings[int(raw_value)] if raw_value else ""
    if cell_type in {"str", "b"}:
        return raw_value
    if raw_value == "":
        return ""
    try:
        number = float(raw_value)
    except ValueError:
        return raw_value
    return int(number) if number.is_integer() else number


def _column_index(cell_ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
    index = 0
    for letter in letters:
        index = index * 26 + (ord(letter) - ord("A") + 1)
    return max(1, index)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
