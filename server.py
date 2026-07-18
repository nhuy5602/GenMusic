from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from src.integrations.kaggle_auto import DEFAULT_MODEL, KaggleJobConfig, refresh_kaggle_job, submit_text_to_music_job
from src.evaluation.project_metrics import build_project_report


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUBMISSION_LOCK = threading.Lock()


class GenMusicHandler(BaseHTTPRequestHandler):
    server_version = "GenMusicVN/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_file(WEB_ROOT / "index.html")
            return
        if path.startswith("/web/"):
            requested = (WEB_ROOT / unquote(path.removeprefix("/web/"))).resolve()
            if not _is_relative_to(requested, WEB_ROOT.resolve()):
                self._send_json({"error": "Duong dan web khong hop le."}, HTTPStatus.BAD_REQUEST)
                return
            self._send_file(requested)
            return
        if path.startswith("/outputs/"):
            requested = (OUTPUT_ROOT / unquote(path.removeprefix("/outputs/"))).resolve()
            if not _is_relative_to(requested, OUTPUT_ROOT.resolve()):
                self._send_json({"error": "Đường dẫn output không hợp lệ."}, HTTPStatus.BAD_REQUEST)
                return
            self._send_file(requested)
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "backend": "genmusic-vn-self-diffusion", "model": DEFAULT_MODEL, "generator": "conditional-diffusion"})
            return
        if path == "/api/kaggle/status":
            query = parse_qs(parsed.query)
            run_id = (query.get("run_id") or [""])[0]
            state_path = (OUTPUT_ROOT / run_id / "kaggle_job" / "job_state.json").resolve()
            if not _is_relative_to(state_path, OUTPUT_ROOT.resolve()):
                self._send_json({"error": "Ma job Kaggle khong hop le."}, HTTPStatus.BAD_REQUEST)
                return
            if not run_id or not state_path.exists():
                self._send_json({"error": "Không tìm thấy job Kaggle."}, HTTPStatus.NOT_FOUND)
                return
            try:
                self._send_json(refresh_kaggle_job(state_path))
            except Exception as exc:  # pragma: no cover - server boundary
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/project/report":
            try:
                self._send_json(build_project_report(OUTPUT_ROOT, output_root=OUTPUT_ROOT / "project_report"))
            except Exception as exc:  # pragma: no cover - server boundary
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"error": "Không tìm thấy tài nguyên."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/generate":
            self._send_json({"error": "Không tìm thấy tài nguyên."}, HTTPStatus.NOT_FOUND)
            return

        if not SUBMISSION_LOCK.acquire(blocking=False):
            self._send_json(
                {"error": "Một yêu cầu tạo nhạc khác đang được gửi. Vui lòng chờ hoàn tất."},
                HTTPStatus.CONFLICT,
            )
            return

        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            job = submit_text_to_music_job(
                text=payload.get("text", ""),
                output_root=OUTPUT_ROOT,
                duration_seconds=int(payload.get("duration_seconds", 30)),
                genre=payload.get("genre") or None,
                config=KaggleJobConfig(
                    model=payload.get("model") or DEFAULT_MODEL,
                    submit=True,
                    wait=False,
                    checkpoint_kernel_ref=payload.get("checkpoint_ref") or None,
                    backing_kernel_ref=payload.get("backing_ref") or None,
                ),
            )
            self._send_json(job)
        except Exception as exc:  # pragma: no cover - server boundary
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        finally:
            SUBMISSION_LOCK.release()

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "Không tìm thấy file."}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Chạy web app MP3 local của GenMusic VN.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), GenMusicHandler)
    print(f"GenMusic VN đang chạy tại http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Đã dừng server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
