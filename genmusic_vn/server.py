from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .kaggle_auto import (
    DEFAULT_CUSTOM_MUSIC_MODEL,
    KaggleJobConfig,
    refresh_kaggle_job,
    submit_text_to_music_job,
    submit_tts_retry_job,
)
from .project_metrics import build_project_report
from .trained_text_model import trained_model_status


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
            self._send_file(WEB_ROOT / unquote(path.removeprefix("/web/")))
            return
        if path.startswith("/outputs/"):
            requested = (OUTPUT_ROOT / unquote(path.removeprefix("/outputs/"))).resolve()
            if not _is_relative_to(requested, OUTPUT_ROOT.resolve()):
                self._send_json({"error": "Invalid output path."}, HTTPStatus.BAD_REQUEST)
                return
            self._send_file(requested)
            return
        if path == "/api/health":
            self._send_json({"status": "ok", "backend": "trained-text-model+custom-composer+tts", "text_model": trained_model_status()})
            return
        if path == "/api/kaggle/status":
            query = parse_qs(parsed.query)
            run_id = (query.get("run_id") or [""])[0]
            state_path = OUTPUT_ROOT / run_id / "kaggle_job" / "job_state.json"
            if not run_id or not state_path.exists():
                self._send_json({"error": "Kaggle job not found."}, HTTPStatus.NOT_FOUND)
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
        self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/kaggle/retry-tts":
            self._handle_tts_retry()
            return
        if parsed.path != "/api/generate":
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        if not SUBMISSION_LOCK.acquire(blocking=False):
            self._send_json(
                {"error": "Another generation request is already being submitted. Please wait for it to finish."},
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
                    model=payload.get("model") or DEFAULT_CUSTOM_MUSIC_MODEL,
                    submit=True,
                    wait=False,
                ),
            )
            self._send_json(job)
        except Exception as exc:  # pragma: no cover - server boundary
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        finally:
            SUBMISSION_LOCK.release()

    def _handle_tts_retry(self) -> None:
        if not SUBMISSION_LOCK.acquire(blocking=False):
            self._send_json(
                {"error": "Another generation request is already being submitted. Please wait for it to finish."},
                HTTPStatus.CONFLICT,
            )
            return

        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            run_id = str(payload.get("run_id", "")).strip()
            state_path = OUTPUT_ROOT / run_id / "kaggle_job" / "job_state.json"
            if not run_id or not state_path.exists():
                self._send_json({"error": "Kaggle job not found."}, HTTPStatus.NOT_FOUND)
                return
            job = submit_tts_retry_job(
                state_path,
                config=KaggleJobConfig(
                    model=payload.get("model") or DEFAULT_CUSTOM_MUSIC_MODEL,
                    submit=True,
                    wait=False,
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
            self._send_json({"error": "File not found."}, HTTPStatus.NOT_FOUND)
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
    parser = argparse.ArgumentParser(description="Run the GenMusic VN local MP3 client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), GenMusicHandler)
    print(f"GenMusic VN running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
