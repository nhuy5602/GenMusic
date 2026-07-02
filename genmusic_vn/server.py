from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .kaggle_auto import KaggleJobConfig, refresh_kaggle_job, run_or_stage_kaggle_job
from .pipeline import create_music_project
from .schemas import to_plain_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


class GenMusicHandler(BaseHTTPRequestHandler):
    server_version = "GenMusicVN/0.1"

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
            self._send_json({"status": "ok"})
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
        self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/generate":
            self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            requested_backend = payload.get("backend", "guide")
            local_backend = "guide" if requested_backend == "kaggle-auto" else requested_backend
            result = create_music_project(
                text=payload.get("text", ""),
                output_root=OUTPUT_ROOT,
                backend=local_backend,
                duration_seconds=int(payload.get("duration_seconds", 30)),
                genre=payload.get("genre") or None,
                render_audio=True,
            )
            data = to_plain_data(result)
            if requested_backend == "kaggle-auto":
                data["backend"] = "kaggle-auto"
                data["kaggle_job"] = run_or_stage_kaggle_job(
                    result,
                    OUTPUT_ROOT,
                    KaggleJobConfig(
                        kaggle_backend=payload.get("kaggle_backend", "musicgen"),
                        model=payload.get("model") or "facebook/musicgen-small",
                        submit=True,
                        wait=False,
                    ),
                )
            for file_item in data["files"]:
                file_path = Path(file_item["path"]).resolve()
                try:
                    relative = file_path.relative_to(OUTPUT_ROOT.resolve())
                    file_item["url"] = "/outputs/" + relative.as_posix()
                except ValueError:
                    file_item["url"] = ""
            self._send_json(data)
        except Exception as exc:  # pragma: no cover - server boundary
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

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
    parser = argparse.ArgumentParser(description="Run the GenMusic VN local web UI.")
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
