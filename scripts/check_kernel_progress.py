"""Pull the live tail of a Kaggle kernel's log while it's still RUNNING.

`kaggle kernels output` only returns files once a kernel session has finished
(returns nothing for a running kernel). This hits the same SSE log-stream
endpoint `KaggleApi.kernels_logs_stream` uses, but with an explicit read
timeout -- the library method's own `requests.get(..., stream=True)` has no
timeout, so it blocks forever waiting for the next byte if the kernel's
script hasn't printed a new line in a while (which is normal: real epochs
here are minutes apart). A bounded read timeout is what makes this usable as
a periodic "is it actually still healthy" check instead of hanging the caller.
"""
import json
import os
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleEnv
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.kaggle_auto import load_kaggle_api_tokens


def kernel_session_status(kernel_ref: str) -> str:
    """Read the lightweight KGAT session status without opening the log SSE stream."""
    os.environ.update(load_kaggle_api_tokens())
    owner_slug, kernel_slug = kernel_ref.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        request = ApiGetKernelSessionStatusRequest()
        request.user_name = owner_slug
        request.kernel_slug = kernel_slug
        response = kaggle.kernels.kernels_api_client.get_kernel_session_status(request)
    status = str(response.status or "UNKNOWN").upper()
    return status.rsplit(".", 1)[-1]


def tail_kernel_log(kernel_ref: str, read_timeout: float = 8.0) -> str:
    # Keep credentials in memory and make the helper work with the project's
    # .env without requiring callers to interpolate a secret in the shell.
    os.environ.update(load_kaggle_api_tokens())
    owner_slug, kernel_slug = kernel_ref.split("/", 1)
    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as kaggle:
        http = kaggle._http_client
        http._init_session()
        base = http._endpoint if http._env == KaggleEnv.PROD else f"{http._endpoint}/api"
        url = f"{base}/v1/kernels/logs/stream/{owner_slug}/{kernel_slug}"
        headers = dict(http._session.headers)
        headers["Accept"] = "text/event-stream, */*"
        headers.pop("Content-Type", None)

        try:
            response = http._session.get(
                url, stream=True, headers=headers, auth=http._session.auth,
                timeout=(5.0, read_timeout),
            )
        except Exception as exc:
            return (
                "[stream connection stopped: "
                f"{type(exc).__name__}: {exc}]"
            )
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        chunks = []
        try:
            if content_type.startswith("text/event-stream"):
                for line in response.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload == "END_OF_LOG":
                            break
                        try:
                            event = json.loads(payload)
                            chunks.append(event.get("data", ""))
                        except json.JSONDecodeError:
                            chunks.append(payload)
            else:
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        chunks.append(chunk)
        except Exception as e:  # includes requests.exceptions.ReadTimeout
            chunks.append(f"\n[stream read stopped: {type(e).__name__}: {e}]")
        finally:
            response.close()
        return "".join(chunks)


if __name__ == "__main__":
    # PowerShell commonly exposes a legacy cp1252 stdout even though Kaggle
    # logs are UTF-8.  Keep monitoring bounded and readable without requiring
    # every caller to set PYTHONIOENCODING explicitly.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    kernel_ref = sys.argv[1]
    status = kernel_session_status(kernel_ref)
    print(f"STATUS: {status}")
    if "--status-only" in sys.argv[2:]:
        raise SystemExit(0)
    read_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    text = tail_kernel_log(kernel_ref, read_timeout=read_timeout)
    print(f"--- {len(text)} chars ---")
    print(text[-4000:])
