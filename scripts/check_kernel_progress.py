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
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleEnv

from src.integrations.kaggle_auto import kaggle_access_token, load_kaggle_api_tokens


def tail_kernel_log(kernel_ref: str, read_timeout: float = 8.0) -> str:
    owner_slug, kernel_slug = kernel_ref.split("/", 1)
    access_token = kaggle_access_token(load_kaggle_api_tokens())
    if access_token:
        # KGAT access tokens authenticate against www.kaggle.com.  KaggleApi's
        # legacy api.kaggle.com session does not attach them to this SSE route
        # and returns 401 even though kernel status/output calls succeed.
        import requests

        session = requests.Session()
        url = (
            "https://www.kaggle.com/api/v1/kernels/logs/stream/"
            f"{owner_slug}/{kernel_slug}"
        )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "text/event-stream, */*",
        }
        try:
            response = session.get(
                url,
                stream=True,
                headers=headers,
                timeout=(5.0, read_timeout),
            )
        except requests.RequestException as exc:
            return f"[stream connection stopped: {type(exc).__name__}: {exc}]"
    else:
        api = KaggleApi()
        api.authenticate()
        kaggle = api.build_kaggle_client()
        http = kaggle._http_client
        http._init_session()
        base = http._endpoint if http._env == KaggleEnv.PROD else f"{http._endpoint}/api"
        url = f"{base}/v1/kernels/logs/stream/{owner_slug}/{kernel_slug}"
        headers = dict(http._session.headers)
        headers["Accept"] = "text/event-stream, */*"
        headers.pop("Content-Type", None)
        response = http._session.get(
            url, stream=True, headers=headers, auth=http._session.auth,
            timeout=(5.0, read_timeout),
        )

    try:
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()
        chunks = []
        # Kaggle sends SSE keep-alives frequently enough that requests' socket
        # read timeout may never fire, even when no new notebook log exists.
        # Bound the whole stream operation as well so a periodic monitor cannot
        # hang forever on a healthy but quiet training batch.
        deadline = time.monotonic() + max(1.0, float(read_timeout))
        try:
            if content_type.startswith("text/event-stream"):
                for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                    if time.monotonic() >= deadline:
                        chunks.append("\n[stream stopped: absolute deadline reached]")
                        break
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
                    if time.monotonic() >= deadline:
                        chunks.append("\n[stream stopped: absolute deadline reached]")
                        break
                    if chunk:
                        chunks.append(chunk)
        except Exception as e:  # includes requests.exceptions.ReadTimeout
            chunks.append(f"\n[stream read stopped: {type(e).__name__}: {e}]")
        finally:
            response.close()
        return "".join(chunks)
    finally:
        response.close()


if __name__ == "__main__":
    # Kaggle logs may contain Vietnamese text and progress symbols. Windows'
    # legacy console encoding otherwise raises UnicodeEncodeError while merely
    # trying to display a healthy job's log.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    kernel_ref = sys.argv[1]
    read_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    text = tail_kernel_log(kernel_ref, read_timeout=read_timeout)
    print(f"--- {len(text)} chars ---")
    print(text[-4000:])
