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

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleEnv


def tail_kernel_log(kernel_ref: str, read_timeout: float = 8.0) -> str:
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

        response = http._session.get(
            url, stream=True, headers=headers, auth=http._session.auth,
            timeout=(5.0, read_timeout),
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
    kernel_ref = sys.argv[1]
    read_timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    text = tail_kernel_log(kernel_ref, read_timeout=read_timeout)
    print(f"--- {len(text)} chars ---")
    print(text[-4000:])
