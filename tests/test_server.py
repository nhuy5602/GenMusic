from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import server as server_module


class WebApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = ThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.GenMusicHandler
        )
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict | bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        if "application/json" in content_type:
            return response.status, json.loads(raw.decode("utf-8"))
        return response.status, raw

    def _post_json(self, payload: object) -> tuple[int, dict | bytes]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request(
            "POST",
            "/api/generate",
            body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(body)),
            },
        )

    def test_health_and_path_validation(self) -> None:
        status, payload = self._request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

        for path in (
            "/web/%2e%2e/server.py",
            "/outputs/%2e%2e/server.py",
        ):
            with self.subTest(path=path):
                status, payload = self._request("GET", path)
                self.assertEqual(status, 400)
                self.assertIn("error", payload)

        status, payload = self._request("GET", "/api/kaggle/status")
        self.assertEqual(status, 404)
        self.assertIn("error", payload)

    def test_valid_unicode_and_boundary_inputs_do_not_raise(self) -> None:
        cases = [
            {
                "text": "Mưa rơi trên phố nhỏ 🌧️\nEm vẫn chờ bình minh.",
                "duration_seconds": 4,
                "genre": "Pop Việt, piano ấm áp",
            },
            {
                "text": "Một hai ba, ta cùng hát!",
                "duration_seconds": 120,
                "genre": "",
            },
            {
                "text": "Một ngày mới đang lên.",
                "genre": "Acoustic",
            },
        ]
        for index, payload in enumerate(cases):
            expected = {
                "status": "staged",
                "run_id": f"mock-{index}",
                "duration_seconds": payload.get("duration_seconds", 30),
            }
            with self.subTest(payload=payload), patch.object(
                server_module,
                "submit_text_to_music_job",
                return_value=expected,
            ) as submit:
                status, response = self._post_json(payload)
                self.assertEqual(status, 200)
                self.assertEqual(response["run_id"], f"mock-{index}")
                submit.assert_called_once()

    def test_invalid_inputs_return_400_without_submitting(self) -> None:
        invalid_payloads = [
            [],
            {},
            {"text": "   ", "duration_seconds": 12},
            {"text": 123, "duration_seconds": 12},
            {"text": "Hợp lệ", "duration_seconds": "12"},
            {"text": "Hợp lệ", "duration_seconds": 4.5},
            {"text": "Hợp lệ", "duration_seconds": 3},
            {"text": "Hợp lệ", "duration_seconds": 121},
            {"text": "Hợp lệ", "duration_seconds": True},
            {"text": "Hợp lệ", "duration_seconds": 12, "genre": 42},
            {"text": "Hợp lệ", "duration_seconds": 12, "genre": "x" * 513},
            {"text": "hát " * 700, "duration_seconds": 120},
        ]
        with patch.object(server_module, "submit_text_to_music_job") as submit:
            for payload in invalid_payloads:
                with self.subTest(payload_type=type(payload).__name__):
                    status, response = self._post_json(payload)
                    self.assertEqual(status, 400)
                    self.assertIn("error", response)
            submit.assert_not_called()

    def test_malformed_empty_and_oversized_json_return_400(self) -> None:
        bodies = [b"{not-json", b"", b"x" * (server_module.MAX_REQUEST_BYTES + 1)]
        with patch.object(server_module, "submit_text_to_music_job") as submit:
            for body in bodies:
                with self.subTest(length=len(body)):
                    status, response = self._request(
                        "POST",
                        "/api/generate",
                        body,
                        headers={
                            "Content-Type": "application/json",
                            "Content-Length": str(len(body)),
                        },
                    )
                    self.assertEqual(status, 400)
                    self.assertIn("error", response)
            submit.assert_not_called()

    def test_concurrent_submission_returns_409(self) -> None:
        body = json.dumps(
            {"text": "Một lời ca", "duration_seconds": 12}
        ).encode("utf-8")
        server_module.SUBMISSION_LOCK.acquire()
        try:
            with patch.object(server_module, "submit_text_to_music_job") as submit:
                status, response = self._request(
                    "POST",
                    "/api/generate",
                    body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                    },
                )
                self.assertEqual(status, 409)
                self.assertIn("error", response)
                submit.assert_not_called()
        finally:
            server_module.SUBMISSION_LOCK.release()


if __name__ == "__main__":
    unittest.main()
