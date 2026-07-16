import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.integrations.kaggle_auto import (
    kaggle_access_token,
    kaggle_auth_available,
    load_kaggle_api_tokens,
)


class KaggleAuthTests(unittest.TestCase):
    def _load(self, project_root: Path, config_dir: Path, **environment: str) -> dict[str, str]:
        process_environment = {"KAGGLE_CONFIG_DIR": str(config_dir), **environment}
        with (
            patch("src.integrations.kaggle_auto.PROJECT_ROOT", project_root),
            patch.dict(os.environ, process_environment, clear=True),
        ):
            return load_kaggle_api_tokens()

    def test_project_dotenv_access_token_overrides_kaggle_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (config_dir / "access_token").write_text("KGAT-file-token", encoding="utf-8")
            (config_dir / "kaggle.json").write_text(
                json.dumps({"username": "file-user", "key": "legacy-file-key"}),
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "KAGGLE_USERNAME=dotenv-user\nKAGGLE_API_TOKEN=KGAT-dotenv-token\n",
                encoding="utf-8",
            )

            values = self._load(root, config_dir)

            self.assertEqual(values["KAGGLE_USERNAME"], "dotenv-user")
            self.assertEqual(values["KAGGLE_API_TOKEN"], "KGAT-dotenv-token")
            self.assertNotIn("KAGGLE_KEY", values)

    def test_kaggle_files_are_used_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (config_dir / "access_token").write_text("KGAT-file-token", encoding="utf-8")
            (config_dir / "kaggle.json").write_text(
                json.dumps({"username": "file-user", "key": "legacy-file-key"}),
                encoding="utf-8",
            )

            values = self._load(root, config_dir)

            self.assertEqual(values["KAGGLE_USERNAME"], "file-user")
            self.assertEqual(kaggle_access_token(values), "KGAT-file-token")
            self.assertTrue(kaggle_auth_available(values))

    def test_process_environment_overrides_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (root / ".env").write_text(
                "KAGGLE_USERNAME=dotenv-user\nKAGGLE_API_TOKEN=KGAT-dotenv-token\n",
                encoding="utf-8",
            )

            values = self._load(
                root,
                config_dir,
                KAGGLE_USERNAME="process-user",
                KAGGLE_API_TOKEN="KGAT-process-token",
            )

            self.assertEqual(values["KAGGLE_USERNAME"], "process-user")
            self.assertEqual(kaggle_access_token(values), "KGAT-process-token")

    def test_env_local_auth_overrides_env_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (root / ".env").write_text(
                "KAGGLE_USERNAME=dotenv-user\nKAGGLE_API_TOKEN=KGAT-dotenv-token\n",
                encoding="utf-8",
            )
            (root / ".env.local").write_text("KAGGLE_KEY=local-legacy-key\n", encoding="utf-8")

            values = self._load(root, config_dir)

            self.assertEqual(values["KAGGLE_USERNAME"], "dotenv-user")
            self.assertEqual(values["KAGGLE_KEY"], "local-legacy-key")
            self.assertNotIn("KAGGLE_API_TOKEN", values)
            self.assertTrue(kaggle_auth_available(values))

    def test_access_token_alias_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (root / ".env").write_text(
                "KAGGLE_USERNAME=dotenv-user\nKAGGLE_ACCESS_TOKEN=KGAT-alias-token\n",
                encoding="utf-8",
            )

            values = self._load(root, config_dir)

            self.assertEqual(values["KAGGLE_API_TOKEN"], "KGAT-alias-token")
            self.assertTrue(kaggle_auth_available(values))

    def test_current_underscore_access_token_prefix_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (root / ".env").write_text(
                "KAGGLE_USERNAME=dotenv-user\nKAGGLE_API_TOKEN=KGAT_current-token\n",
                encoding="utf-8",
            )

            values = self._load(root, config_dir)

            self.assertEqual(kaggle_access_token(values), "KGAT_current-token")
            self.assertTrue(kaggle_auth_available(values))

    def test_kgat_token_pasted_into_legacy_env_name_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (root / ".env").write_text(
                "KAGGLE_USERNAME=dotenv-user\nKAGGLE_KEY=KGAT-pasted-token\n",
                encoding="utf-8",
            )

            values = self._load(root, config_dir)

            self.assertEqual(values["KAGGLE_API_TOKEN"], "KGAT-pasted-token")
            self.assertNotIn("KAGGLE_KEY", values)
            self.assertTrue(kaggle_auth_available(values))

    def test_non_kgat_access_token_file_is_not_modern_auth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "kaggle-home"
            config_dir.mkdir()
            (config_dir / "access_token").write_text("legacy-key-in-wrong-file", encoding="utf-8")

            values = self._load(root, config_dir)

            self.assertIsNone(kaggle_access_token(values))
            self.assertFalse(kaggle_auth_available(values))


if __name__ == "__main__":
    unittest.main()
