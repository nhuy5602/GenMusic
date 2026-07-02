from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate audio from a GenMusic VN prompt pack using Stable Audio CLI.")
    parser.add_argument("--prompt-pack", required=True)
    parser.add_argument("--model", default="small-music")
    parser.add_argument("--out", default="/kaggle/working/genmusic_vn")
    args = parser.parse_args()

    executable = shutil.which("stable-audio")
    if executable is None:
        raise SystemExit("stable-audio CLI not found. Install Stable Audio tooling in the Kaggle notebook first.")

    prompt_pack = json.loads(Path(args.prompt_pack).read_text(encoding="utf-8"))
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prompt_pack.get('run_id', 'genmusic_vn')}_stable_audio.wav"

    subprocess.run(
        [
            executable,
            "--model",
            args.model,
            "-p",
            prompt_pack["prompt"],
            "--duration",
            str(int(prompt_pack.get("duration_seconds", 30))),
            "-o",
            str(output_path),
        ],
        check=True,
    )
    (output_dir / "used_prompt_pack.json").write_text(json.dumps(prompt_pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

