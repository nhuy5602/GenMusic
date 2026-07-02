from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate audio from a GenMusic VN prompt pack using MusicGen on Kaggle GPU.")
    parser.add_argument("--prompt-pack", required=True)
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--out", default="/kaggle/working/genmusic_vn")
    args = parser.parse_args()

    from audiocraft.data.audio import audio_write
    from audiocraft.models import MusicGen

    prompt_pack = json.loads(Path(args.prompt_pack).read_text(encoding="utf-8"))
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MusicGen.get_pretrained(args.model)
    model.set_generation_params(duration=int(prompt_pack.get("duration_seconds", 30)))
    wav = model.generate([prompt_pack["prompt"]])

    stem = output_dir / f"{prompt_pack.get('run_id', 'genmusic_vn')}_musicgen"
    audio_write(str(stem), wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
    (output_dir / "used_prompt_pack.json").write_text(json.dumps(prompt_pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(stem.with_suffix(".wav"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

