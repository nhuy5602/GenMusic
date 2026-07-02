from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate audio from a GenMusic VN prompt pack using Stable Audio Open on Kaggle GPU.")
    parser.add_argument("--prompt-pack", required=True)
    parser.add_argument("--model", default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--out", default="/kaggle/working/genmusic_vn")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from diffusers import StableAudioPipeline

    prompt_pack = json.loads(Path(args.prompt_pack).read_text(encoding="utf-8"))
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableAudioPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(42) if device == "cuda" else torch.Generator().manual_seed(42)
    audio = pipe(
        prompt_pack["prompt"],
        negative_prompt=prompt_pack.get("negative_prompt", "low quality"),
        num_inference_steps=args.steps,
        audio_end_in_s=float(prompt_pack.get("duration_seconds", 30)),
        num_waveforms_per_prompt=1,
        generator=generator,
    ).audios

    output = audio[0].T.float().cpu().numpy()
    output_path = output_dir / f"{prompt_pack.get('run_id', 'genmusic_vn')}_stable_audio_open.wav"
    sf.write(output_path, output, pipe.vae.sampling_rate)
    (output_dir / "used_prompt_pack.json").write_text(json.dumps(prompt_pack, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

