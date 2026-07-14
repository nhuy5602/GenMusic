"""Smoke-test inference for the distilled MicroDiT student.

Decodes with Vocos (charactr/vocos-mel-24khz), not the teacher's BigVGAN: the
student's mel is now natively in Vocos's own format (100 mels, 24kHz, n_fft=1024,
hop=256 -- see MusicDiffusionConfig), so Vocos decodes it directly with no
resampling. Feeding this mel through the teacher's BigVGAN decoder (the
previous version of this script) was decoding a foreign, differently-scaled
mel representation through a vocoder trained on the teacher's own latent space
-- effectively garbage audio. See docs/experiments/vocoder_fix.md.
"""
import sys
import torch
import torchaudio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.text_to_music_diffusion import MusicDiffusionConfig, compute_mel_spectrogram, render_mel_to_wav
from src.models.dit_transformer import MicroDiT
from src.data.preprocess_raw_vietnamese import compute_style_embedding


def run_student_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running distilled student inference on device: {device}")

    config = MusicDiffusionConfig(frames_per_chunk=128)  # 100 mel / 24kHz / n_fft=1024 / hop=256
    student = MicroDiT(config, dim=256, depth=4, heads=4, style_dim=512).to(device)

    student_ckpt_path = Path("/kaggle/working/distilled_student.pt")
    if not student_ckpt_path.exists():
        student_ckpt_path = PROJECT_ROOT / "outputs/distilled_student.pt"

    if student_ckpt_path.exists():
        print(f"Loading distilled student checkpoint from {student_ckpt_path}...")
        ckpt = torch.load(student_ckpt_path, map_location=device)
        student.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    else:
        print("[WARNING] No distilled student checkpoint found. Running inference with randomly initialized weights.")

    student.eval()

    lyrics_text = "Đêm nay mưa rơi rơi trên lối mòn xưa cũ. Lòng anh nhớ em nhiều người ơi có biết chăng."

    if Path("/kaggle/input").exists():
        songs_dir = Path("/kaggle/input")
        mp3_files = list(songs_dir.rglob("*.mp3")) + list(songs_dir.rglob("*.wav"))
        output_dir = Path("/kaggle/working")
    else:
        songs_dir = PROJECT_ROOT / "dataset" / "vietnamese_songs"
        mp3_files = list(songs_dir.glob("*.mp3")) + list(songs_dir.glob("*.wav"))
        output_dir = PROJECT_ROOT / "outputs" / "student_test"

    if not mp3_files:
        print("[ERROR] Please add at least one mp3/wav song to use as a style anchor.")
        return

    style_song = mp3_files[0]
    print(f"Using style anchor song: {style_song.name}")

    prompt_wav, sr = torchaudio.load(style_song)
    prompt_wav = torchaudio.functional.resample(prompt_wav.to(device), sr, config.sample_rate)
    prompt_wav = prompt_wav.mean(dim=0)  # mono

    target_samples = config.frames_per_chunk * config.hop_length
    if prompt_wav.shape[0] > target_samples:
        prompt_wav = prompt_wav[:target_samples]
    else:
        prompt_wav = torch.nn.functional.pad(prompt_wav, (0, target_samples - prompt_wav.shape[0]))

    # cond (backing mel condition): the actual mel of the style song, matching
    # the teacher/student's shared mel convention exactly (no ad hoc mel_transform).
    mel = compute_mel_spectrogram(prompt_wav.cpu(), config).to(device)  # (n_mels, frames)
    cond = mel.transpose(0, 1).unsqueeze(0)  # (1, frames, n_mels)

    # style_prompt: real MuQ-MuLan embedding of the style song (same 512-dim
    # space the teacher conditions on), not a resized copy of the backing mel.
    style_prompt = compute_style_embedding(prompt_wav.cpu().numpy(), device=str(device)).to(device).unsqueeze(0)

    steps = 16
    dt = 1.0 / steps
    xt = torch.randn(1, config.frames_per_chunk, config.n_mels, device=device)

    print(f"Sampling using Euler ODE Solver ({steps} steps)...")
    with torch.inference_mode():
        for step in range(steps):
            t_tensor = torch.tensor([step * dt], device=device)
            v_pred = student(x=xt, cond=cond, texts=[lyrics_text], timestep=t_tensor, style_prompt=style_prompt)
            xt = xt + dt * v_pred

    mel_out = xt.transpose(1, 2).squeeze(0)  # (n_mels, frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "student_generated_song.wav"
    render_mel_to_wav(mel_out, output_file, config, vocoder_type="vocos")

    print(f"✅ Student inference success! Generated song saved at: {output_file.resolve()}")


if __name__ == "__main__":
    run_student_inference()
