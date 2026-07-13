import os
import sys
import torch
import torchaudio
import json
import random
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "DiffRhythm2-main"))

from src.models.text_to_music_diffusion import MusicDiffusionConfig
from src.models.dit_transformer import MicroDiT
from bigvgan.model import Generator
import pedalboard

def run_student_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running distilled student inference on device: {device}")
    
    # 1. Download official decoder weights for final audio decoding (BigVGAN)
    repo_id = "ASLP-lab/DiffRhythm2"
    print("Downloading BigVGAN decoder config and weights...")
    decoder_config_path = hf_hub_download(repo_id=repo_id, filename="decoder.json", local_dir="./ckpt")
    decoder_ckpt_path = hf_hub_download(repo_id=repo_id, filename="decoder.bin", local_dir="./ckpt")
    
    # Initialize BigVGAN decoder
    decoder = Generator(decoder_config_path, decoder_ckpt_path).to(device)
    decoder.eval()
        
    # 2. Initialize Student Model (MicroDiT)
    config = MusicDiffusionConfig(
        n_mels=64,
        frames_per_chunk=128  # seq_len = 128
    )
    student = MicroDiT(config, dim=256, depth=4, heads=4).to(device)
    
    # Load distilled student weights
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
        
    # 3. Setup input data
    lyrics_text = "Đêm nay mưa rơi rơi trên lối mòn xưa cũ. Lòng anh nhớ em nhiều người ơi có biết chăng."
    
    # Locate style song (supports local and Kaggle paths)
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
    
    # Load and process style audio to obtain mel condition (backing_mel) & style prompt
    # Note: DiffRhythm2 uses 24kHz audio and Mel dimension 80
    prompt_wav, sr = torchaudio.load(style_song)
    prompt_wav = torchaudio.functional.resample(prompt_wav.to(device), sr, 24000)
    
    # Target length of 128 frames
    target_samples = 128 * 480 # hop size is typically around 480 for 5Hz frame-level models or similar
    if prompt_wav.shape[1] > target_samples:
        prompt_wav = prompt_wav[:, :target_samples]
    else:
        import torch.nn.functional as F
        prompt_wav = F.pad(prompt_wav, (0, target_samples - prompt_wav.shape[1]))
    prompt_wav = prompt_wav.mean(dim=0, keepdim=True) # Mono
    
    # Compute Mel spectrogram of prompt_wav to act as the cond (backing_mel) and style_prompt
    # For Student model, we use 64 Mel bins
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        win_length=1024,
        hop_length=480,
        n_mels=64
    ).to(device)
    
    with torch.no_grad():
        mel = mel_transform(prompt_wav) # (1, 64, seq_len)
        # Log scaling
        mel = torch.log(torch.clamp(mel, min=1e-5))
        # Match expected student length of 128
        if mel.shape[2] > 128:
            mel = mel[:, :, :128]
        elif mel.shape[2] < 128:
            mel = torch.nn.functional.pad(mel, (0, 128 - mel.shape[2]))
            
    cond = mel.transpose(1, 2) # (1, 128, 64)
    style_prompt = cond # (1, 128, 64)
        
    # 4. ODE Solver (Euler) to denoise vocal mel
    steps = 16
    dt = 1.0 / steps
    
    # Initial noise x0 (1, 128, 64)
    xt = torch.randn(1, 128, 64, device=device)
        
    print(f"Sampling using Euler ODE Solver ({steps} steps)...")
    with torch.inference_mode():
        for step in range(steps):
            t_val = step * dt
            t_tensor = torch.tensor([t_val], device=device)
                
            # Predict velocity
            v_pred = student(
                x=xt,
                cond=cond,
                texts=[lyrics_text],
                timestep=t_tensor,
                style_prompt=style_prompt
            )
            
            # Euler step: xt_next = xt + dt * v_pred
            xt = xt + dt * v_pred
            
        # xt is now the denoised vocal mel (1, 128, 64)
        # Transpose to channel-first (1, 64, 128)
        latent = xt.transpose(1, 2)
        
        # Decode mel to audio via BigVGAN
        audio = decoder.decode_audio(latent, overlap=5, chunk_size=20)
        
        # Save output
        audio_np = audio.float().cpu().numpy().squeeze()[None, :]
        
        # Make fake stereo
        left_channel = audio_np
        right_channel = audio_np.copy() * 0.8
        delay_samples = int(0.01 * decoder.h.sampling_rate)
        right_channel = np.roll(right_channel, delay_samples)
        right_channel[:, :delay_samples] = 0
        stereo_audio = np.concatenate([left_channel, right_channel], axis=0)
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir / "student_generated_song.mp3"
        with pedalboard.io.AudioFile(str(output_file), "w", decoder.h.sampling_rate, 2) as f:
            f.write(stereo_audio)
            
    print(f"✅ Student inference success! Generated song saved at: {output_file.resolve()}")

if __name__ == "__main__":
    run_student_inference()
