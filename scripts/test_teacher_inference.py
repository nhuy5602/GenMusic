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

from muq import MuQMuLan
from diffrhythm2.cfm import CFM
from diffrhythm2.backbones.dit import DiT
from bigvgan.model import Generator
import inference as _inference_module
from inference import CNENTokenizer, parse_lyrics, make_fake_stereo, inference

def run_teacher_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running teacher inference test on device: {device}")
    
    # 1. Download official weights from HF
    repo_id = "ASLP-lab/DiffRhythm2"
    print(f"Downloading pretrained teacher from HF repo '{repo_id}'...")
    
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir="./ckpt")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir="./ckpt")
    decoder_config_path = hf_hub_download(repo_id=repo_id, filename="decoder.json", local_dir="./ckpt")
    decoder_ckpt_path = hf_hub_download(repo_id=repo_id, filename="decoder.bin", local_dir="./ckpt")
    
    with open(config_path) as f:
        model_config = json.load(f)
        
    # 2. Instantiate models
    model_config['use_flex_attn'] = False
    model = CFM(
        transformer=DiT(**model_config),
        num_channels=model_config['mel_dim'],
        block_size=model_config['block_size'],
    ).to(device)
    
    # Load safetensors
    from safetensors.torch import load_file
    ckpt = load_file(model_path)
    model.load_state_dict(ckpt)
    model.eval()
    
    # Load MuLAN & decoder
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir="./ckpt").to(device)
    
    # Initialize tokenizer and inject into inference module's global (required by parse_lyrics)
    lrc_tokenizer = CNENTokenizer()
    _inference_module.lrc_tokenizer = lrc_tokenizer  # patch global so parse_lyrics() works
    
    decoder = Generator(decoder_config_path, decoder_ckpt_path).to(device)
    
    # 3. Setup input data
    lyrics_text = "[start]\n[verse]\nĐêm nay mưa rơi rơi trên lối mòn xưa cũ.\nLòng anh nhớ em nhiều người ơi có biết chăng.\n[stop]"
    lyrics_token = parse_lyrics(lyrics_text)
    lyrics_token = torch.tensor(sum(lyrics_token, []), dtype=torch.long, device=device)
    
    # Locate style song (supports local and Kaggle paths)
    if Path("/kaggle/input").exists():
        songs_dir = Path("/kaggle/input")
        mp3_files = list(songs_dir.rglob("*.mp3")) + list(songs_dir.rglob("*.wav"))
        output_dir = Path("/kaggle/working")
    else:
        songs_dir = PROJECT_ROOT / "dataset" / "vietnamese_songs"
        mp3_files = list(songs_dir.glob("*.mp3")) + list(songs_dir.glob("*.wav"))
        output_dir = PROJECT_ROOT / "outputs" / "teacher_test"
    
    if not mp3_files:
        print("[ERROR] Please add at least one mp3/wav song to use as a style anchor.")
        return
        
    style_song = mp3_files[0]
    print(f"Using style anchor song: {style_song.name}")
    
    # Load & resample style audio
    prompt_wav, sr = torchaudio.load(style_song)
    prompt_wav = torchaudio.functional.resample(prompt_wav.to(device), sr, 24000)
    if prompt_wav.shape[1] > 24000 * 10:
        start = random.randint(0, prompt_wav.shape[1] - 24000 * 10)
        prompt_wav = prompt_wav[:, start:start+24000*10]
    prompt_wav = prompt_wav.mean(dim=0, keepdim=True)
    
    # Extract style embedding using MuLAN
    with torch.no_grad():
        style_prompt_embed = mulan(wavs = prompt_wav).to(device).squeeze(0)
        
    if device.type != 'cpu':
        model = model.half()
        decoder = decoder.half()
        style_prompt_embed = style_prompt_embed.half()
        
    # 4. Generate audio
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating audio using the official DiffRhythm 2 teacher model...")
    with torch.inference_mode():
        latent = model.sample_block_cache(
            text=lyrics_token.unsqueeze(0),
            duration=30, # 30 frames (around 6 seconds)
            style_prompt=style_prompt_embed.unsqueeze(0),
            steps=16,
            cfg_strength=2.0,
            process_bar=True
        )
        latent = latent.transpose(1, 2)
        audio = decoder.decode_audio(latent, overlap=5, chunk_size=20)
        
        # Save output
        audio_np = audio.float().cpu().numpy().squeeze()[None, :]
        # Make stereo
        left_channel = audio_np
        right_channel = audio_np.copy() * 0.8
        delay_samples = int(0.01 * decoder.h.sampling_rate)
        right_channel = np.roll(right_channel, delay_samples)
        right_channel[:, :delay_samples] = 0
        stereo_audio = np.concatenate([left_channel, right_channel], axis=0)
        
        import pedalboard
        output_file = output_dir / "teacher_generated_song.mp3"
        with pedalboard.io.AudioFile(str(output_file), "w", decoder.h.sampling_rate, 2) as f:
            f.write(stereo_audio)
            
    print(f"✅ Success! Generated test song at: {output_file.resolve()}")

if __name__ == "__main__":
    run_teacher_test()
