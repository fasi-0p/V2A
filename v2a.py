import os
import glob
import subprocess
import tempfile
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import scipy.signal as sps
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
VIDEO_DIR = "/kaggle/input/v2a-input"   # CHANGE THIS
OUTPUT_PATH = "/kaggle/working/v2a_output_clean.wav"

TARGET_SR = 22050
TARGET_DURATION = 10.0   # output audio length
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
GRIFFIN_LIM_ITER = 80

# ======================================================
#           Noise Reduction (DSP ONLY, NO MODELS)
# ======================================================

def denoise_audio(y):
    """
    Clean audio using energy-based segmentation.
    """
    intervals = librosa.effects.split(y, top_db=25)
    if len(intervals) == 0:
        return y
    
    cleaned = np.concatenate([y[start:end] for start, end in intervals])
    # FIX: librosa fix_length only uses keyword args
    cleaned = librosa.util.fix_length(cleaned, size=len(y))
    return cleaned

def wiener_filter_mel(mel):
    """Apply 2D Wiener filter to reduce mel noise."""
    return sps.wiener(mel, mysize=(5,5))

# ======================================================
#            FFmpeg Audio Extraction
# ======================================================
def extract_audio_from_video(video_path, sr=TARGET_SR, target_dur=TARGET_DURATION):
    """
    Robust ffmpeg-based audio extractor.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", str(sr),
            "-loglevel", "quiet",
            wav_path
        ]
        subprocess.run(cmd)

        y, _ = librosa.load(wav_path, sr=sr, mono=True)

    except Exception as e:
        print(f"Error loading audio from {video_path}: {e}")
        return None

    # Denoise
    y = denoise_audio(y)

    # pad/trim
    target_len = int(sr * target_dur)
    y = librosa.util.fix_length(y, size=target_len)

    return y.astype(np.float32)

# ======================================================
#           Mel Spectrogram + Inversion
# ======================================================
def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=1.0
    )
    mel = np.maximum(mel, 1e-8)  # avoid log of zero
    return mel

def mel_to_audio(mel, sr):
    y = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=1.0,
        n_iter=GRIFFIN_LIM_ITER
    )
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

# ======================================================
#         Visualization (Mel Spectrogram)
# ======================================================
def show_mel(mel):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        librosa.power_to_db(mel, ref=np.max),
        sr=TARGET_SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Aggregated Mel Spectrogram")
    plt.show()

# ======================================================
#            MAIN V2A AGGREGATION PIPELINE
# ======================================================
def build_aggregated_audio_from_videos():
    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.*")))
    if len(video_paths) == 0:
        raise ValueError("No videos found.")

    mel_list = []

    for p in tqdm(video_paths, desc="Processing videos"):
        seg = extract_audio_from_video(p)
        if seg is None:
            continue
        
        mel = audio_to_mel(seg, TARGET_SR)
        mel_list.append(mel)

    if len(mel_list) == 0:
        raise ValueError("No valid audio extracted.")

    # Weighted aggregation to reduce noise smearing
    mel_stack = np.stack(mel_list, axis=0)
    weights = np.linspace(0.7, 1.0, mel_stack.shape[0])[:, None, None]
    aggregated_mel = np.sum(mel_stack * weights, axis=0) / np.sum(weights)

    # Wiener filter (noise smoothing)
    aggregated_mel = wiener_filter_mel(aggregated_mel)

    # Visualize mel
    show_mel(aggregated_mel)

    # Convert mel → audio
    final_audio = mel_to_audio(aggregated_mel, TARGET_SR)

    # Ensure 10 seconds
    final_audio = librosa.util.fix_length(final_audio, size=int(TARGET_DURATION * TARGET_SR))

    # Save
    sf.write(OUTPUT_PATH, final_audio, TARGET_SR)
    print("✔ Final audio saved at:", OUTPUT_PATH)

# ======================================================
# RUN
# ======================================================
build_aggregated_audio_from_videos()
