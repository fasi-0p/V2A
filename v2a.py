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
from scipy.ndimage import median_filter
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
VIDEO_DIR = "/kaggle/input/v2a-bird-dataset"     # CHANGE THIS
OUTPUT_PATH = "/kaggle/working/v2a_output_clean3.wav"

TARGET_SR = 22050
TARGET_DURATION = 10.0
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
GRIFFIN_LIM_ITER = 200    # higher = cleaner

# ======================================================
#                    DSP UTILITIES
# ======================================================

def bandpass_filter(y, sr, low=80, high=8000, order=4):
    ny = 0.5 * sr
    low_norm = low / ny
    high_norm = high / ny
    b, a = sps.butter(order, [low_norm, high_norm], btype="band")
    return sps.filtfilt(b, a, y)

def estimate_noise_spectrum(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, top_db=35):
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        quiet = y[:int(0.5 * sr)]
    else:
        energies = [(y[s:e] ** 2).mean() for s, e in intervals]
        idx = np.argmin(energies)
        s, e = intervals[idx]
        quiet = y[s:e]
        if e - s < int(0.2 * sr):
            quiet = y[:int(0.5 * sr)]

    S = librosa.stft(quiet, n_fft=n_fft, hop_length=hop_length)
    noise_mag = np.abs(S).mean(axis=1)
    return noise_mag

def spectral_subtract(y, sr, n_fft=N_FFT, hop_length=HOP_LENGTH, noise_scale=1.0):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(S), np.angle(S)

    noise_mag = estimate_noise_spectrum(y, sr, n_fft, hop_length)
    noise_mag = noise_mag[:, None] * noise_scale

    mag_clean = np.maximum(mag - noise_mag, 1e-8)
    S_clean = mag_clean * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean, hop_length=hop_length, length=len(y))

    return y_clean

def energy_denoise(y, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y

    cleaned = np.concatenate([y[s:e] for s, e in intervals])
    return librosa.util.fix_length(cleaned, size=len(y))

def enhance_audio(y, sr):
    # bandpass → spectral subtraction → energy cleanup
    y = bandpass_filter(y, sr)
    y = spectral_subtract(y, sr, noise_scale=1.0)
    y = energy_denoise(y, top_db=30)
    return y

# ======================================================
#                MEL PROCESSING
# ======================================================

def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, power=1.0
    )
    return np.maximum(mel, 1e-8)

def smooth_mel(mel):
    mel = median_filter(mel, size=(3, 3))
    mel = sps.wiener(mel, mysize=(9, 9))
    return np.maximum(mel, 1e-8)

def mel_to_audio(mel, sr):
    y = librosa.feature.inverse.mel_to_audio(
        mel, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_iter=GRIFFIN_LIM_ITER,
        power=1.0
    )
    return y / (np.max(np.abs(y)) + 1e-9)

# ======================================================
#                FFMPEG AUDIO EXTRACTION
# ======================================================

def extract_audio(video_path, sr=TARGET_SR, target_dur=TARGET_DURATION):
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
        print("Error extracting audio:", e)
        return None

    # DSP enhancement
    y = enhance_audio(y, sr)

    # enforce duration
    y = librosa.util.fix_length(y, size=int(sr * target_dur))

    # normalize per-clip energy
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    y = y / rms

    return y.astype(np.float32)

# ======================================================
#                V2A AGGREGATION PIPELINE
# ======================================================

def build_aggregated_audio():
    video_paths = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.*")))
    if not video_paths:
        raise ValueError("No videos found in VIDEO_DIR.")

    mel_list = []

    for p in tqdm(video_paths, desc="Processing videos"):
        seg = extract_audio(p)
        if seg is None:
            continue

        # reject overly quiet clips
        if np.sqrt(np.mean(seg**2)) < 1e-4:
            continue

        mel = audio_to_mel(seg, TARGET_SR)
        mel_list.append(mel)

    if not mel_list:
        raise ValueError("No valid audio extracted.")

    # log-domain weighted average
    mel_stack = np.stack(mel_list, axis=0)
    log_stack = np.log(np.maximum(mel_stack, 1e-8))

    weights = np.linspace(0.7, 1.0, mel_stack.shape[0])[:, None, None]
    aggregated_log = np.sum(log_stack * weights, axis=0) / np.sum(weights)

    aggregated_mel = np.exp(aggregated_log)

    # smoothing
    aggregated_mel = smooth_mel(aggregated_mel)

    # invert mel → audio
    final_audio = mel_to_audio(aggregated_mel, TARGET_SR)
    final_audio = librosa.util.fix_length(
        final_audio,
        size=int(TARGET_DURATION * TARGET_SR)
    )

    sf.write(OUTPUT_PATH, final_audio, TARGET_SR)
    print("✔ Clean final audio saved at:", OUTPUT_PATH)

# ======================================================
# RUN
# ======================================================
build_aggregated_audio()
