#INFERENCE
import warnings
warnings.filterwarnings("ignore")

import os, subprocess
from pathlib import Path
import torch, numpy as np
import cv2
import torchaudio
import torchaudio.transforms as T_audio
import sentencepiece as spm
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import av
import soundfile as sf

# ---- CONFIG (point these to your files) ----
VIDEO_PATH = "/kaggle/input/v2a-transcripts-3/1.mp4"   # input video for inference
MODEL_DIR = "/kaggle/working/"                         # dir containing v2a_spm.model and checkpoints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- MODEL / PREPROCESS constants (must match training opt2) ----
SAMPLE_RATE = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

FPS = 4
FRAME_SIZE = 224
MAX_FRAMES = 32
MAX_TOKENS = 128

# model sizes used for the un-finetuned "opt2" training
EMBED_DIM = 320
TRANSFORMER_HEADS = 4
ENC_LAYERS = 2       # encoder used 2 layers in the opt2 training code
DEC_LAYERS = 4
PAD_ID = 0

# checkpoint names (opt2)
V2T_CKPT = os.path.join(MODEL_DIR, "v2t_res50_opt2.pt")
TTS_CKPT = os.path.join(MODEL_DIR, "tts_res50_opt2.pt")
SP_MODEL = os.path.join(MODEL_DIR, "v2a_spm.model")

# sanity checks
assert os.path.exists(SP_MODEL), f"SentencePiece model not found at {SP_MODEL}"
assert os.path.exists(V2T_CKPT), f"V2T checkpoint not found at {V2T_CKPT}"
assert os.path.exists(TTS_CKPT), f"TTS checkpoint not found at {TTS_CKPT}"

# ---- load sentencepiece ----
sp = spm.SentencePieceProcessor()
sp.Load(SP_MODEL)

def encode_text_for_model(text, max_len=MAX_TOKENS):
    ids = sp.EncodeAsIds(text)[:max_len]
    if len(ids) < max_len:
        ids = ids + [PAD_ID] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

def decode_ids(ids):
    return sp.DecodeIds([int(i) for i in ids if int(i) != PAD_ID])

# ---- mel extractor (not needed to run inference but kept for reference) ----
mel_extractor = T_audio.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH, n_mels=N_MELS, power=1.0
)

# ---- model classes (must match training exactly for opt2) ----
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = nn.Identity()

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_layers=ENC_LAYERS, heads=TRANSFORMER_HEADS):
        super().__init__()
        self.backbone = resnet
        self.proj = nn.Linear(2048, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(MAX_FRAMES, embed_dim))
    def forward(self, frames):
        # frames: (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        x = frames.view(B*T, C, H, W)
        feats = self.backbone(x)                # (B*T, 2048)
        feats = self.proj(feats).view(B, T, -1) + self.pos[:T].unsqueeze(0)
        out = self.trans(feats.permute(1,0,2)).permute(1,0,2)  # B x T x D
        return out

class TextDecoder(nn.Module):
    def __init__(self, vocab, embed=EMBED_DIM, heads=TRANSFORMER_HEADS, num_layers=DEC_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=PAD_ID)
        layer = nn.TransformerDecoderLayer(d_model=embed, nhead=heads, dim_feedforward=embed*4)
        self.trans = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out = nn.Linear(embed, vocab)
        self.pos = nn.Parameter(torch.randn(MAX_TOKENS, embed))
    def forward(self, tgt_ids, memory):
        B,S = tgt_ids.shape
        emb = self.embed(tgt_ids) + self.pos[:S].unsqueeze(0)
        emb_s = emb.permute(1,0,2)               # S x B x E
        mem = memory.permute(1,0,2)              # S_mem x B x E
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(emb.device)
        out = self.trans(emb_s, mem, tgt_mask=mask).permute(1,0,2)  # B x S x E
        return self.out(out)

class VideoToTextModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.enc = VideoEncoder()
        self.dec = TextDecoder(vocab)
    def forward(self, frames, tgt_ids):
        mem = self.enc(frames)
        return self.dec(tgt_ids, mem)

class SimpleTTS2(nn.Module):
    def __init__(self, vocab, embed=EMBED_DIM, n_mels=N_MELS):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.prenet = nn.Sequential(nn.Linear(embed, embed), nn.ReLU(), nn.LayerNorm(embed))
        self.conv1 = nn.Conv1d(embed, embed, kernel_size=5, padding=2)
        self.attn = nn.MultiheadAttention(embed, TRANSFORMER_HEADS)
        self.conv2 = nn.Conv1d(embed, n_mels, kernel_size=5, padding=2)
    def forward(self, token_ids):
        # token_ids: B x S
        x = self.emb(token_ids)            # B x S x E
        x = self.prenet(x)
        x_t = x.permute(1,0,2)             # S x B x E
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = attn_out.permute(1,2,0) # B x E x S
        conv_in = torch.relu(self.conv1(attn_out))  # B x E x S
        mel = self.conv2(conv_in)         # B x n_mels x S
        # training used raw mel (no extra scaling). Keep same format.
        return mel

# ---- instantiate models and load checkpoints ----
vocab_size_real = sp.get_piece_size()
v2t = VideoToTextModel(vocab_size_real).to(DEVICE)
tts = SimpleTTS2(vocab_size_real).to(DEVICE)

# load weights (try strict -> fallback to strict=False and report)
def try_load(model, path, device):
    sd = torch.load(path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
        print(f"Loaded {path} (strict=True).")
    except RuntimeError as e:
        print(f"Strict load failed for {path} — trying strict=False. Error: {e}")
        missing = model.load_state_dict(sd, strict=False)
        print(f"Loaded {path} with strict=False. Missing keys: {len(missing.missing)}  Unexpected keys: {len(missing.unexpected)}")
        if len(missing.missing) > 0:
            print("  examples missing keys:", missing.missing[:10])
        if len(missing.unexpected) > 0:
            print("  examples unexpected keys:", missing.unexpected[:10])

try_load(v2t, V2T_CKPT, DEVICE)
try_load(tts, TTS_CKPT, DEVICE)
v2t.eval(); tts.eval()

# ---- mel -> stft inverse + griffin ----
inverse_mel = T_audio.InverseMelScale(n_stft=N_FFT//2 + 1, n_mels=N_MELS, sample_rate=SAMPLE_RATE).to(DEVICE)
griffin = T_audio.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_iter=60).to(DEVICE)

# ---- helpers: extract frames ----
def extract_frames(video_path, max_frames=MAX_FRAMES, size=FRAME_SIZE, fps_target=FPS):
    container = av.open(str(video_path))
    transform = T.Compose([T.ToTensor(), T.Resize((size, size))])
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        img = transform(img)
        frames.append(img)
        if len(frames) >= max_frames: break
    container.close()
    if len(frames) == 0:
        raise RuntimeError("Could not decode frames — check video codec")
    while len(frames) < max_frames:
        frames.append(torch.zeros_like(frames[0]))
    # stack => (T, C, H, W) -> we need (1, T, C, H, W)
    return torch.stack(frames).unsqueeze(0)

# ---- greedy decode (autoregressive with model's decoder) ----
def greedy_decode_v2t(frames_tensor, max_tokens=MAX_TOKENS):
    frames_tensor = frames_tensor.to(DEVICE).float()
    generated = torch.zeros((1, max_tokens), dtype=torch.long).to(DEVICE)
    out_ids = []
    with torch.no_grad():
        for t in range(max_tokens - 1):
            logits = v2t(frames_tensor, generated[:, :t+1])  # (1, S, V)
            if logits.shape[1] <= t:
                break
            probs = logits[0, t]        # vocab logits
            nid = int(torch.argmax(probs).item())
            if nid == PAD_ID:
                break
            generated[0, t+1] = nid
            out_ids.append(nid)
    text = decode_ids(out_ids)
    return text, out_ids

# ---- TTS: text -> mel -> spectrogram -> waveform ----
def tts_text_to_wav(text, duration_sec, out_wav_path):
    toks = encode_text_for_model(text).to(DEVICE)
    with torch.no_grad():
        mel = tts(toks).squeeze(0)   # B removed -> (n_mels, Tpred) or (n_mels, S)
    # ensure mel shape (n_mels, T)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    # If model produced (n_mels, S) but as float; inverse_mel expects (n_mels, T)
    mel = mel.to(DEVICE)
    # inverse mel -> linear magnitude (n_fft_bins, T)
    spec = inverse_mel(mel)               # (n_fft_bins, T)
    # Griffin expects (channel, freq, time) -> pass spec.unsqueeze(0)
    wav = griffin(spec.unsqueeze(0)).squeeze(0).cpu().numpy()
    desired_len = int(SAMPLE_RATE * duration_sec)
    if desired_len > 0 and abs(desired_len - len(wav)) > 100:
        # naive resample/stretch to match duration (keeps samplerate same)
        wav = torchaudio.functional.resample(torch.tensor(wav).unsqueeze(0), orig_freq=SAMPLE_RATE, new_freq=SAMPLE_RATE).squeeze(0).numpy()
    sf.write(out_wav_path, wav, SAMPLE_RATE)
    return out_wav_path

# ---- mux helper ----
def mux_audio_with_video(orig_video_path, wav_path, out_video_path):
    cmd = ['ffmpeg', '-y', '-i', str(orig_video_path), '-i', str(wav_path),
           '-c:v', 'copy', '-c:a', 'aac', str(out_video_path), '-loglevel', 'error']
    subprocess.run(cmd, check=True)
    return out_video_path

# ---- run pipeline ----
video_path = Path(VIDEO_PATH)
frames = extract_frames(video_path)   # (1, T, C, H, W)

# compute duration robustly
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
duration = float(total_frames / fps) if (fps and total_frames) else 8.0
cap.release()

print("Running Video->Text inference...")
text, ids = greedy_decode_v2t(frames)
print("Predicted text (preview):", text[:400])

out_wav = video_path.with_suffix("").stem + "_generated.wav"
tts_text_to_wav(text, duration, str(out_wav))
print("Saved audio:", out_wav)

out_mp4 = video_path.with_name(video_path.stem + "_with_gen_audio.mp4")
try:
    mux_audio_with_video(video_path, out_wav, out_mp4)
    print("Saved muxed video:", out_mp4)
except Exception as e:
    print("Mux failed:", e)
