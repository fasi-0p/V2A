import warnings
warnings.filterwarnings("ignore")

# ---- install optional deps (runs in notebook) ----
# sentencepiece for tokenizer
# try:
#     import sentencepiece as spm
# except:
#     !pip install --quiet sentencepiece
#     import sentencepiece as spm

# safetensors for optional checkpoint format
try:
    from safetensors.torch import save_file as safetensor_save
    HAVE_SAFETENSORS = True
except:
    HAVE_SAFETENSORS = False

# standard imports
import os, random, subprocess, math, json, tempfile
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torchaudio, torchaudio.transforms as T_audio
import cv2
import av

# ========== CONFIG ==========
VIDEO_DIR = "C:\Users\FASI OWAIZ AHMED\Desktop\v2a\videos"           # place your .mp4 files here
TRANSCRIPT_DIR = "C:\Users\FASI OWAIZ AHMED\Desktop\v2a\transcripts" # timestamped .txt files (same basename)
EXTRACTED_WAV_DIR = "audio_segs"
MEL_DIR = "mels"
OUTPUT_DIR = "weights"
os.makedirs(EXTRACTED_WAV_DIR, exist_ok=True)
os.makedirs(MEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

FPS = 4
FRAME_SIZE = 224
MAX_FRAMES = 32

VOCAB_SIZE = 8000  # sentencepiece vocab size
MAX_TOKENS = 128
PAD_ID = 0

EMBED_DIM = 320
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 4

BATCH_SIZE = 4
LR = 2e-4
EPOCHS_V2T = 20
EPOCHS_TTS = 40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ========== torchaudio mel extractor ==========
mel_extractor = T_audio.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    power=1.0
)

# inverse mel + griffinlim (for inference later)
# inverse_mel = T_audio.InverseMelScale(n_stft=N_FFT//2+1, n_mels=N_MELS, sample_rate=SAMPLE_RATE, max_iter=32)
# griffin = T_audio.GriffinLim(n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, power=1.0, n_iter=60)

def wav_to_mel_np(wav_path, mel_path):
    try:
        wav, sr = torchaudio.load(wav_path)  # (C, T)
        if wav.size(1) < 512:
            return False
        if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
        mel = mel_extractor(wav)  # (1, n_mels, T)
        mel = torch.log1p(mel).squeeze(0)  # (n_mels, T)
        np.save(mel_path, mel.cpu().numpy().astype(np.float32))
        return True
    except Exception as e:
        return False

# ========== transcript parsing ==========
def parse_timestamped_transcript(path: str):
    segs = []
    if not os.path.exists(path):
        return segs
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if "-->" not in ln: continue
            try:
                left, txt = ln.split("  ", 1)
                s, _, e = left.split()
                segs.append({"start": float(s), "end": float(e), "text": txt.strip()})
            except:
                continue
    return segs

# ========== extract wav segments & mel dataset prep ==========
print("Preparing audio segments & mel files...")
video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")])
for vid in tqdm(video_files):
    base = Path(vid).stem
    tfile = os.path.join(TRANSCRIPT_DIR, base + ".txt")
    segs = parse_timestamped_transcript(tfile)
    vpath = os.path.join(VIDEO_DIR, vid)

    if len(segs) == 0:
        # fallback: split into 8s segments
        with av.open(vpath) as container:
            # safe fallback
            if len(container.streams.video) == 0:
                continue
            duration = float(container.streams.video[0].duration * container.streams.video[0].time_base)
        segs = []
        s = 0.0
        while s < duration:
            e = min(duration, s + 8.0)
            segs.append({"start": s, "end": e, "text": ""})
            s = e

    for i, seg in enumerate(segs):
        wav_out = os.path.join(EXTRACTED_WAV_DIR, f"{base}_{i:04d}.wav")
        mel_out = os.path.join(MEL_DIR, f"{base}_{i:04d}.npy")
        if os.path.exists(mel_out): continue
        # ffmpeg extract
        cmd = ["ffmpeg", "-y", "-i", vpath, "-ss", str(seg["start"]), "-to", str(seg["end"]),
               "-ar", str(SAMPLE_RATE), "-ac", "1", wav_out, "-loglevel", "error"]
        try:
            subprocess.run(cmd, check=True)
        except:
            continue
        ok = wav_to_mel_np(wav_out, mel_out)
        if not ok and os.path.exists(wav_out):
            os.remove(wav_out)
print("Data prep done.")

# ========== Train SentencePiece tokenizer on transcripts ==========
# combine all transcript text (quick)
all_text_path = os.path.join(tempfile.gettempdir(), "v2a_all_text.txt")
with open(all_text_path, "w", encoding="utf-8") as out:
    for vid in video_files:
        base = Path(vid).stem
        tfile = os.path.join(TRANSCRIPT_DIR, base + ".txt")
        segs = parse_timestamped_transcript(tfile)
        for s in segs:
            if s["text"].strip():
                out.write(s["text"].strip() + "\n")
# if text empty, write a dummy
if os.path.getsize(all_text_path) == 0:
    with open(all_text_path, "w") as out:
        out.write("hello this is a sample transcript. ")

sp_model = os.path.join(OUTPUT_DIR, "v2a_spm")
spm.SentencePieceTrainer.Train(
    f"--input={all_text_path} --model_prefix={sp_model} --vocab_size={VOCAB_SIZE} --model_type=bpe --unk_id=1 --pad_id=0 --bos_id=-1 --eos_id=-1"
)
sp = spm.SentencePieceProcessor()
sp.Load(sp_model + ".model")
VOCAB_SIZE_REAL = sp.get_piece_size()

def encode_text(text, max_len=MAX_TOKENS):
    ids = sp.EncodeAsIds(text)[:max_len]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def decode_ids(ids):
    # ids: list or tensor
    return sp.DecodeIds([int(i) for i in ids if int(i) != 0])

# ========== Datasets & collate ==========
class VideoTextDataset(Dataset):
    def __init__(self):
        self.videos = [v for v in sorted(os.listdir(VIDEO_DIR)) if v.endswith(".mp4")]
        self.transform = T.Compose([T.ToTensor(), T.Resize((FRAME_SIZE, FRAME_SIZE))])
    def __len__(self): return len(self.videos)
    def __getitem__(self, idx):
        fname = self.videos[idx]
        base = Path(fname).stem
        vpath = os.path.join(VIDEO_DIR, fname)
        tpath = os.path.join(TRANSCRIPT_DIR, base + ".txt")
        segs = parse_timestamped_transcript(tpath)
        if len(segs) == 0:
            segs = [{"start": 0.0, "end": 8.0, "text": ""}]
        text = " ".join([s["text"] for s in segs]).strip() or " "
        toks = encode_text(text)
        # extract frames
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or (fps * 8))
        step = max(1, int(fps / FPS))
        frames = []
        for f in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, img = cap.read()
            if not ret: break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            frames.append(img)
            if len(frames) >= MAX_FRAMES: break
        cap.release()
        if len(frames) < MAX_FRAMES:
            pad = [torch.zeros(3, FRAME_SIZE, FRAME_SIZE)] * (MAX_FRAMES - len(frames))
            frames += pad
        frames = torch.stack(frames)
        return frames, toks

class TextMelDataset(Dataset):
    def __init__(self):
        self.mels = sorted([f for f in os.listdir(MEL_DIR) if f.endswith(".npy")])
    def __len__(self): return len(self.mels)
    def __getitem__(self, idx):
        fname = self.mels[idx]
        base, segid = fname[:-4].rsplit("_", 1)
        segid = int(segid)
        tpath = os.path.join(TRANSCRIPT_DIR, base + ".txt")
        segs = parse_timestamped_transcript(tpath)
        txt = segs[segid]["text"] if segid < len(segs) else ""
        toks = encode_text(txt or " ")
        mel = torch.from_numpy(np.load(os.path.join(MEL_DIR, fname))).float() # (n_mels, T)
        return toks, mel

def tts_collate_fn(batch):
    toks = torch.stack([b[0] for b in batch])
    mels = [b[1] for b in batch]
    maxlen = max(m.shape[1] for m in mels)
    padded = []
    for m in mels:
        if m.shape[1] < maxlen:
            pad = torch.zeros(m.shape[0], maxlen - m.shape[1])
            m = torch.cat([m, pad], dim=1)
        padded.append(m)
    mels = torch.stack(padded)
    return toks, mels

# ========== Models (improved small architectures) ==========
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
for p in resnet.parameters(): p.requires_grad = False
resnet.fc = nn.Identity()

class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.backbone = resnet
        self.proj = nn.Linear(2048, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, TRANSFORMER_HEADS, embed_dim*4)
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos = nn.Parameter(torch.randn(MAX_FRAMES, embed_dim))
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        x = frames.view(B*T, C, H, W)
        feats = self.backbone(x)  # (B*T, 2048)
        feats = self.proj(feats).view(B, T, -1) + self.pos[:T].unsqueeze(0)
        out = self.trans(feats.permute(1,0,2)).permute(1,0,2)
        return out

class TextDecoder(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE_REAL, embed=EMBED_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        layer = nn.TransformerDecoderLayer(embed, TRANSFORMER_HEADS, embed*4)
        self.trans = nn.TransformerDecoder(layer, num_layers=TRANSFORMER_LAYERS)
        self.out = nn.Linear(embed, vocab)
        self.pos = nn.Parameter(torch.randn(MAX_TOKENS, embed))
    def forward(self, tgt_ids, memory):
        B,S = tgt_ids.shape
        emb = self.embed(tgt_ids) + self.pos[:S].unsqueeze(0)
        emb = emb.permute(1,0,2)
        mem = memory.permute(1,0,2)
        mask = nn.Transformer.generate_square_subsequent_mask(S).to(emb.device)
        out = self.trans(emb, mem, tgt_mask=mask).permute(1,0,2)
        return self.out(out)

class VideoToTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = VideoEncoder()
        self.dec = TextDecoder(VOCAB_SIZE_REAL)
    def forward(self, frames, tgt_ids):
        mem = self.enc(frames)
        return self.dec(tgt_ids, mem)

# improved TTS (attention + convs)
class SimpleTTS2(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE_REAL, embed=EMBED_DIM, n_mels=N_MELS):
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
        x_t = x.permute(1,0,2)             # S x B x E (for attn)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = attn_out.permute(1,2,0) # B x E x S
        conv_in = torch.relu(self.conv1(attn_out))  # B x E x S
        mel = self.conv2(conv_in)         # B x n_mels x S
        return mel

# ========== create loaders & models ==========
v2t_ds = VideoTextDataset()
tts_ds = TextMelDataset()
v2t_loader = DataLoader(v2t_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
tts_loader = DataLoader(tts_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=tts_collate_fn)

v2t_model = VideoToTextModel().to(DEVICE)
tts_model = SimpleTTS2().to(DEVICE)

opt_v2t = torch.optim.Adam(v2t_model.parameters(), lr=LR)
opt_tts = torch.optim.Adam(tts_model.parameters(), lr=LR)

ce = nn.CrossEntropyLoss(ignore_index=PAD_ID)
mse = nn.MSELoss()

# ========== Train Video->Text ==========
print("Training Video→Text ...")
for ep in range(EPOCHS_V2T):
    v2t_model.train()
    total = 0.0
    for frames, toks in tqdm(v2t_loader):
        frames = frames.to(DEVICE)
        toks = toks.to(DEVICE)
        inp = toks[:, :-1]
        tar = toks[:, 1:]
        logits = v2t_model(frames, inp)  # B x S x V
        loss = ce(logits.reshape(-1, VOCAB_SIZE_REAL), tar.reshape(-1))
        opt_v2t.zero_grad(); loss.backward(); opt_v2t.step()
        total += loss.item()
    print(f"Epoch {ep+1}/{EPOCHS_V2T} loss: {total/len(v2t_loader):.4f}")

# save v2t
torch.save(v2t_model.state_dict(), os.path.join(OUTPUT_DIR, "v2t_res50_opt2.pt"))
if HAVE_SAFETENSORS:
    # save tensors into safetensor format (optional)
    import collections
    sd = v2t_model.state_dict()
    safetensor_save({k:sd[k].cpu() for k in sd}, os.path.join(OUTPUT_DIR, "v2t_res50_opt2.safetensors"))
print("Saved v2t")

# ========== Train Text->Mel ==========
print("Training Text→Mel ...")
for ep in range(EPOCHS_TTS):
    tts_model.train()
    total = 0.0
    for toks, mels in tqdm(tts_loader):
        toks = toks.to(DEVICE)
        mels = mels.to(DEVICE)
        pred = tts_model(toks)  # B x n_mels x S_pred
        L = min(pred.size(2), mels.size(2))
        loss = mse(pred[:, :, :L], mels[:, :, :L])
        opt_tts.zero_grad(); loss.backward(); opt_tts.step()
        total += loss.item()
    print(f"Epoch {ep+1}/{EPOCHS_TTS} loss: {total/len(tts_loader):.4f}")

# save tts
torch.save(tts_model.state_dict(), os.path.join(OUTPUT_DIR, "tts_res50_opt2.pt"))
if HAVE_SAFETENSORS:
    sd = tts_model.state_dict()
    safetensor_save({k:sd[k].cpu() for k in sd}, os.path.join(OUTPUT_DIR, "tts_res50_opt2.safetensors"))
print("Saved tts")
print("TRAINING COMPLETE. Models saved to", OUTPUT_DIR)
