"""NUS-48E -> log-mel segments, with speaker-independent folds.

Pipeline: wav (44.1 kHz) -> 16 kHz mono -> log-mel (64 x T, 10 ms hop)
          -> fixed-length segments, silent ones dropped.
"""
from pathlib import Path

import librosa
import numpy as np

SR = 16_000
N_FFT, WIN, HOP, N_MELS = 512, 400, 160, 64  # 25 ms window, 10 ms hop
FPS = SR // HOP                               # 100 frames per second
LABELS = {"read": 0, "sing": 1}
CLASS_NAMES = ["speech", "singing"]


def logmel(y: np.ndarray) -> np.ndarray:
    """Waveform at SR -> log-mel in dB, shape (N_MELS, T)."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, win_length=WIN,
                                       hop_length=HOP, n_mels=N_MELS, fmin=50, fmax=8000)
    return librosa.power_to_db(m, ref=1.0, top_db=None).astype(np.float32)


def activity(y: np.ndarray, floor_db: float = 35.0) -> np.ndarray:
    """Per-frame bool: is this frame within `floor_db` of the file's loud frames?"""
    rms_db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0] + 1e-10)
    return rms_db > np.percentile(rms_db, 99) - floor_db


def load_corpus(root, cache_dir="cache/nus") -> list[dict]:
    """One record per wav: speaker, song, label, feat (N_MELS x T), active (T,).

    Features are computed once and cached as .npz (first run takes a few minutes).
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    recs = []
    for wav in sorted(Path(root).glob("*/*/*.wav")):
        speaker, kind = wav.parts[-3], wav.parts[-2]
        if kind not in LABELS:
            continue
        out = cache / f"{speaker}_{kind}_{wav.stem}.npz"
        if not out.exists():
            y, _ = librosa.load(wav, sr=SR, mono=True)
            feat, act = logmel(y), activity(y)
            n = min(feat.shape[1], len(act))
            np.savez(out, feat=feat[:, :n].astype(np.float16), active=act[:n])
            print(f"  cached {out.name}")
        d = np.load(out)
        recs.append(dict(speaker=speaker, song=wav.stem, label=LABELS[kind],
                         feat=d["feat"], active=d["active"]))
    if not recs:
        raise FileNotFoundError(f"no */read|sing/*.wav under {root}")
    return recs


def segments(recs, seg_sec: float, hop_sec: float | None = None, min_active: float = 0.5):
    """Cut records into (N, N_MELS, L) float16 segments.

    Returns X, y, meta where meta[i] = (record index, start frame) so other
    models (the F0 baseline) can be scored on exactly the same segments.
    """
    L = round(seg_sec * FPS)
    H = round((hop_sec or seg_sec) * FPS)
    X, y, meta = [], [], []
    for i, r in enumerate(recs):
        f, a = r["feat"], r["active"]
        for s in range(0, f.shape[1] - L + 1, H):
            if a[s:s + L].mean() >= min_active:
                X.append(f[:, s:s + L])
                y.append(r["label"])
                meta.append((i, s))
    return np.stack(X).astype(np.float16), np.array(y, dtype=np.int64), meta


def make_folds(recs) -> list[list[str]]:
    """Group speakers who sang the same set of songs.

    In NUS-48E this gives 6 pairs of singers (by design each song is sung by
    a male and a female singer). Splitting by these groups guarantees that
    test singers are never heard during training.
    """
    songs: dict[str, set] = {}
    for r in recs:
        songs.setdefault(r["speaker"], set()).add(r["song"])
    groups: dict[frozenset, list] = {}
    for spk, s in sorted(songs.items()):
        groups.setdefault(frozenset(s), []).append(spk)
    return sorted(sorted(g) for g in groups.values())


def split(folds, k: int):
    """Fold k is test, fold k+1 is validation, the rest is train."""
    v = (k + 1) % len(folds)
    train = [s for i, f in enumerate(folds) if i not in (k, v) for s in f]
    return train, folds[v], folds[k]


def select(recs, speakers):
    keep = set(speakers)
    return [r for r in recs if r["speaker"] in keep]
