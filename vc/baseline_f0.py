"""DSP baseline: 5 hand-made pitch features + logistic regression.

Question it answers: how much of speech-vs-singing is explained by F0 alone?
Scored on exactly the same segments and folds as the CNN.

    python -m vc.baseline_f0 --seg 0.25 0.5 1 2 3
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vc.data import HOP, SR, load_corpus, make_folds, segments, select, split
from vc.metrics import scores


def f0_track(wav: Path, cache: Path) -> np.ndarray:
    """pYIN F0 in semitones (NaN = unvoiced) on the same 10 ms grid as the log-mel."""
    if not cache.exists():
        y, _ = librosa.load(wav, sr=SR)
        f0, _, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=SR, frame_length=1024, hop_length=HOP)
        np.save(cache, 12 * np.log2(f0 / 55.0))  # semitones above A1
        print(f"  pyin {cache.name}")
    return np.load(cache)


def pitch_features(st: np.ndarray) -> np.ndarray:
    voiced = ~np.isnan(st)
    if voiced.sum() < 3:
        return np.zeros(5)
    v = st[voiced]
    d = np.abs(np.diff(st))[voiced[1:] & voiced[:-1]]  # semitone jump between voiced neighbours
    d = d if len(d) else np.zeros(1)
    return np.array([voiced.mean(),          # how much of the time there is pitch
                     np.median(v),           # pitch height
                     v.std(),                # pitch range
                     np.median(d),           # typical frame-to-frame glide
                     (d < 0.2).mean()])      # share of 'held note' frames


def features_for(recs, tracks, seg):
    _, y, meta = segments(recs, seg)
    L = round(seg * 100)
    X = np.stack([pitch_features(tracks[(recs[i]["speaker"], recs[i]["label"], recs[i]["song"])][s:s + L])
                  for i, s in meta])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/nus-smc-corpus_48")
    p.add_argument("--seg", type=float, nargs="+", default=[1.0])
    args = p.parse_args()

    recs = load_corpus(args.data)
    folds = make_folds(recs)
    cache = Path("cache/f0")
    cache.mkdir(parents=True, exist_ok=True)
    kinds = {0: "read", 1: "sing"}
    tracks = {(r["speaker"], r["label"], r["song"]):
              f0_track(Path(args.data) / r["speaker"] / kinds[r["label"]] / f"{r['song']}.wav",
                       cache / f"{r['speaker']}_{kinds[r['label']]}_{r['song']}.npy")
              for r in recs}

    out = Path("results/runs.jsonl")
    out.parent.mkdir(exist_ok=True)
    for seg in args.seg:
        for k in range(len(folds)):
            tr, _, te = split(folds, k)  # same training singers as the CNN
            Xtr, ytr = features_for(select(recs, tr), tracks, seg)
            Xte, yte = features_for(select(recs, te), tracks, seg)
            clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            res = scores(yte, clf.fit(Xtr, ytr).predict(Xte))
            print(f"seg {seg}s fold {k}: acc {res['accuracy']:.4f} macro-F1 {res['macro_f1']:.4f}")
            with out.open("a") as f:
                f.write(json.dumps(dict(model="f0_logreg", seg=seg, fold=k, test_speakers=te, **res)) + "\n")


if __name__ == "__main__":
    main()
