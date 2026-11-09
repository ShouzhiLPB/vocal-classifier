"""Test a trained model on your own recordings (never used for training).

    python eval_folder.py --ckpt checkpoints/cnn_seg1.0_fold0.pt --data data/own
Layout: data/own/<NAME>/{read,sing}/*.wav   (same layout as NUS-48E)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import predict
from vc.data import CLASS_NAMES, load_corpus, segments
from vc.metrics import scores
from vc.model import VocalCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/own")
    p.add_argument("--seg", type=float, default=1.0)
    args = p.parse_args()

    model = VocalCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    recs = load_corpus(args.data, cache_dir="cache/own")
    X, y, meta = segments(recs, args.seg)
    pred = predict(model, torch.from_numpy(X), "cpu")
    rec_of = np.array([i for i, _ in meta])

    print(f"\n{'file':45s} truth     segments  predicted singing")
    for i, r in enumerate(recs):
        m = rec_of == i
        share = f"{pred[m].mean():.0%}" if m.any() else "(all silent)"
        print(f"{r['song'][:45]:45s} {CLASS_NAMES[r['label']]:9s} {m.sum():8d}  {share}")

    res = scores(y, pred)
    print(f"\nTOTAL {len(y)} segments: acc {res['accuracy']:.4f}  macro-F1 {res['macro_f1']:.4f}  "
          f"confusion {res['confusion']}")
    Path("results").mkdir(exist_ok=True)
    Path("results/own_recordings.json").write_text(json.dumps(dict(ckpt=args.ckpt, seg=args.seg, **res), indent=2))


if __name__ == "__main__":
    main()
