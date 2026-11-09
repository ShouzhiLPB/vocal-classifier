"""Cross-dataset check: every ccmusic clip is SUNG, so a good model should say 'singing'.

    python eval_ccmusic.py --ckpt checkpoints/cnn_seg0.5_fold0.pt
Reads the Hugging Face .arrow files directly with pyarrow (no `datasets` needed).
"""
import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import pyarrow as pa
import soundfile as sf
import torch

from vc.data import SR, logmel
from vc.model import VocalCNN

METHOD = {0: "falsetto", 1: "chest"}
GENDER = {0: "female", 1: "male"}


def clips(root):
    for f in sorted(Path(root).glob("*/*.arrow")):
        table = pa.ipc.open_stream(f.open("rb")).read_all()
        for row in table.select(["audio", "singing_method", "gender"]).to_pylist():
            y, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            yield librosa.resample(y.mean(1), orig_sr=sr, target_sr=SR), row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/chest_falsetto/default")
    args = p.parse_args()

    model = VocalCNN()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    hits = {}
    with torch.no_grad():
        for y, row in clips(args.data):
            feat = logmel(y)
            if feat.shape[1] < 16:  # shorter than 4 poolings can handle
                continue
            sung = int(model(torch.from_numpy(feat)[None]).argmax(1)) == 1
            hits.setdefault(f"{GENDER[row['gender']]}_{METHOD[row['singing_method']]}", []).append(sung)

    res = {k: dict(n=len(v), predicted_singing=float(np.mean(v))) for k, v in sorted(hits.items())}
    every = [h for v in hits.values() for h in v]
    res["all"] = dict(n=len(every), predicted_singing=float(np.mean(every)))
    print(json.dumps(res, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/ccmusic_ood.json").write_text(json.dumps(dict(ckpt=args.ckpt, **res), indent=2))


if __name__ == "__main__":
    main()
