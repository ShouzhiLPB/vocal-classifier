"""Train + evaluate the CNN with speaker-independent cross-validation.

    python train.py --seg 1.0 --fold 0          # one run (tonight)
    python train.py --seg 0.25 0.5 1 2 3        # all folds, all lengths
Each run appends one JSON line to results/runs.jsonl.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vc.data import N_MELS, FPS, load_corpus, logmel, make_folds, segments, select, split
from vc.metrics import latency_ms, n_macs, n_params, scores
from vc.model import VocalCNN


def spec_augment(x: torch.Tensor, max_f: int = 8, max_t_frac: float = 0.15) -> torch.Tensor:
    """SpecAugment: hide one random band of mel bins and one random stretch of frames."""
    B, M, T = x.shape

    def band(n, width):
        w = torch.randint(0, width + 1, (B, 1), device=x.device)
        s = (torch.rand(B, 1, device=x.device) * (n - w + 1)).long()
        r = torch.arange(n, device=x.device)
        return (r >= s) & (r < s + w)

    mask = band(M, max_f)[:, :, None] | band(T, max(1, int(T * max_t_frac)))[:, None, :]
    return torch.where(mask, x.mean(dim=(1, 2), keepdim=True), x)


@torch.no_grad()
def predict(model, X, device, bs: int = 512) -> np.ndarray:
    model.eval()
    out = [model(X[i:i + bs].to(device).float()).argmax(1).cpu() for i in range(0, len(X), bs)]
    return torch.cat(out).numpy()


def run(recs, folds, k: int, seg: float, args, device) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tr_spk, va_spk, te_spk = split(folds, k)
    Xtr, ytr, _ = segments(select(recs, tr_spk), seg, hop_sec=seg / 2)  # overlap = more training data
    Xva, yva, _ = segments(select(recs, va_spk), seg)
    Xte, yte, _ = segments(select(recs, te_spk), seg)
    Xtr, Xva, Xte = map(torch.from_numpy, (Xtr, Xva, Xte))
    ytr_t = torch.from_numpy(ytr)
    print(f"fold {k} seg {seg}s | train {len(ytr)} val {len(yva)} test {len(yte)} | test={te_spk}")

    model = VocalCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    w = torch.tensor(len(ytr) / (2 * np.bincount(ytr, minlength=2)), dtype=torch.float32, device=device)

    best, best_state, wait = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            xb = spec_augment(Xtr[idx].to(device).float())
            loss = F.cross_entropy(model(xb), ytr_t[idx].to(device), weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_f1 = scores(yva, predict(model, Xva, device))["macro_f1"]
        print(f"  epoch {epoch:2d}  loss {loss.item():.3f}  val macro-F1 {val_f1:.4f}")
        if val_f1 > best:
            best, wait = val_f1, 0
            best_state = {n: t.detach().clone() for n, t in model.state_dict().items()}
        elif (wait := wait + 1) >= args.patience:
            break

    model.load_state_dict(best_state)
    test = scores(yte, predict(model, Xte, device))  # the ONLY time the test set is used
    ckpt = Path("checkpoints") / f"cnn_seg{seg}_fold{k}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"  TEST acc {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}  confusion {test['confusion']}")
    return dict(model="cnn", seg=seg, fold=k, test_speakers=te_spk, val_macro_f1=best, **test,
                n_train=len(ytr), n_test=len(yte), epochs_run=epoch + 1)


def cost(seg: float) -> dict:
    """Params, MACs and single-thread CPU latency for ONE segment (batch size 1)."""
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = VocalCNN().eval()
    x = torch.randn(1, N_MELS, round(seg * FPS))
    wav = np.random.randn(round(seg * 16_000)).astype(np.float32)
    with torch.no_grad():
        res = dict(params=n_params(model), macs=n_macs(model, x),
                   cpu_model_ms=latency_ms(lambda: model(x)),
                   cpu_feature_plus_model_ms=latency_ms(lambda: model(torch.from_numpy(logmel(wav))[None])))
    torch.set_num_threads(n_threads)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/nus-smc-corpus_48")
    p.add_argument("--seg", type=float, nargs="+", default=[1.0], help="segment length(s) in seconds")
    p.add_argument("--fold", default="all", help="fold index, or 'all'")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    recs = load_corpus(args.data)
    folds = make_folds(recs)
    print("folds:", folds)
    ks = range(len(folds)) if args.fold == "all" else [int(args.fold)]

    out = Path("results/runs.jsonl")
    out.parent.mkdir(exist_ok=True)
    for seg in args.seg:
        c = cost(seg)
        for k in ks:
            res = run(recs, folds, k, seg, args, device) | c
            with out.open("a") as f:
                f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
