"""Fast checks that need no dataset: run with `python -m pytest -q`."""
import numpy as np
import torch

from vc.data import FPS, N_MELS, SR, logmel, make_folds, segments, split
from vc.metrics import n_macs, scores
from vc.model import VocalCNN


def fake_recs():
    """6 pairs of singers; each pair shares a song set, like NUS-48E."""
    recs = []
    for p in range(6):
        for spk in (f"M{p}", f"F{p}"):
            for song in (f"{p}a", f"{p}b"):
                for label in (0, 1):
                    recs.append(dict(speaker=spk, song=song, label=label,
                                     feat=np.zeros((N_MELS, 500), np.float16),
                                     active=np.ones(500, bool)))
    return recs


def test_folds_are_singer_pairs():
    folds = make_folds(fake_recs())
    assert len(folds) == 6 and all(len(f) == 2 for f in folds)


def test_no_singer_leaks_between_train_val_test():
    folds = make_folds(fake_recs())
    tested = []
    for k in range(len(folds)):
        tr, va, te = split(folds, k)
        assert not set(tr) & set(va) and not set(tr) & set(te) and not set(va) & set(te)
        assert len(tr) + len(va) + len(te) == 12
        tested += te
    assert sorted(tested) == sorted({s for f in folds for s in f})  # each singer tested exactly once


def test_segments_shape_and_silence_filter():
    rec = fake_recs()[0]
    rec["active"][:260] = False  # first 2.6 s silent
    X, y, meta = segments([rec], seg_sec=1.0)
    assert X.shape == (2, N_MELS, FPS)          # only the 2 non-silent seconds survive
    assert [s for _, s in meta] == [300, 400]


def test_logmel_frame_rate():
    assert logmel(np.zeros(SR, np.float32)).shape == (N_MELS, FPS + 1)  # center=True adds 1 frame


def test_model_any_length_and_loudness_invariant():
    model = VocalCNN().eval()
    for T in (25, 100, 300):
        x = torch.randn(3, N_MELS, T)
        out = model(x)
        assert out.shape == (3, 2)
        assert torch.allclose(out, model(x + 20.0), atol=1e-4)  # +20 dB louder -> same answer


def test_scores():
    s = scores([0, 0, 1, 1], [0, 1, 1, 1])
    assert s["accuracy"] == 0.75 and s["confusion"] == [[1, 1], [0, 2]]


def test_macs_linear():
    assert n_macs(torch.nn.Linear(10, 3), torch.zeros(1, 10)) == 30
