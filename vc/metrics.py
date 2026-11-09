"""Effect metrics (accuracy, F1, confusion) and cost metrics (params, MACs, latency)."""
import time

import numpy as np
import torch
import torch.nn as nn


def scores(y_true, y_pred, n_classes: int = 2) -> dict:
    cm = np.zeros((n_classes, n_classes), dtype=int)  # rows = truth, cols = prediction
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    tp = np.diag(cm)
    recall = tp / np.maximum(cm.sum(1), 1)
    precision = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(accuracy=float(tp.sum() / cm.sum()),
                balanced_accuracy=float(recall.mean()),
                macro_f1=float(f1.mean()),
                confusion=cm.tolist())


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def n_macs(model: nn.Module, x: torch.Tensor) -> int:
    """Multiply-accumulates of Conv2d + Linear layers for ONE example."""
    total = 0

    def conv(m, _, out):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
        total += out[0].numel() * k

    def linear(m, _, out):
        nonlocal total
        total += m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv if isinstance(m, nn.Conv2d) else linear)
             for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model(x[:1])
    for h in hooks:
        h.remove()
    return total


def latency_ms(fn, n_warmup: int = 50, n_runs: int = 500) -> dict:
    """Wall-clock latency of fn(): warm up first, then report median and p95."""
    for _ in range(n_warmup):
        fn()
    t = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return dict(median=float(np.median(t)), p95=float(np.percentile(t, 95)))
