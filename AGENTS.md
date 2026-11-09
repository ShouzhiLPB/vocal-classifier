# Rules for anyone (human or AI) changing this repo

1. **No singer may appear in two of train / val / test.** `tests/test_core.py` guards this; keep it green.
2. The test set is used **once**, after training. Model choices are made on validation only.
3. Every model is scored on the **same segments** (`segments()` meta) — compare like with like.
4. Report mean ± std over all folds, never the best fold.
5. Keep files small and single-purpose: `vc/data.py` data, `vc/model.py` model, `vc/metrics.py` measures.
