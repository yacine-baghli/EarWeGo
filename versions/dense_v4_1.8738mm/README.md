# Dense V4 — 1.8738 mm validation mean error

This folder is a source snapshot of the best validated classical model from
commit `21bdc53`. It contains the implementation, configuration, dependency
list, and aggregate validation metrics.

Challenge data, participant splits, trained weights, host metadata, and
per-subject results are intentionally excluded.

## Result

- Validation subjects: 30
- Mean distance: **1.8738 mm**
- Median distance: 1.7177 mm
- P90: 2.5800 mm
- Success rate at 2 mm: 65.5%
- Success rate at 3 mm: 84.1%
- Success rate at 5 mm: 95.9%

The aggregate record is in [`validation_metrics.json`](validation_metrics.json).

## Reproduce training

Provide the challenge dataset privately at the path configured in
`configs/base.yaml`, then run from this folder:

```powershell
python train.py --config configs/v4_dense.yaml
```

Training creates model artifacts locally. The folder-level `.gitignore`
prevents data, split files, model weights, and detailed run records from being
committed.
