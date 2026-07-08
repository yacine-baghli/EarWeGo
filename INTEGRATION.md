# Wiring the refinements into `src/predictor.py`

Put `refinement.py` in `src/`. Then make three edits to `LandmarkPredictor.predict()`
(the method around lines 170–266). Everything is behind flags so the A/B harness can
toggle each one independently.

## 1. Add a `refine` argument to the method

```python
def predict(self, mesh=None, side="left", known_landmarks=None, ear_detector=None,
            refine=None):
    refine = refine or {}                       # e.g. {"clamp": True, "resample": True, "selective_snap": True}
    from src.refinement import clamp_scale, resample_contours, selective_snap
```

## 2. Clamp the scale (Step 2) — right after `procrustes_align`

Find:
```python
    aligned, transform = procrustes_align(
        initial_for_ssm, self.ssm.get_mean_shape(), allow_scale=True
    )
```
Add immediately after:
```python
    if refine.get("clamp"):
        # bounds can come from config.model.scale_clamp = [lo, hi]
        transform = clamp_scale(transform, *refine.get("scale_bounds", (0.92, 1.08)))
    coeff = self.ssm.project(aligned)
```
(Note: clamp the transform used for the inverse in Step 4; `aligned`/`coeff` still use the
projected shape, which is what you want — the clamp only affects how you map back to mesh space.)

## 3. Replace Step 6 (snapping) with resample + selective snap

Find the current Step 6:
```python
    # Step 6: Proximity query snap to surface
    if mesh is not None:
        try:
            closest, _, _ = mesh.nearest.on_surface(result)
            result = closest
        except Exception:
            pass
    return result
```
Replace with:
```python
    # Step 6a: enforce equal-arc-length spacing (fixes the 63 non-anchor landmarks)
    if refine.get("resample"):
        result = resample_contours(result)

    # Step 6b: snap to surface — selective by default, legacy (all points) as fallback
    if mesh is not None:
        if refine.get("selective_snap"):
            result = selective_snap(result, mesh)
        elif refine.get("legacy_snap", True):
            try:
                closest, _, _ = mesh.nearest.on_surface(result)
                result = closest
            except Exception:
                pass
    return result
```

## Config-driven version (recommended, ties into the run system)

Add to `configs/base.yaml`:
```yaml
model:
  refine:
    clamp: false
    scale_bounds: [0.92, 1.08]
    resample: false
    selective_snap: false
    legacy_snap: true
```
Have `predict()` default `refine` from `config["model"]["refine"]`, then create
`configs/v2_refined.yaml` overriding those to `true`. Each combination becomes a versioned
run you can compare on the board.

## Order matters

`clamp → SSM reconstruct → inverse transform → (KNN blend) → resample → selective snap`.
Resample must run **before** snapping so snapping doesn't re-introduce uneven spacing.
`selective_snap` and `legacy_snap` are mutually exclusive (selective wins).
