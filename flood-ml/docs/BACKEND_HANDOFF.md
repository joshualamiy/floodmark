# Backend handoff: plugging in the real model

## 1. Get the code and models

- **Code:** `flood-ml/src/inference/` on the `ml` branch (PR #1). It needs only `onnxruntime==1.30.0`, `numpy==2.4.6`, and `pillow==12.3.0`, as listed in `flood-ml/requirements.txt`.
- **Models:** `floodmark_models_v3.zip` (11 MB). Get it from the ML side through a private channel. It's **not** in git and must not go in git or anywhere public: the training data licenses (FRED NC-SA, Flood Master non-commercial, CC BY-SA images) don't allow publishing the weights.

  ```
  stage_a.onnx  stage_b.onnx  config.json    # flood classifier (required)
  water/water.onnx  water/config.json        # optional water overlay
  ```

  Unzip it anywhere and point `FLOODML_MODEL_DIR` at that folder.

## 2. Docker (worker image)

```dockerfile
COPY flood-ml/src /app/floodml_src
COPY models /models
ENV PYTHONPATH=/app/floodml_src FLOODML_MODEL_DIR=/models
RUN pip install onnxruntime==1.30.0 numpy==2.4.6 pillow==12.3.0
```

The build context must include `flood-ml/`.

## 3. Replace the stub in `backend/floodmark_pipeline/inference.py`

```python
from inference import load_models, predict

_MODELS = load_models()  # once per worker process

def model_run(image_bytes: bytes) -> Prediction:
    p = predict(image_bytes, _MODELS)  # raw jpeg bytes are fine
    return Prediction(
        model_version=p.model_version,          # same keys as the stub
        status=p.status,
        confidence=p.confidence,
        stage_a_probabilities=p.stage_a_probs,
        stage_b_probabilities=p.stage_b_probs,
        stage_probabilities=p.stage_probabilities,
        thresholds=p.thresholds,
        note=p.note,
        heatmap_bytes=p.heatmap_png,
    )
```

Notes:
- Calling it from `asyncio.to_thread` is safe.
- It takes about 5 to 20 ms per frame on CPU, including preprocessing.
- All resizing and preprocessing happens inside `predict()`. Pass the raw frame.

## 4. Alerts: smooth per camera

```python
from inference import TemporalSmoother

smoother = TemporalSmoother(n=3, blocklist={"11372", "17397", "13750"})  # persist with to_dict()/from_dict()
alert_status = smoother.update(camera_id, p).status
```

- Store the raw `status` for every frame, but only show "flooded" on the map once `alert_status` says so, i.e. 3 flooded frames in a row.
- Cameras 11372 (grass), 17397 (bridge pier), and 13750 (gated booth) repeat false alarms, so they are blocklisted.

## 5. Show honestly on the website

- `"wet"` means **possible flooding, below the alert threshold**, not "wet pavement". Show the `note` text.
- `heatmap_status` tells you whether the overlay is meaningful: `shown`, `no_strong_evidence`, and so on. `heatmap_note` explains it.
- A "dry" result doesn't prove a road is safe. Known limits are in `reports/EVALUATION.md`.

## 6. Tests

`backend/tests/test_inference.py` should use the real models and `pytest.skip()` when `FLOODML_MODEL_DIR` isn't set, because CI has no model files.

## Swapping models later

A new model means a new zip with the same three files. No code changes: `config.json` carries the thresholds and input size. The `model_version` output changes on its own, so rows in the `predictions` table stay tied to the right model.
