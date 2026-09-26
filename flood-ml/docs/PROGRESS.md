# Progress log

## 2026-09-25: Phase 0 (recon)

- **Hardware:** Apple M5 Max (18 CPU cores, 32-core GPU), 36 GB RAM, 1.5 TiB free disk.
- **Python:** `my_env/bin/python` is 3.12.14, which TensorFlow supports. The venv mixes 3.12 and 3.14: `pip` targets 3.14, so we always use `my_env/bin/python -m pip`.
- **Packages installed:** TF 2.21.0, keras 3.15.1, tensorboard, tf2onnx 1.17.0, onnx 1.23.0, onnxruntime 1.30.0, and the Phase 1 data packages (pandas, pillow, imagehash, folium, opencv-headless, scikit-learn, matplotlib, pytest, ruff). numpy is pinned to 2.4.6, because 2.5.x drops Python 3.11, which CI uses.
- **GPU:** `tensorflow-metal` 1.2.0 breaks `import tensorflow` on 2.21, so I uninstalled it. Training is CPU only.
- **CPU speed:** MobileNetV3Small about 280 img/s, EfficientNetB0 about 73 img/s (full fine-tune at 224 px).
- **Git:**
  - Pointed `origin` at https://github.com/joshualamiy/floodmark (public).
  - Remote `main` was rewritten (`62615c9`) and no longer shares history with local `ml`/`main` (`57fd474`).
  - Re-basing `ml` onto `origin/main` was blocked by a permission check and waits for the user's OK.
- **Data inventory:** an earlier session already downloaded Roadway Flooding, Flood Area Segmentation, Water/V-FloodNet, FRED (front camera), TinyCamML, and a local Flood Master copy. See `docs/PLAN.md` §1.
- **Files created:**
  - the `flood-ml/` layout and `flood-ml/.gitignore` (image-bearing reports stay local)
  - `pyproject.toml` (ruff and pytest config)
  - `requirements.txt` (inference runtime) and `requirements-train.txt`
  - `docs/PLAN.md` with all subagent briefs
