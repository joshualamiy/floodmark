# clip zero-shot 'is this a road' filter
from __future__ import annotations

import torch
from PIL import Image

POSITIVE_PROMPTS = [
    "a photo of a road",
    "a street",
    "a highway seen from a traffic camera",
    "a flooded street",
    "a car driving on a road",
    "a road with flood water on it",
]
NEGATIVE_PROMPTS = [
    "an aerial photo",
    "a satellite image",
    "a river",
    "a lake",
    "an indoor room",
    "a field",
    "a beach",
]

_MODEL_CACHE: dict[str, tuple] = {}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    key = f"{model_name}:{pretrained}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    import open_clip

    device = _device()
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    with torch.no_grad():
        pos_tok = tokenizer(POSITIVE_PROMPTS).to(device)
        neg_tok = tokenizer(NEGATIVE_PROMPTS).to(device)
        pos_feat = model.encode_text(pos_tok)
        neg_feat = model.encode_text(neg_tok)
        pos_feat = pos_feat / pos_feat.norm(dim=-1, keepdim=True)
        neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)

    bundle = (model, preprocess, device, pos_feat, neg_feat)
    _MODEL_CACHE[key] = bundle
    return bundle


def score_batch(image_paths: list[str], bundle=None, batch_size: int = 64) -> list[tuple[bool, float, float]]:
    if bundle is None:
        bundle = load_clip()
    model, preprocess, device, pos_feat, neg_feat = bundle

    results: list[tuple[bool, float, float]] = [None] * len(image_paths)  # type: ignore
    batch_imgs = []
    batch_idx = []

    def flush():
        if not batch_imgs:
            return
        with torch.no_grad():
            tensor = torch.stack(batch_imgs).to(device)
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            pos_sims = (feat @ pos_feat.T).max(dim=-1).values
            neg_sims = (feat @ neg_feat.T).max(dim=-1).values
            for j, i in enumerate(batch_idx):
                p = float(pos_sims[j].item())
                n = float(neg_sims[j].item())
                results[i] = (p > n, p, n)
        batch_imgs.clear()
        batch_idx.clear()

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(preprocess(img))
            batch_idx.append(i)
        except Exception:  # noqa: BLE001
            results[i] = (False, 0.0, 0.0)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()

    return results  # type: ignore

