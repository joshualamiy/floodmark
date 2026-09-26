"""Local upload demo."""
from __future__ import annotations

import io

import gradio as gr
from PIL import Image

from inference import predict

LIMITS = (
    "**Known limits** (reports/EVALUATION.md): on held-out test data flood recall is 0.64 "
    "overall and 0.29 on an elevated fixed-camera flood video; small or distant floods are "
    "often missed; plain wet pavement was never detected (0/9). A dry prediction does not "
    "prove a road is safe. Flood-score attribution explains the classifier; the water "
    "overlay comes from a separate model and does not establish road flooding."
)


def _image(data):
    if data is None:
        return None
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def run(image, raw_debug=False, show_water=False):
    if image is None:
        return "-", "-", {}, None, "", None, None, ""
    pred = predict(image, heatmap=True, raw_heatmap=raw_debug)
    status = pred.status if not pred.note else f"{pred.status} ({pred.note})"
    water_image, water_note = None, "Water overlay is off."
    if show_water:
        from inference.water import load_water_model, predict_water

        model = load_water_model()
        if model is None:
            water_note = "Water overlay is unavailable; the classifier still works."
        else:
            water = predict_water(image, model=model)
            layer = _image(water["water_overlay_png"])
            water_image = Image.alpha_composite(image.convert("RGBA"), layer.convert("RGBA"))
            water_note = water["note"]
    return (status, f"{pred.confidence:.3f}", pred.stage_probabilities,
            _image(pred.heatmap_png), pred.heatmap_note or "", _image(pred.raw_heatmap_png),
            water_image, water_note)


def build_app():
    with gr.Blocks(title="floodmark demo") as demo:
        gr.Markdown("# floodmark")
        gr.Markdown("Upload a camera frame to inspect the flood prediction.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Camera frame")
            out_heatmap = gr.Image(type="pil", label="Evidence supporting the flood score")
        out_heatnote = gr.Markdown()
        gr.Markdown("Orange shows relative contribution. Color intensity is not a pixel probability or water depth.")
        with gr.Row():
            out_status = gr.Textbox(label="Status")
            out_conf = gr.Textbox(label="Model confidence")
        out_probs = gr.Label(label="Model probabilities (dry / wet / flooded)")
        show_water = gr.Checkbox(label="Show experimental predicted water overlay", value=False)
        out_water = gr.Image(type="pil", label="Predicted water — separate from the classifier explanation")
        out_waternote = gr.Markdown()
        with gr.Accordion("Inspect raw classifier evidence", open=False):
            raw_debug = gr.Checkbox(label="Show raw attribution, including low-confidence predictions", value=False)
            gr.Markdown("The raw view is scaled within each image. Bright areas can appear even when flooding is unlikely.")
            out_raw = gr.Image(type="pil", label="Raw Stage B attribution (debugging)")
        gr.Markdown(LIMITS)
        inputs = [inp, raw_debug, show_water]
        outputs = [out_status, out_conf, out_probs, out_heatmap, out_heatnote,
                   out_raw, out_water, out_waternote]
        for control in inputs:
            control.change(run, inputs=inputs, outputs=outputs)
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=False)
