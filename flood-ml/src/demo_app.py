"""Local-only demo: upload a frame, see status + stage probs + heatmap.
Run: PYTHONPATH=src ../my_env/bin/python src/demo_app.py -> http://127.0.0.1:7860
"""
from __future__ import annotations

import io

import gradio as gr  # only imported here, never from src/inference
from PIL import Image

from inference import predict

LIMITS = (
    "**Known limits** (reports/EVALUATION.md): flood recall is 0.64 overall and drops to "
    "0.29 on elevated fixed-camera views; small or distant floods (<20% of the road) are "
    "often missed; plain wet pavement is never detected (\"wet\" here means a flood score "
    "below the alert threshold, not standing water on dry-looking road); and odd, non-road "
    "camera views can cause false alarms."
)


def run(image):
    if image is None:
        return "-", "-", {}, None
    pred = predict(image, heatmap=True)
    status = pred.status if not pred.note else f"{pred.status} ({pred.note})"
    heatmap = None
    if pred.heatmap_png is not None:
        heatmap = Image.open(io.BytesIO(pred.heatmap_png))
    return status, f"{pred.confidence:.3f}", pred.stage_probabilities, heatmap


def build_app():
    with gr.Blocks(title="floodmark demo") as demo:
        gr.Markdown("# floodmark: flood-frame classifier\nTwo-stage: dry/wet, then flooded/not.")
        with gr.Row():
            inp = gr.Image(type="pil", label="Camera frame")
            out_heatmap = gr.Image(type="pil", label="Heatmap (Stage B, where the model sees water)")
        with gr.Row():
            out_status = gr.Textbox(label="Status")
            out_conf = gr.Textbox(label="Confidence")
        out_probs = gr.Label(label="Stage probabilities (dry / wet / flooded)")
        gr.Markdown(LIMITS)
        inp.change(run, inputs=inp, outputs=[out_status, out_conf, out_probs, out_heatmap])
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7860, share=False)
