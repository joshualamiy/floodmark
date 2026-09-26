"""A tiny local labeling tool for 511GA frames: stdlib `http.server` plus one
embedded HTML page, no new heavy dependencies.

Launch (serves on 127.0.0.1:8765 by default):

    cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.label_tool
    cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.label_tool --port 8000 --ga511-root data/ga511

Then open http://127.0.0.1:8765/ in a browser. Filters: unlabeled, likely_wet,
by camera, by split (from `data/processed/ga511_camera_splits.json` if it
exists). Keyboard: 1=dry, 2=wet, 3=flooded, 0=unusable, left/right arrows to
navigate. Every keypress POSTs to `/api/label` and appends one row to
`data/ga511/labels.csv` (`frame_id, label, labeled_at`); if a frame is
labeled twice, the last row wins (readers of labels.csv reduce that way, see
`prep.sources.load_labels_csv`). This file is only ever written by this tool
and the user manually -- `prep.sources` treats it as `label_source=manual`
and it always overrides `ai_review`.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

VALID_LABELS = {"dry", "wet", "flooded", "unusable"}

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>511GA frame labeling</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
  #frame { max-width: 90vw; max-height: 60vh; display: block; margin: 0 auto; border: 1px solid #444; }
  .bar { display: flex; gap: 12px; align-items: center; justify-content: center; margin: 10px 0; flex-wrap: wrap; }
  .meta { text-align: center; color: #aaa; font-size: 14px; }
  select, button { font-size: 14px; padding: 4px 8px; }
  .label-btn { padding: 8px 14px; cursor: pointer; }
  .label-btn.dry { background: #8c8c8c; }
  .label-btn.wet { background: #4C9BE8; }
  .label-btn.flooded { background: #D64545; color: white; }
  .label-btn.unusable { background: #444; color: #ccc; }
  #status { text-align: center; color: #6f6; height: 20px; }
  #counter { text-align: center; color: #999; }
</style>
</head>
<body>
  <div class="bar">
    <label>Filter:
      <select id="filter">
        <option value="unlabeled">unlabeled</option>
        <option value="likely_wet">likely_wet</option>
        <option value="all">all</option>
      </select>
    </label>
    <label>Camera: <input id="camera" size="8" placeholder="camera_id"></label>
    <label>Split:
      <select id="split">
        <option value="">any</option>
        <option value="train">train</option>
        <option value="val">val</option>
        <option value="test">test</option>
      </select>
    </label>
    <button id="reload">reload</button>
  </div>
  <div id="counter"></div>
  <img id="frame" src="">
  <div class="meta" id="meta"></div>
  <div class="bar">
    <button class="label-btn dry" data-label="dry">1 dry</button>
    <button class="label-btn wet" data-label="wet">2 wet</button>
    <button class="label-btn flooded" data-label="flooded">3 flooded</button>
    <button class="label-btn unusable" data-label="unusable">0 unusable</button>
    <button id="prev">&larr; prev</button>
    <button id="next">next &rarr;</button>
  </div>
  <div id="status"></div>

<script>
let frames = [];
let i = 0;

function qs() {
  const f = document.getElementById('filter').value;
  const cam = document.getElementById('camera').value.trim();
  const split = document.getElementById('split').value;
  const p = new URLSearchParams({filter: f});
  if (cam) p.set('camera', cam);
  if (split) p.set('split', split);
  return p.toString();
}

async function load() {
  const res = await fetch('/api/frames?' + qs());
  frames = await res.json();
  i = 0;
  render();
}

function render() {
  const c = document.getElementById('counter');
  if (frames.length === 0) {
    c.textContent = 'no frames match this filter';
    document.getElementById('frame').src = '';
    document.getElementById('meta').textContent = '';
    return;
  }
  if (i < 0) i = 0;
  if (i >= frames.length) i = frames.length - 1;
  const fr = frames[i];
  c.textContent = `${i + 1} / ${frames.length}`;
  document.getElementById('frame').src = '/images/' + encodeURIComponent(fr.path);
  document.getElementById('meta').textContent =
    `frame_id=${fr.frame_id} camera=${fr.camera_id} weak_label=${fr.weak_label || '-'} ` +
    `precip_1h=${fr.precip_1h_mm ?? '-'} precip_3h=${fr.precip_3h_mm ?? '-'} ` +
    `manual=${fr.manual_label || '-'} ai_review=${fr.ai_review_label || '-'} split=${fr.split || '-'}`;
}

async function label(lab) {
  if (frames.length === 0) return;
  const fr = frames[i];
  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({frame_id: fr.frame_id, label: lab}),
  });
  document.getElementById('status').textContent = `labeled ${fr.frame_id} as ${lab}`;
  setTimeout(() => document.getElementById('status').textContent = '', 1200);
  fr.manual_label = lab;
  i += 1;
  render();
}

document.getElementById('reload').onclick = load;
document.getElementById('filter').onchange = load;
document.getElementById('split').onchange = load;
document.getElementById('camera').onchange = load;
document.getElementById('prev').onclick = () => { i -= 1; render(); };
document.getElementById('next').onclick = () => { i += 1; render(); };
document.querySelectorAll('.label-btn').forEach(b => {
  b.onclick = () => label(b.dataset.label);
});
document.addEventListener('keydown', (e) => {
  if (e.key === '1') label('dry');
  else if (e.key === '2') label('wet');
  else if (e.key === '3') label('flooded');
  else if (e.key === '0') label('unusable');
  else if (e.key === 'ArrowLeft') { i -= 1; render(); }
  else if (e.key === 'ArrowRight') { i += 1; render(); }
});
load();
</script>
</body>
</html>
"""


def _read_csv_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _last_write_wins(rows: list[dict], key: str = "frame_id") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        k = r.get(key)
        if k:
            out[k] = r
    return out


class FrameStore:
    """Reads frames.csv / labels.csv / ai_review_labels.csv / the camera
    splits file fresh on every call, so the tool always reflects the latest
    state written by the concurrently running collector, the user's own
    earlier labels, and this process's own appends.
    """

    def __init__(self, ga511_root: Path):
        self.ga511_root = Path(ga511_root)

    def _frames_csv(self) -> Path:
        return self.ga511_root / "frames.csv"

    def _labels_csv(self) -> Path:
        return self.ga511_root / "labels.csv"

    def _ai_review_csv(self) -> Path:
        return self.ga511_root / "ai_review_labels.csv"

    def _splits_json(self) -> Path:
        return self.ga511_root.parent / "processed" / "ga511_camera_splits.json"

    def list_frames(self, filt: str, camera: str | None, split: str | None) -> list[dict]:
        raw_rows = _read_csv_dicts(self._frames_csv())
        manual = _last_write_wins(_read_csv_dicts(self._labels_csv()))
        ai_review = _last_write_wins(_read_csv_dicts(self._ai_review_csv()))
        splits = {}
        if self._splits_json().exists():
            try:
                splits = json.loads(self._splits_json().read_text())
            except Exception:  # noqa: BLE001
                splits = {}

        out = []
        for r in raw_rows:
            if not r.get("frame_id") or not r.get("camera_id"):
                continue
            if r.get("dead_reason"):
                continue
            path = r.get("path") or ""
            if not path:
                continue
            if not (self.ga511_root / path).exists():
                continue

            frame_id = r["frame_id"]
            cam_id = r["camera_id"]
            manual_label = manual.get(frame_id, {}).get("label")
            ai_label = ai_review.get(frame_id, {}).get("label")
            frame_split = splits.get(cam_id)

            if camera and cam_id != camera:
                continue
            if split and frame_split != split:
                continue
            if filt == "unlabeled" and manual_label:
                continue
            if filt == "likely_wet" and r.get("weak_label") != "likely_wet":
                continue

            out.append({
                "frame_id": frame_id,
                "path": path,
                "camera_id": cam_id,
                "weak_label": r.get("weak_label") or None,
                "precip_1h_mm": r.get("precip_1h_mm") or None,
                "precip_3h_mm": r.get("precip_3h_mm") or None,
                "manual_label": manual_label,
                "ai_review_label": ai_label,
                "split": frame_split,
            })
        return out

    def append_label(self, frame_id: str, label: str) -> None:
        if label not in VALID_LABELS:
            raise ValueError(f"invalid label {label!r}")
        path = self._labels_csv()
        is_new = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["frame_id", "label", "labeled_at"])
            w.writerow([frame_id, label, datetime.now(timezone.utc).isoformat()])


def make_handler(store: FrameStore, ga511_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, obj, status=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/frames":
                qs = parse_qs(parsed.query)
                filt = qs.get("filter", ["unlabeled"])[0]
                camera = qs.get("camera", [None])[0]
                split = qs.get("split", [None])[0]
                frames = store.list_frames(filt, camera, split)
                self._send_json(frames)
                return
            if parsed.path.startswith("/images/"):
                from urllib.parse import unquote
                rel = unquote(parsed.path[len("/images/"):])
                img_path = (ga511_root / rel).resolve()
                try:
                    img_path.relative_to(ga511_root.resolve())
                except ValueError:
                    self.send_response(403)
                    self.end_headers()
                    return
                if not img_path.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                data = img_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/label":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
                store.append_label(payload["frame_id"], payload["label"])
            except Exception as e:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(e)}, status=400)
                return
            self._send_json({"ok": True})

    return Handler


def create_server(ga511_root: Path, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    store = FrameStore(Path(ga511_root))
    handler = make_handler(store, Path(ga511_root))
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description="511GA frame labeling tool")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ga511-root", default="data/ga511")
    args = ap.parse_args()

    server = create_server(Path(args.ga511_root), host=args.host, port=args.port)
    print(f"511GA labeling tool at http://{args.host}:{server.server_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
