"""Replay server for demo cameras: each request returns the next frame of a folder.

Frames live in ``DEMO_FRAMES_DIR/<view_id>/`` and play in filename order, so
name them ``01_dry.jpg``, ``02_dry.jpg``, ``03_flood.jpg``, ... The last frame
repeats once the sequence is over. ``GET /reset/<view_id>`` starts it over and
``GET /status`` shows where each camera is.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from aiohttp import web

logger = logging.getLogger(__name__)

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def list_frames(frames_dir: Path, view_id: str) -> list[Path]:
    folder = frames_dir / view_id
    if not folder.is_dir():
        return []
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_TYPES)


class Replay:
    def __init__(self, frames_dir: Path) -> None:
        self.frames_dir = frames_dir
        self._positions: dict[str, int] = {}

    def next_frame(self, view_id: str) -> Path | None:
        frames = list_frames(self.frames_dir, view_id)
        if not frames:
            return None
        position = self._positions.get(view_id, 0)
        frame = frames[min(position, len(frames) - 1)]
        self._positions[view_id] = min(position + 1, len(frames))
        return frame

    def reset(self, view_id: str) -> None:
        self._positions[view_id] = 0

    def status(self) -> dict[str, dict[str, int]]:
        return {
            view_id: {"served": position, "frames": len(list_frames(self.frames_dir, view_id))}
            for view_id, position in self._positions.items()
        }


async def serve_frame(request: web.Request) -> web.Response:
    replay: Replay = request.app["replay"]
    frame = replay.next_frame(request.match_info["view_id"])
    if frame is None:
        raise web.HTTPNotFound(text="no frames for this view id")
    logger.info("served %s", frame)
    return web.Response(body=frame.read_bytes(), content_type=IMAGE_TYPES[frame.suffix.lower()])


async def reset(request: web.Request) -> web.Response:
    request.app["replay"].reset(request.match_info["view_id"])
    return web.json_response({"reset": request.match_info["view_id"]})


async def status(request: web.Request) -> web.Response:
    return web.json_response(request.app["replay"].status())


def build_app(frames_dir: Path) -> web.Application:
    app = web.Application()
    app["replay"] = Replay(frames_dir)
    app.add_routes([web.get("/status", status), web.get("/reset/{view_id}", reset), web.get("/{view_id}", serve_frame)])
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    frames_dir = Path(os.getenv("DEMO_FRAMES_DIR", "demo_frames"))
    port = int(os.getenv("DEMO_REPLAY_PORT", "8080"))
    logger.info("serving demo frames from %s on port %d", frames_dir.resolve(), port)
    web.run_app(build_app(frames_dir), port=port)


if __name__ == "__main__":
    main()
