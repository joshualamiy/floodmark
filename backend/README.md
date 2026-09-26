# Floodmark Pipeline

The pipeline schedules active cameras every five minutes, queues complete captures in Redis/ARQ, stores normalized JPEGs in S3-compatible object storage, and writes image and prediction records directly to PostgreSQL.

## Run

1. Copy `.env.example` to `.env` and provide the PostgreSQL and S3-compatible storage credentials.
2. Start the scheduler, worker, and Redis with `docker compose up --build` from this directory. The Compose build context is the repository root so the private model bundle and `flood-ml/src` are included.
3. Scale workers with `docker compose up --scale worker=2` after benchmarking the configured downstream limits.

`DATABASE_URL` must point at the existing schema used by the website. The worker database credential needs read access to `cameras` and write access to `images` and `predictions` only.

Set `SCHEDULER_RUN_AT_STARTUP=true` in `.env` to enqueue one capture cycle as soon as the scheduler starts. Leave it `false` in normal operation.

Set `DEBUG=true` to log capture start/completion timing. Failures are always logged.

## Behavior

- Scheduler job IDs are deterministic per source camera and five-minute UTC slot, so scheduler overlap cannot enqueue a duplicate capture.
- The scheduler consumes `arq:scheduler`; capture workers consume `arq:queue`. Scheduler-created capture jobs are explicitly routed to the capture queue.
- Frame and heatmap keys use the source view ID and scheduled UTC slot: `captures/{view_id}/YYYYMMDDTHHmmZ.jpg` and `heatmaps/{view_id}/YYYYMMDDTHHmmZ.png`.
- Downloads stream into an in-memory byte budget and reject unsupported content types or oversized responses.
- The worker loads the ONNX sessions once per process from `FLOODML_MODEL_DIR` and runs the real flood classifier on each normalized JPEG.
- Raw model status is stored in `predictions.status`; the alert status in `predictions.alert_status` is decided per camera from its own database history (`alerting.py`), so it is shared between workers and survives restarts. In order:
  1. Cameras in `ALERT_BLOCKLIST` never alert. The default list holds the cameras the ML evaluation and live daytime frames found calling dry scenes flooded (`11372`, `17397`, `13750`, `13417`, `17356`, `13536`, `14256`).
  2. A frame byte-identical to the camera's previous capture (a frozen feed) is not counted toward confirmation and does not reset it.
  3. A water frame counts only when its flood score (`stage_probabilities.flooded`) is at least `ALERT_BASELINE_MARGIN` above the camera's own median over the last `ALERT_BASELINE_DAYS` days (once `ALERT_BASELINE_MIN_FRAMES` exist). A static scene that always looks a little like water cannot confirm itself; a camera that scores near 1.0 all day effectively can never alert, which is intended.
  4. `ALERT_STREAK_FRAMES` counted water frames in a row are required before `flooded`. In storm mode (at least `ALERT_MIN_RAIN_MM` of rain at the camera in the last `ALERT_RAIN_WINDOW_HOURS` hours) only `ALERT_STORM_STREAK_FRAMES` are needed.
  5. When `ALERT_REQUIRE_RAIN` is true, a confirmed flood is downgraded to `wet` unless it has rained as above. Rain comes from Open-Meteo, is looked up only for water frames, is cached per 0.1° grid cell for 15 minutes, and a failed lookup lets the alert through. Rain far upstream of a camera is not seen, so this is a filter for dry-day false alarms, not a flood model.
- While a camera's confirmation is in progress (a water frame below the streak, or a frozen repeat), the worker re-captures it after `ALERT_FAST_POLL_SECONDS` instead of waiting for the next five-minute slot, up to `ALERT_FAST_POLL_MAX` times per chain. Fast re-polls get their own job IDs and second-resolution object keys (`captures/{view_id}/YYYYMMDDTHHmmssZ.jpg`), so they never overwrite a slot capture. A confirmed flood in a storm therefore shows in about two minutes, while a dry-day glitch still cannot get past `wet`. Confirmation can never be faster than 511GA refreshes the snapshot: an unchanged snapshot is skipped, not counted.
- Every suppressed `flooded` call is logged as `flood call suppressed camera_id=... reason=...`; use those lines to extend the blocklist.
- `wet` means possible flooding below the model's flood alert threshold. Heatmap keys are stored alongside predictions so weak overlays are not presented as evidence.
- A job runs at most three times. ARQ retains its final result for `ARQ_RESULT_TTL_SECONDS` (15 minutes by default), and a database image row is marked `error` with its final message when one exists.

No image bytes are written to a local volume. An S3 object can exist without an image row if PostgreSQL fails after upload; reconciliation is intentionally deferred beyond this MVP.

## Recovering From An Older Queue Layout

If a prior deployment put `cron:schedule_captures` jobs on the capture queue, remove the stale job before restarting. Substitute the job ID from the worker error:

```sh
docker compose exec redis redis-cli ZREM arq:queue cron:schedule_captures:1790403319725
docker compose exec redis redis-cli DEL arq:result:cron:schedule_captures:1790403319725
```

For a disposable local stack, `docker compose down -v` clears the Redis queue entirely.
