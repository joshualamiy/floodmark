# Floodmark Pipeline

The pipeline schedules active cameras every five minutes, queues complete captures in Redis/ARQ, stores normalized JPEGs in Cloudflare R2, and writes image and prediction records directly to PostgreSQL.

## Run

1. Copy `.env.example` to `.env` and provide the PostgreSQL and R2 credentials.
2. Start the scheduler, worker, and Redis with `docker compose up --build` from this directory.
3. Scale workers with `docker compose up --scale worker=2` after benchmarking the configured downstream limits.

`DATABASE_URL` must point at the existing schema used by the website. The worker database credential needs read access to `cameras` and write access to `images` and `predictions` only.

Set `SCHEDULER_RUN_AT_STARTUP=true` in `.env` to enqueue one capture cycle as soon as the scheduler starts. Leave it `false` in normal operation.

Set `DEBUG=true` to log capture start/completion timing. Failures are always logged.

## Behavior

- Scheduler job IDs are deterministic per source camera and five-minute UTC slot, so scheduler overlap cannot enqueue a duplicate capture.
- The scheduler consumes `arq:scheduler`; capture workers consume `arq:queue`. Scheduler-created capture jobs are explicitly routed to the capture queue.
- Frame and heatmap keys use the source view ID and scheduled UTC slot: `captures/{view_id}/YYYYMMDDTHHmmZ.jpg` and `heatmaps/{view_id}/YYYYMMDDTHHmmZ.png`.
- Downloads stream into an in-memory byte budget and reject unsupported content types or oversized responses.
- The MVP `model_run()` is deterministic from normalized image bytes. Replace it with ONNX session inference without changing the prediction persistence contract.
- A job runs at most three times. ARQ retains its final result for `ARQ_RESULT_TTL_SECONDS` (15 minutes by default), and a database image row is marked `error` with its final message when one exists.

No image bytes are written to a local volume. An R2 object can exist without an image row if PostgreSQL fails after upload; reconciliation is intentionally deferred beyond this MVP.

## Recovering From An Older Queue Layout

If a prior deployment put `cron:schedule_captures` jobs on the capture queue, remove the stale job before restarting. Substitute the job ID from the worker error:

```sh
docker compose exec redis redis-cli ZREM arq:queue cron:schedule_captures:1790403319725
docker compose exec redis redis-cli DEL arq:result:cron:schedule_captures:1790403319725
```

For a disposable local stack, `docker compose down -v` clears the Redis queue entirely.
