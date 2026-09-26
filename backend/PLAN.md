# Image Processing Pipeline Plan

## Goal

Build a Dockerized image-processing pipeline for scheduled Georgia 511GA
camera captures. The pipeline will use Cloudflare R2 for durable image
storage, PostgreSQL for metadata and predictions, Redis/ARQ for job
orchestration, and ONNX Runtime CPU inference.

The pipeline will not use TensorFlow, CUDA, NVIDIA runtime support, or local
image storage.

For the MVP, model distribution is intentionally deferred. The worker uses a
temporary `model_run()` implementation that returns realistic randomized
predictions matching the production inference contract.

## Architecture

```text
scheduler
  -> Redis/ARQ work queue
     -> one or more identical workers
        -> download image into memory
        -> upload image to R2
        -> insert image metadata in PostgreSQL
        -> run ONNX CPU inference
        -> upload heatmap to R2
        -> write prediction to PostgreSQL
```

All services run under one `compose.yml`:

- `redis`
- `scheduler`
- `worker`

The website, PostgreSQL database, and Cloudflare R2 remain external services.

The queue represents complete camera capture jobs, not individual pipeline
stages. Every worker performs the full lifecycle for one image. Worker replicas
are interchangeable and can consume jobs in parallel.

Concurrency is a core requirement rather than an optimization. The target is
300-500 concurrent camera downloads because that produces the fastest observed
capture throughput. This concurrency target applies primarily to the network
download portion of the worker; CPU inference, R2 writes, and database writes
must use separate bounded limits.

## Scheduling

The scheduler runs every five minutes and reads active cameras from
PostgreSQL. It enqueues one capture job per camera.

Each capture job contains:

- source name
- source camera ID
- scheduled capture time
- unique capture ID

The ARQ job ID is deterministic for a camera and scheduled UTC five-minute
slot, for example:

```text
capture:ga511:10651:2026-09-26T21:15Z
```

This prevents duplicate scheduler runs from enqueueing the same capture. The
actual successful request time is stored as `fetched_at`; it does not replace
the deterministic job identity.

## Worker

For each camera capture job, the worker:

1. Request `GET https://511ga.org/map/Cctv/{source_view_id}`.
2. Enforce request timeout, response-size limit, and supported content type.
3. Convert the binary response to RGB and re-encode it as JPEG in memory.
4. Compute SHA-256 and collect content metadata.
5. Upload the normalized image to R2.
6. Insert the corresponding row in the `images` table.
7. Run `model_run()` using the in-memory image bytes.
8. Upload the heatmap to R2 when present.
9. Insert the prediction and mark the image as processed.
10. Release the image bytes.

The worker should not write camera images to a local volume. R2 is the durable
source of truth and is used when a completed database record needs to be
reprocessed.

If R2 upload succeeds but the database insert fails, leave the object in R2
and record enough information for a later orphan-object reconciliation task.

## R2 Object Keys

Use deterministic keys containing the source view ID and scheduled UTC capture
time:

```text
captures/{source_view_id}/YYYYMMDDTHHmmZ.jpg
heatmaps/{source_view_id}/YYYYMMDDTHHmmZ.png
```

The scheduled UTC five-minute slot is part of the deterministic job identity,
so retries and duplicate scheduler runs intentionally address the same object.
The stored `images.r2_key` remains the canonical location for later retrieval
and reprocessing.

The worker should use `predict()` initially. `predict_batch()` can be added
later if benchmarks show that micro-batching improves throughput.

The model version stored in `predictions.model_version` must be a stable
serialized representation of the returned data version, Stage A run ID, and
Stage B run ID.

The frontend should surface the model's wet-status note, whose current meaning
is water detected below the flood alert threshold rather than ordinary wet
pavement.

## Database Writes and Idempotency

Workers will write directly to PostgreSQL using a restricted worker database
credential. The website remains responsible for read APIs and presentation.

Use the existing schema:

- `images` stores R2 location, checksum, metadata, and processing state.
- `predictions` stores model output and heatmap location.

Prediction writes must rely on the existing unique constraint for
`image_id`, `model_name`, and `model_version` so inference retries do not
create duplicate predictions.

An image should only be marked processed after its prediction and all required
metadata have been committed successfully.

## Retries and Dead-Letter Handling

ARQ camera jobs should be attempted at most three times, including the initial
attempt. Retries should use a delay and preserve the original job payload.
The MVP uses ARQ retry state rather than adding retry-count columns to the
database.

After the final failure:

- Mark the image as `error` when an image row exists. (`error` is the existing
  database enum value.)
- Store the final error in `processing_error`.
- Preserve the R2 image and any diagnostic data.
- Record the failed job in a dead-letter/failure record for inspection.
- Do not retry the failed job automatically.

Capture failures without an image row should be logged with the camera ID,
capture ID, and error. They must not prevent other cameras from being
processed.

The system should eventually include reconciliation tasks for:

- Failed or unprocessed image rows that can be retried from R2.
- R2 objects uploaded without a matching database row.
- Predictions that exist while the image remains unprocessed.

## Resource Limits

The worker must be designed for high network concurrency. Its download layer
should support a configurable target of 300-500 simultaneous camera requests,
using asynchronous I/O and per-request timeouts. One slow camera must not block
other downloads.

The worker must not run 300-500 model inferences or database transactions at
the same time. Use separate bounded controls for:

- Camera download concurrency: target 300-500.
- R2 upload concurrency: benchmark and cap independently.
- ONNX inference concurrency: start conservatively and tune against the 9900X.
- PostgreSQL connection and transaction concurrency: respect the database
  connection pool limit.

Images are processed in memory, so the worker also needs a maximum response
size, bounded in-flight byte budget, and backpressure when downstream stages
fall behind. The memory budget must account for downloaded JPEG bytes, decoded
PIL/NumPy images, heatmaps, and queued jobs. If the budget is exhausted, new
downloads should wait rather than being accepted unboundedly.

The worker should load the ONNX sessions once and reuse them for multiple
jobs. Worker count, download concurrency, inference concurrency, and database
pool size should all be configurable and benchmarked rather than assumed.

## Docker and Compose

The scheduler and worker may share a Python base image, but should have
separate commands and responsibilities. The worker image needs only the
`flood-ml` runtime dependencies:

- `onnxruntime`
- `numpy`
- `Pillow`

No local image volume is required. Redis data persistence may use a named
volume if queue recovery across Redis restarts is needed.

Compose health checks should cover Redis and worker process liveness. Secrets
such as database, R2, and camera API credentials must come from environment
variables and must not be committed.

## Implementation Order

1. Finalize environment variables and R2 client configuration.
2. Add the camera job payload, key-generation helpers, and database access.
3. Implement the five-minute scheduler and single ARQ work queue.
4. Implement the unified worker: in-memory capture, RGB/JPEG normalization,
   R2 upload, database metadata, `model_run()`, heatmap upload, and prediction
   persistence.
5. Add retry, failure, and dead-letter handling.
6. Add reconciliation jobs for incomplete work and orphaned R2 objects.
7. Implement bounded high-concurrency downloads and separate downstream
   backpressure controls.
8. Benchmark 300-500 download concurrency and tune memory, R2, inference, and
   PostgreSQL limits.
9. Add `compose.yml`, worker scaling, health checks, and configuration.
10. Add tests for key generation, idempotency, retry behavior, concurrency
    limits, and database state transitions.

## Verification

The completed pipeline should be verified by:

- Running one scheduled capture against a test camera.
- Confirming the image and heatmap exist in R2.
- Confirming matching `images` and `predictions` rows exist in PostgreSQL.
- Confirming inference uses the expected model version and status logic.
- Measuring download throughput at concurrency targets of 300 and 500.
- Confirming downstream limits prevent inference, memory, and database
  saturation while downloads remain highly concurrent.
- Simulating download, R2, database, and inference failures.
- Confirming each job stops after three attempts and is recorded as failed.
- Restarting workers and confirming queued and unprocessed work can recover.
- Running the complete stack through the single Compose file.
