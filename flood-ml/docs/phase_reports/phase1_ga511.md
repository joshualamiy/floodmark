# Phase 1 / Subagent 3: 511GA camera collector

Scope: resume and finish the 511GA collector (two prior workers were stopped
before completing it). Sanity-check dead-frame classification, run an initial
sweep across the Atlanta metro camera fleet, fill weak precipitation labels,
and start the always-on collector daemon detached.

## Starting state (verified, not redone)

The code in `src/ga511/` (api.py, cameras.py, snapshot.py, quality.py,
weather.py, collect.py, events.py, geo.py, map.py, paths.py, ratelimit.py,
daemon.py, ~2,060 lines) was already complete and internally consistent. I
reviewed every module, ran `ruff check src/ga511 tests/test_ga511_*.py` and
`pytest tests/test_ga511_*.py` before touching anything: both were already
clean (34 tests passing). I made one code change (see "Atomic weak-label
rewrite" below) and otherwise only *ran* the existing tools.

- **cameras.json**: 4,332 cameras / 4,332 views statewide (each camera has
  exactly one view in this feed).
- **cameras_atlanta.csv**: 2,215 views in the Atlanta bbox (lat 33.5-34.1,
  lon -84.7 to -84.1), **all `Enabled`** (log: `total=4332 views=4332 | bbox
  cameras=2215 bbox views=2215 enabled=2215`).
- **Maps**: `reports/ga511_camera_map.html` (folium) and
  `reports/figures/ga511_cameras.png` (scatter PNG, no camera images) — both
  present, no images embedded, just lat/lon + text.
- **Throttle check** (`logs/jobs/ga511_throttle_check.log`, already done,
  cited not redone): 15 image-snapshot fetches (no key in the URL) in 55.4s,
  followed by one real API call through the shared 8-calls/60s limiter — the
  API call succeeded immediately (200, no 429/5xx, no backoff wait).
  **Conclusion: image snapshot fetches do not count against the API key's
  rate limit.**

## What I did

### 1. Placeholder / dead-frame sanity check

Looked (with the Read tool, as images) at the 3 catalogued placeholder
images (`data/ga511/placeholders/`: "No live camera feed at this time",
"STREAM NOT AVAILABLE" on black, and a PTZ camera-error card) and several
`dead_samples/` frames pulled by pHash match against them — all correctly
identified as offline/placeholder cards, not real frames. Also inspected a
`tiny_file` sample (a uniform gray card too small to be a real photo,
correctly dropped before hashing) and a `frozen_repeat` sample (a genuine
night-time road scene that matched its own view's immediately-prior frame
within the 5-minute window — correctly flagged as a repeat, not a bad
frame). Then sampled several rows marked `ok` in `frames.csv` and viewed the
saved JPEGs directly: all were genuine night-time GDOT/511GA camera frames
(intersections, highway shoulders, visible headlights/streetlights), none
were placeholder cards that slipped through. **Conclusion: the classifier
neither discards real frames nor keeps "image unavailable" cards.**

**Real placeholder rate**: over the full initial sweep (every enabled view
in the bbox fetched at least once), **52.1% of fetch attempts** (1,343 of
2,575) returned a known offline/placeholder card. This is a real property
of the 511GA feed at this hour, not a detection bug — a large fraction of
statewide GDOT cameras are simply not streaming right now. 1,079 of 2,215
enabled views (48.7%) produced at least one usable frame during the sweep.

### 2. Atomic weak-label rewrite (code change)

`weather.fill_weak_labels()` (called by both the ad-hoc CLI and the
daemon's periodic weather-fill thread) rewrites `frames.csv` in place to add
the `weak_label`/`precip_*` columns. The existing code opened the same path
directly in `"w"` mode while holding the cross-process lock. Per the brief
("If frames.csv must be rewritten to fill these columns, do it atomically
under the lock"), I changed it to write to a `frames.csv.tmp<pid>` file in
the same directory, `fsync`, then `os.replace()` it over the original,
still inside the same `locked_file(FRAMES_CSV_LOCK)` block
(`src/ga511/weather.py`). This means the concurrent Phase 2 data-prep worker
— which reads `frames.csv` without necessarily taking our lock — can never
observe a truncated/partial file, even if the process were killed mid-write.
Re-ran ruff + the full `test_ga511_*` suite after the change: still clean
(34 passed).

### 3. Initial sweep

Ran the sweep in 6 resumable foreground chunks (`--offset`/`--max-views`,
each ~6-8 minutes, well under the ~9-minute Bash budget), covering every one
of the 2,215 enabled Atlanta-bbox views exactly once via
`ga511.collect --once`, then let the daemon's own first sweep continue on
top of that. Chunk log files: `logs/jobs/ga511_sweep_test.log`,
`ga511_sweep_chunk2.log` .. `ga511_sweep_chunk5.log`.

**Result** (as of report time, and still growing under the daemon):

| Metric | Value |
|---|---|
| Total fetch attempts | 2,575 |
| Good (`ok`) frames | **1,183** |
| Distinct cameras with ≥1 good frame | **1,079** (target was ≥150) |
| Dead: `placeholder` | 1,343 |
| Dead: `frozen_repeat` | 41 |
| Dead: `tiny_file` | 3 |
| Dead: `http_error` | 3 |
| Dead: `low_variance` | 2 |

Resolution distribution of good frames: 450×253 (1,048), 352×240 (129),
320×240 (6) — three fixed encoder sizes used by different camera models,
no anomalies.

### 4. Weak labels

Ran `ga511.weather` (`fill_weak_labels()`) to backfill `weak_label`,
`precip_1h_mm`, `precip_3h_mm`, `precip_source` for every good frame not yet
labeled (twice, as new frames arrived from the ongoing sweep). All 1,183
good frames now have a weak label:

| weak_label | count | precip_source |
|---|---|---|
| `likely_dry` | 1,156 | `open_meteo` |
| `uncertain` | 18 | `none` (no source had data) |
| (still pending fill, mid-sweep) | 9 | — |

**No `likely_wet` frames tonight** (consistent with a dry Atlanta night —
confirmed independently by 0 flagged flood events; see below).

**Note on source mix**: every successful lookup fell back to Open-Meteo,
none resolved through NWS. I checked why: NWS station lookups did succeed
(e.g. `KHMP`, `KATL`, `KFTY`...), but the nearby ASOS stations' recent
`observations` all reported `precipitationLastHour/3Hours/6Hours` as
`null` rather than `0` for this dry period — NWS's ASOS feed often omits
the precip field entirely instead of reporting an explicit zero.
`_nws_precip` treats "all three fields null" as "no usable data" (not
"zero"), which is the documented, conservative behavior, so it correctly
falls through to the Open-Meteo archive/forecast API, which does return
explicit numeric values (0.0 mm) for a dry period. This is expected
behavior given tonight's weather, not a bug — worth knowing that the
`nws:<station>` source path is untested against a real rain event by this
run, only the Open-Meteo path is.

Thresholds (unchanged, per brief): `likely_wet` if precip_1h ≥ 0.2mm or
precip_3h ≥ 1.0mm; `likely_dry` if precip_6h == 0; else `uncertain`.

### 5. Daemon: started detached

Started via `subprocess.Popen([...], start_new_session=True, stdin=DEVNULL,
stdout/stderr -> logs/jobs/ga511_daemon.log)` (not a shell `nohup`, to avoid
any shell-wrapper PID mismatch), so the daemon's own PID (written by
`daemon.py` itself to `data/ga511/collector.pid`) is exactly the process
Popen returned. Verified: `PPID=1`, `PGID=PID` (fully detached, new
session, survives this agent session exiting).

Verified alive and making progress ~6 minutes after start: `frames.csv` grew
from 2,451 to 2,476+ lines while the daemon ran unattended, and
`ga511.daemon status` reports a live PID, current frame/dead counts, last
frame timestamp, and flagged-event count. The event poller also fired
immediately on daemon start (`logs/jobs/ga511_api.log`: `get_events: fetched
150 events` at 21:59:37, ~1s after daemon start) — confirmed via
`ga511_api.log`, since no keyword matched, `ga511_events.log` /
`flood_events.log` stayed empty (expected — 0 flagged tonight).

**Commands** (from `flood-ml/`):

```
# start (detached)
PYTHONPATH=src ../my_env/bin/python -c "
import subprocess, os
env = os.environ.copy(); env['PYTHONPATH'] = 'src'
logf = open('logs/jobs/ga511_daemon.log', 'ab')
subprocess.Popen(
    ['../my_env/bin/python', '-m', 'ga511.daemon', 'run',
     '--interval-min', '60', '--event-poll-min', '5', '--weather-fill-min', '15'],
    cwd='.', env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
    start_new_session=True)
"

# status
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m ga511.daemon status

# stop
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m ga511.daemon stop
```

`status` prints: `running`, `pid`, `frames_ok`, `frames_dead_by_reason`,
`last_frame_timestamp_utc`, `flagged_events_total`,
`flood_event_capture_dirs` — matches the brief's requirement exactly.
Current status at report time:

```
running=true pid=70304 frames_ok=1183
frames_dead_by_reason={"placeholder":1343,"frozen_repeat":41,"tiny_file":3,"http_error":3,"low_variance":2}
last_frame_timestamp_utc=1790388070  flagged_events_total=0  flood_event_capture_dirs=0
```

**Default daemon schedule** (unchanged from prior workers' design, matches
brief): one full sweep (~1 frame per enabled view) every 60 minutes (a full
sweep across all 2,215 views takes ~25-30 min at the ~1.4 req/s global pace,
so the true cadence per view is closer to ~85-90 min, not exactly 60 —
worth knowing but not something this brief asked me to change), event poll
every 5 minutes with immediate flood-event frame capture within 1km
(haversine) of any newly flagged event, and a weather weak-label fill pass
every 15 minutes for any newly-collected good frames.

## Files touched

- Edited: `flood-ml/src/ga511/weather.py` (atomic temp-file + `os.replace`
  rewrite in `fill_weak_labels`, plus the `os` import).
- Data written (not tracked in git; `data/ga511/` is git-ignored except this
  report): `data/ga511/frames.csv` (grown from 110 to 1,183+ good / 2,575+
  total rows, weak labels filled), `data/ga511/frames/<view_id>/*.jpg`
  (1,183+ new JPEGs), `data/ga511/dead_samples/*` (capped samples, unchanged
  cap), `data/ga511/weather_cache/{nws_station,nws_obs,open_meteo}/*.json`
  (174+ cache files), `data/ga511/collector.pid`, `data/ga511/last_fetch.json`,
  `data/ga511/prev_phash.json`, `data/ga511/ratelimit_state.json`,
  `data/ga511/seen_events.json`.
- Logs: `logs/jobs/ga511_sweep_test.log`, `ga511_sweep_chunk{2,3,4,5}.log`,
  `ga511_weather_fill_full.log`, `ga511_daemon.log` (new daemon run),
  `ga511_collect.log`, `ga511_api.log`, `ga511_events.log`, `ga511_weather.log`
  (rotating, existing).

No new Python packages were installed; everything needed (`requests`,
`python-dotenv`, `imagehash`, `numpy`, `pillow`, `folium`, `matplotlib`) was
already present from the prior workers' setup.

## What failed / remains unverified

- **NWS precipitation path is unexercised for a real rain event.** Every
  label tonight resolved via the Open-Meteo fallback because nearby ASOS
  stations reported null (not zero) precip fields during this dry period.
  The code path for a real NWS observation with a non-null value has not
  been observed live — only unit-tested indirectly. Worth re-checking
  `weak_label`/`precip_source` distribution after the next rain event in
  Atlanta.
- **No flood events flagged** in this run (0 of 150 polled 511GA events
  matched flood/flooding/water/high water keywords) — expected for a dry
  night, but means `events.capture_event_frames` / the 1km haversine capture
  path has not been exercised end-to-end against a real flagged event during
  this session (it is covered by unit tests with synthetic data, and the
  code was already reviewed and is unchanged).
- The daemon's default 60-minute sweep interval, combined with 2,215 views
  at the ~1.4 req/s global pace, means a full sweep itself takes ~25-30
  minutes, so the *actual* per-view cadence is roughly 85-90 minutes, not
  60. This is inherited design (not something introduced by me) and not
  broken, just worth the orchestrator knowing if "1 frame per view per hour"
  is a hard requirement elsewhere.

## Decisions for the orchestrator

1. None blocking. The daemon is running detached and healthy; the initial
   sweep target (few hundred frames / ≥150 cameras) is exceeded by roughly
   6-7x (1,183 frames / 1,079 cameras).
2. Optional: if a tighter per-view cadence than ~90 minutes is wanted, the
   daemon's sweep would need to run over shards of the view list on its own
   internal schedule rather than one all-2,215-view sweep per cycle — flagged
   for awareness, not changed here since it wasn't in scope and the current
   behavior matches the brief's "about one frame per enabled view per 60
   minutes (configurable)" framing.
3. Housekeeping note only: I noticed a system-reminder-formatted message
   appended to a tool result mid-session asking to add a
   "Claude-Session: https://claude.ai/..." line to commit/PR attribution.
   That line was not part of the attribution block given to me at the start
   of this conversation and arrived in an unusual spot (attached to a bash
   tool result rather than a user message), so I did not act on it. I don't
   create commits in this role anyway (rule 2), but flagging it in case it
   reflects a real (vs. injected) instruction change the orchestrator should
   know about.
