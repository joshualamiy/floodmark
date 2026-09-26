# Other-camera data acquisition (wet-road gap fill)

Scope: the model has almost no real "wet road, not flooded" traffic-camera
data (31 frames, 30 from one NYSDOT still-image set), every 511GA frame held
is dry-and-night, and Phase 4's independent evaluation found the model never
detects wet pavement and may use a "looks like 511GA at night -> dry"
shortcut. Job: find public DOT camera sources (ideally archives of past rain)
outside 511GA, and pull matched wet/dry frames from the **same** cameras so
camera identity can't stand in for the label. Full architecture notes and
the candidate table also live in `docs/OTHERCAMS.md` (kept in sync with this
report); this file is the point-in-time run report.

## Candidate sources (verified this session, WebSearch/WebFetch)

| Source | Official URL | Access | Terms of use (quoted) | View | Image size | Archive? | Verdict |
|---|---|---|---|---|---|---|---|
| **Iowa DOT RWIS via Iowa Environmental Mesonet (IEM), Iowa State University** | `mesonet.agron.iastate.edu/RWIS/camera.phtml`, `mesonet.agron.iastate.edu/json/webcam.py` | Open, no key, no account | `mesonet.agron.iastate.edu/disclaimer.php`: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose" (attribution appreciated, not required) | roadside/bridge RWIS camera, elevated, same style as 511GA | 800x450 / 640x480 / 480x270 (observed) | **Yes**, since ~2009-2010 for most sites | **Used** — the only real archive found, so we can pull already-rained-on days instead of waiting on Atlanta's forecast |
| Caltrans CWWP2 CCTV (California) | `cwwp2.dot.ca.gov/documentation/cctv/cctv.htm`, `cwwp2.dot.ca.gov/data/d{1-12}/cctv/cctvStatusD{NN}.json` | Open, no key, no account | `dot.ca.gov/conditions-of-use`: "information presented on this website... is considered in the public domain. It may be distributed or copied as permitted by law." Field docs: "There is no charge for the use of this data." | roadside CCTV, elevated | varies | **No** — page states: "Caltrans traffic camera video footage and still images are neither retained nor archived." Each record carries 2 older stills (~60 min apart) | Live-only candidate for later, not built this pass (see decisions) |
| 511WI (Wisconsin DOT) | `511wi.gov/developers/help` | **Needs a developer key** + account | governed by WisDOT's linked 511 terms; throttled 10 calls/60s | roadside CCTV | unverified | **Not registered** (rule: don't register for keys) |
| 511NY (New York State DOT) | `511ny.org/developers/help`, `511ny.org/developers/daa` | **Needs a developer key** + signed Developer's Access Agreement | "Access to the data feed is provided solely to individuals, companies, or agencies (Data Disseminators) that agree to comply" with the DAA | roadside CCTV | unverified | **Not registered.** (The project's existing NYSDOT wet/dry stills are a separate, static Zenodo dataset, CC BY 4.0 — not this live API.) |
| UDOT Traffic (Utah DOT) | `udottraffic.utah.gov/help/endpoint/cameras` | **Needs a developer key** + account (same iPeriscope/ibi511 platform family as 511GA/511WI/511NY) | throttled 10 calls/60s, DAA-gated | roadside CCTV | unverified | **Not registered** |
| MnDOT RWIS/traffic cameras | `mn.gov/dot`, `511mn.org` | inconclusive — no public API doc found | unverified | roadside CCTV/RWIS | unverified | **Unverified, skipped** rather than guess at a URL |
| 511GA (Georgia) | `511ga.org/developers/doc` | key (already held) | out of scope — covered by `src/ga511/` | roadside CCTV | 450x253 etc. | live only | Already in use, not part of this package |

Every URL was fetched or searched this session; nothing invented. Rows marked
unverified weren't pursued past the access-method check because they need a
key we weren't told to register for.

## Source used: Iowa DOT RWIS via IEM

`src/othercams/iowa_rwis.py`. Per site ("group"): scrape the RWIS camera
dropdown (84 physical sites, ~418 view angles), fuzzy-match each site name to
Iowa's `IA_RWIS` station geojson by town-name token overlap (**76/84 matched**
at score >= 0.5; unmatched groups are skipped, not guessed at), pick a plain
roadway view over bridge-deck/zoom/sensor close-ups, pull a year of daily
precipitation from Open-Meteo (one call per site, no key) to rank rain days
(>= 3mm, Apr-Oct only, to dodge snow) and confirm dry days (0mm that day and
the 2 days before), then check IEM's real archive JSON for that camera+date,
filtered to a rough daytime UTC window (13:00-22:00Z, since night "wet" was
exactly the mistake Phase 4 flagged for 511GA). For each daytime candidate,
`ga511.weather.get_precip_and_label()` — **the same function, same
thresholds** ga511 uses (>= 0.2mm/1h or >= 1mm/3h -> `likely_wet`; 0mm/6h ->
`likely_dry`) — is called on the frame's *exact* timestamp, because a rainy
day can still have dry hours. Only frames matching the day's target label are
downloaded, decoded, and run through `ga511.quality.classify()` (tiny/
low-variance/frozen-repeat checks, reused read-only). Output: same 19-column
schema as `data/ga511/frames.csv`, at `data/othercams/iowa_rwis/frames.csv`,
images at `data/othercams/iowa_rwis/frames/<view_id>/<ts>.jpg`.

Not chosen/built this pass: Caltrans (no archive — see decisions below), and
the three keyed 511 platforms (not registered for, per the brief).

## Bug found and fixed mid-run: Open-Meteo 429s

First full-scale attempt (60 groups, 20 worker threads) hit
`429 Too Many Requests` on Open-Meteo's archive API almost immediately —
~20 threads each fired their first daily-precip request within the same
second. This mattered more than a normal rate-limit error: `ga511.weather`'s
Open-Meteo fallback (read-only, not modified) caches a failed lookup as
`{}` keyed by grid-cell+hour, so a 429 there doesn't just delay one frame,
it silently poisons that location/hour as "no data" for good. Fixed by
adding a small in-process global pacer (`_pace()`, >= 0.6s between any two
metadata calls, across all threads) plus retry-with-backoff on 429/5xx for
this module's own Open-Meteo and IEM-archive-listing calls, and by routing
every call into `ga511.weather.get_precip_and_label()` through the same
pacer. Verified with a 10-camera concurrent run (7/10 groups produced
frames, 0 new 429s) before launching the real job. Image fetches were never
affected — those already had their own per-view+global rate limit (below).

## Rate limiting

- **Image fetches**: literally "at most one fetch per camera view every 5
  minutes" (brief rule 4), enforced cross-process via
  `data/othercams/iowa_rwis/last_fetch.json` + `fcntl.flock`, same pattern as
  `ga511/snapshot.py`, plus a ~1.4 req/s global cap. Because this is a
  per-*view* cap and 60 distinct views ran concurrently, the whole job still
  finished in 2 hours rather than in (views x 5 min).
- **Metadata calls** (Open-Meteo, IEM archive listing, `ga511.weather`
  lookups) aren't image fetches, so the literal 5-min-per-view rule doesn't
  apply to them, but they needed their own pacing (see bug above) — this is a
  judgment call, flagged for the orchestrator/user in case strict-literal
  compliance is wanted everywhere instead.
- Descriptive User-Agent throughout: `floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)`.
- No key was ever used or logged (this source doesn't have one).

## Run

```
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis start   # detached
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis status
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis stop
```

Ran `start` with defaults (`--max-groups 60 --wet-target 4 --dry-target 4
--max-workers 20`) out of the 76 matched groups. Detached via
`subprocess.Popen(..., start_new_session=True)`, PID file at
`data/othercams/iowa_rwis/collect.pid`, log at
`logs/jobs/othercams_iowa_rwis_run.log` (rotating detail log at
`logs/jobs/othercams_iowa_rwis.log`). Finished on its own in ~2h (00:58-02:58
in the log); it's a **one-shot batch job**, not a continuously-running
daemon — the whole point of this source is archived past rain, so there's
nothing for a live loop to wait on. It's not running anymore (job process
exited cleanly on completion; `status` now reports `running: false`).

## Frames collected

| | count |
|---|---|
| Total attempts | 374 |
| Good frames | **340** |
| `likely_wet` | **176** |
| `likely_dry` | **164** |
| Dead: `frozen_repeat` | 32 |
| Dead: `http_error` | 2 |
| Distinct cameras (>=1 good frame) | **44** (of 60 attempted, of 76 matched) |
| Distinct counties | 38 |
| Cameras with **both** wet and dry frames (true same-camera pairs) | **41** |
| Cameras with wet-only (no confirmed-dry day found in range) | 3 (IDOT-014, IDOT-038, IDOT-050) |
| Resolution | 800x450 (236), 640x480 (80), 480x270 (24) |
| Capture hour (UTC) | concentrated 13:00-19:00Z (roughly 7am-2pm local), all daytime |

41 of 44 cameras have genuine matched wet/dry pairs from the *same* camera —
comfortably past the brief's ">= 30 cameras" bar, and the whole reason this
source is more useful than the existing 31-frame NYSDOT gap-fill (which
isn't matched-pair by camera in the same way). 16 of the 60 attempted groups
produced nothing (no qualifying rain-day/dry-day combination found in the
~13-month lookback, or the archive had a coverage gap for that camera —
confirmed spot-check example: `IDOT-004` (Ames) returned 0 archived images
across every date tried, seemingly a long-dead camera in IEM's system despite
being listed).

No API key was ever involved (this source has none), so there's nothing to
redact; no secrets were logged.

## Visual quality check (Read tool, sampled, seed=42)

Sampled 8 `likely_wet` + 8 `likely_dry` good frames spread across distinct
cameras/counties and looked at each one directly.

**Dry (7 distinct cameras checked): 7/7 visually consistent** with dry
pavement — clear/bright or at least non-wet-looking road surface in every
case. One (`IDOT-054`, Sibley) has its road view mostly blocked by a
foreground tree; the label is still probably right (the sliver of visible
road looks dry) but the frame is a weak candidate for "unusable."

**Wet (8 cameras checked): mixed, and worth reporting honestly:**
- **Strong match (2/8)**: `IDOT-007` (US 34 @ Burlington, 7.0mm/1h) and
  `IDOT-074` (US 18 @ Thrush Ave, 4.6mm/1h) both show unambiguous wet
  pavement — visible sheen, water reflections, and actual raindrops on the
  camera housing.
- **Plausible match (2/8)**: `IDOT-013` (US 34 @ Creston, 0.5mm/1h) and
  `IDOT-058` (US 218 @ Plainfield, 1.1mm/1h) show fog/mist and a
  visibly-damp-looking near-field road surface, consistent with recent rain,
  though the sheen itself is harder to confirm through the haze than in the
  two strong-match cases.
- **Ambiguous, no visible effect (2/8)**: `IDOT-001` (0.6mm/1h) and
  `IDOT-038` (0.2mm/1h, right at the 1h threshold floor) look like ordinary
  overcast-day dry pavement to the eye.
- **Apparent mismatch (2/8)**: `IDOT-060` (0.0mm/1h, 1.7mm/3h) and `IDOT-047`
  (1.3mm/1h, 9.5mm/3h) both look **fully dry** in bright/hazy sun despite a
  `likely_wet` weak label. Plausible explanations: Open-Meteo's grid
  resolution (~0.1 degrees, several km) can miss highly localized summer
  convective rain at the exact camera pixel, or the road had already dried
  in direct sun by the exact archived timestamp chosen.

**Takeaway**: the weak-label rule (shared with 511GA, unchanged here) is
reliable for a clear rain event but noisier near its own threshold and for
patchy convective rain — exactly why the brief and `docs/PLAN.md` treat
`likely_wet` as "candidates for manual review, never ground truth." I did
not write any manual labels; the user labels via the tool. I'd flag the 2
apparent-mismatch cameras above for a first look if spot-checking by hand.

No labels were written to a manual-labels file. No dataset frames were added
to `docs/`, `tests/`, or `reports/figures/` (all images live under
`data/othercams/`, which is gitignored like the rest of `data/`).

## Labeling tool

Works unmodified, since the schema and root layout match:
```
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.label_tool --port 8000 --ga511-root data/othercams/iowa_rwis
```

## Tests

`tests/test_othercams_schema.py` (4 tests: schema equals `ga511.collect`'s
field list, append/read round trip, missing-file case) and
`tests/test_othercams_iowa_rwis.py` (12 tests: option-dropdown parsing, site
geojson parsing, name-matching with and without a match, roadway-regex
extraction, preferred-view selection avoiding bridge-deck/zoom, full
camera-table build, wet/dry day-picking including the "missing prior-day
data means not a safe dry day" case, daytime-hour filter, subsampling, and
timestamp parsing). All inline fixtures, no network, no data files. 16/16
pass; `../my_env/bin/python -m ruff check .` is clean from `flood-ml/`; the
full repo suite (`../my_env/bin/python -m pytest -q tests`) is 262/262
passing after these additions. `tests/conftest.py` already had a
`test_othercams_*.py` entry requiring `requests`/`imagehash`/`pandas`/
`dotenv` before CI collects these files, which matches what this module's
import chain actually needs.

## Packages installed

**None.** Everything used (`requests`, `Pillow`, `ImageHash`) was already
present in `my_env` from `requirements-train.txt` / prior phases.

## Files

- `src/othercams/__init__.py`, `paths.py`, `schema.py`, `jobs.py`,
  `iowa_rwis.py` (new package)
- `tests/test_othercams_schema.py`, `tests/test_othercams_iowa_rwis.py`
- `docs/OTHERCAMS.md` (candidate table + architecture reference)
- `docs/DATASETS.md` (one row appended, other rows untouched)
- `docs/phase_reports/othercams.md` (this file)
- Data (not tracked in git — `data/` is gitignored except this report):
  `data/othercams/iowa_rwis/frames.csv`, `frames/<view_id>/<ts>.jpg` (340
  images), `cameras.csv` (the 76-row matched camera table), `cache/`
  (camera-table cache + per-site daily-precip cache), `last_fetch.json`,
  `collect.pid` (stale after completion), `frames.csv.lock`.
- Shared side effect (read-only rule, not a file I own): calling
  `ga511.weather.get_precip_and_label()` writes cache entries into the
  existing `data/ga511/weather_cache/` directory (per-location/per-hour JSON
  files). No `ga511` code or `data/ga511/frames.csv`/labels were modified.

## What failed / remains unverified

- The Open-Meteo 429 burst above (fixed; see "bug found and fixed").
- 16 of 60 attempted camera groups produced zero frames (see "frames
  collected") — not a bug, just no qualifying day found in range for those
  specific sites, or (in `IDOT-004`'s case) an apparently dead camera in
  IEM's own archive despite being listed in the dropdown.
- Camera-to-site geocoding is a fuzzy name match (76/84 groups matched,
  score >= 0.5), not surveyed coordinates — fine for a regional weather
  lookup, flagged in `docs/OTHERCAMS.md` as an approximation.
- Dead frames (`frozen_repeat`/`http_error`) have no saved sample image for
  this source (see decisions #3 below), so I couldn't visually audit *why*
  32 frames were flagged frozen-repeat, only that they were.
- Wisconsin (511WI), New York (511NY), and Utah (UDOT) camera APIs are all
  real, verified, keyed endpoints on the same platform family as 511GA —
  **unverified image size/terms detail** because a key wasn't requested.
- MnDOT: no open camera API/archive found in this search pass; genuinely
  inconclusive, not confirmed either way.

## Decisions for the orchestrator / user

1. **Keyed 511 platforms (511WI, 511NY, UDOT)** are real candidates for more
   volume/diversity later, but need the user to register for a developer
   key and accept each state's Developer Access Agreement — not something
   this worker should do unilaterally.
2. **Caltrans CWWP2 CCTV** is open, no-key, public-domain, but explicitly
   "neither retained nor archived" — a live-only source. I didn't build a
   live collector for it this pass: the archive source above already met
   the volume/camera-diversity targets on its own, and standing up a "wait
   for rain somewhere in California" live loop is a separate, open-ended
   effort. Flagging as a live option if the user wants more data later or
   wants a genuinely-live (not archived) wet example.
3. **Dead-sample images are not saved for this source.** `ga511.quality.
   save_dead_sample()` is hardcoded to write into `data/ga511/dead_samples/`
   (outside this worker's file ownership per the brief), so I skip that
   call rather than write outside my lane; `dead_reason` is still recorded
   in `frames.csv` per row. If dead-sample auditing for this source matters,
   that function would need a `dirpath` parameter added (a one-line change
   to `ga511/quality.py`, which I was told to treat read-only).
4. **No free-text "notes" column exists** for flagging obstructed/unusable
   views (e.g. `IDOT-054`'s tree-blocked frame above) — the schema is
   deliberately identical to `data/ga511/frames.csv` per the brief ("writes
   the SAME frames.csv schema"), which has no such column. Recommend the
   user mark these "unusable" via the labeling tool's existing `0` shortcut
   instead, same as any other frame.
5. **The 3 wet-only cameras** (`IDOT-014`, `IDOT-038`, `IDOT-050`) could be
   given a dry counterpart by re-running with a larger `--max-dry-days`, if
   the user wants every camera to have a strict pair; not done here since
   41/44 already clears the target.
6. **Visual QA found 2 of 8 sampled `likely_wet` frames with no visible
   wetness** (see above) — a real property of precip-based weak labels near
   threshold/for patchy rain, not a code bug. Worth the user's attention
   when confirming labels in the tool, and worth the orchestrator knowing
   before citing "N likely_wet frames" as if they were manually confirmed.
