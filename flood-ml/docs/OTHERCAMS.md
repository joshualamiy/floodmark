# Other-camera sources (wet-road gap fill)

Why this exists: the 511GA collector has almost no real "wet road, not
flooded" frames (31 total, 30 from one NYSDOT still-image set), every 511GA
frame we hold is dry-and-night, and the independent evaluation found the
model never detects wet pavement and may be using a "looks like 511GA at
night -> dry" shortcut. This package pulls real wet vs. dry roadside-camera
frames from public DOT camera archives outside 511GA, so camera identity
can't stand in for the wet/dry label. See `docs/phase_reports/othercams.md`
for what was actually collected.

## Candidate sources

| Source | Official URL | Access | Terms of use (quoted) | View | Image size | Archive? | Verdict |
|---|---|---|---|---|---|---|---|
| **Iowa DOT RWIS via Iowa Environmental Mesonet (IEM), Iowa State University** | `mesonet.agron.iastate.edu/RWIS/camera.phtml` (viewer), `mesonet.agron.iastate.edu/json/webcam.py` (archive JSON, no key) | Open, no key, no account. Scrape the camera dropdown once for the id list; site lat/lon from `geojson/network.php?network=IA_RWIS` | `mesonet.agron.iastate.edu/disclaimer.php`: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose" (attribution to "the Iowa Environmental Mesonet of Iowa State University" appreciated, not required) | roadside/bridge-mounted RWIS camera, elevated, same style as 511GA | 800x450 (observed) | **Yes** — since ~2009-2010 for most sites, ~5-10 min cadence, 1 min during severe weather | **Used.** Only real archive we found; lets us pick real past rain days instead of waiting |
| Caltrans CWWP2 CCTV (California) | `cwwp2.dot.ca.gov/documentation/cctv/cctv.htm` (docs), `cwwp2.dot.ca.gov/data/d{1-12}/cctv/cctvStatusD{NN}.json` (camera list) | Open, no key, no account | `dot.ca.gov/conditions-of-use`: "information presented on this website, unless otherwise indicated, is considered in the public domain. It may be distributed or copied as permitted by law." Field docs: "There is no charge for the use of this data." | roadside CCTV, elevated | varies by camera | **No** — page states explicitly: "Caltrans traffic camera video footage and still images are neither retained nor archived." Each camera JSON record does carry 2 older stills (`referenceImage1UpdateAgoURL`, `referenceImage2UpdatesAgoURL`, ~60 min apart) | Candidate for a **live** collector later if it's raining somewhere in CA; not built this pass (see decisions below) |
| 511WI (Wisconsin DOT) | `511wi.gov/developers/help`, `511wi.gov/help/endpoint/cameras` | **Needs a developer API key** + registered account | Access governed by WisDOT's 511 terms/legal notices (linked from the developer portal); throttled 10 calls/60s | roadside CCTV | unverified | **Not registered** — flagged for the user (rule: don't register for keys) |
| 511NY (New York State DOT) | `511ny.org/developers/help`, `511ny.org/developers/daa` | **Needs a developer key** + a signed Developer's Access Agreement | "The Developer's Access Agreement sets out the terms and conditions governing access to real-time and static transportation data... Access to the data feed is provided solely to individuals, companies, or agencies (Data Disseminators) that agree to comply" | roadside CCTV | unverified | **Not registered.** (Note: the project's existing NYSDOT wet/dry stills came from a separate, static Zenodo dataset — CC BY 4.0 — not this live API; see `docs/DATASETS.md`.) |
| UDOT Traffic (Utah DOT) | `udottraffic.utah.gov/help/endpoint/cameras`, `udottraffic.utah.gov/api/v2/get/cameras` | **Needs a developer key** + registered account (same iPeriscope/ibi511 platform family as 511GA/511WI/511NY) | Throttled 10 calls/60s; access gated behind account signup, same pattern as 511GA's own DAA | roadside CCTV | unverified | **Not registered** |
| MnDOT RWIS/traffic cameras | `mn.gov/dot`, `511mn.org` | inconclusive — no public JSON/API doc found in this search pass | unverified | roadside CCTV / RWIS | unverified | **Unverified, skipped.** Didn't find a documented open endpoint; not worth guessing at a URL |
| 511GA (Georgia) | `511ga.org/developers/doc` | key (already held by this project) | covered by the existing `docs/PLAN.md` / ga511 collector, not this package | roadside CCTV | 450x253 / 352x240 / 320x240 | live only | **Already in use** by `src/ga511/`, out of scope here |

Every URL above was fetched or searched this session (WebSearch/WebFetch); none were invented. Rows marked "unverified" mean the terms/size weren't confirmed because the source wasn't pursued past the access-method check (keyed sources weren't registered for, per the brief).

## What was built: `src/othercams/iowa_rwis.py`

One source, chosen because it's the only **open, no-key, real archive** found — the entire point of this phase is pulling *past* rain events instead of waiting for Atlanta to get wet.

**Pipeline, per camera site ("group"):**
1. `load_camera_table()` scrapes the RWIS camera dropdown (84 physical sites, ~418 individual view angles) and fuzzy-matches each site's name to Iowa's `IA_RWIS` station geojson (lat/lon/county) by town-name token overlap. **76 of 84 groups matched** (score >= 0.5); the rest are skipped rather than guessed at. Bridge-deck/zoom/sensor close-up views are deprioritized in favor of a plain roadway view per site.
2. `daily_precip()` pulls one year of daily precipitation totals for that site from Open-Meteo's archive API (one HTTP call per site, no key) to rank candidate rain days (>= 3mm, Apr-Oct only, to dodge snow) and confirm dry days (0mm that day and the 2 days before).
3. `list_archive_images()` asks IEM's real archive JSON for that camera+date, filtered to a rough daytime UTC window (13:00-22:00Z), so we don't grade wetness on a headlight-lit night frame (the exact mistake the independent evaluation flagged for 511GA).
4. For each daytime candidate, `ga511.weather.get_precip_and_label()` — the **same function, same thresholds** the 511GA collector uses (>= 0.2mm/1h or >= 1mm/3h -> `likely_wet`; 0mm/6h -> `likely_dry`) — is called on that frame's *exact* timestamp, because a "rainy day" can still have dry hours. Only frames whose weak label matches the day's target are downloaded.
5. Each image is decoded and run through `ga511.quality.classify()` (tiny-file, low-variance, frozen-repeat checks — reused as-is, read-only). No placeholder-image catalog exists for this source yet (IEM's archive JSON only lists images that really exist, so "camera offline" placeholder cards are rare here vs. 511GA); `placeholder_hashes=[]` is passed, a no-op check.
6. Good frames go to `data/othercams/iowa_rwis/frames/<view_id>/<ts>.jpg`; every attempt (good or dead) gets one row in `data/othercams/iowa_rwis/frames.csv`, same 19-column schema as `data/ga511/frames.csv`.

## Rate limiting — read this before changing intervals

- **Image fetches**: literally "at most one fetch per camera view every 5 minutes" (brief rule 4), enforced cross-process via a `last_fetch.json` + `fcntl.flock`, exactly like `ga511/snapshot.py`. A global ~1.4 req/s cap applies across all views too. Because this is a per-*view* cap and ~60-76 distinct views run concurrently, a whole run still finishes in well under two hours rather than in view-count x 5 minutes.
- **Metadata calls** (Open-Meteo daily precip, IEM's archive-listing JSON, and the `ga511.weather` lookups this module triggers) are **not** image fetches, so the literal 5-min view rule doesn't apply to them — but they still needed throttling. The first full-scale run hit Open-Meteo 429s from ~20 threads bursting their first request simultaneously (see phase report). Fixed with a small in-process global pacer (`_pace()`, >= 0.6s between any two such calls, any thread) plus retry-with-backoff on 429/5xx. This matters because `ga511.weather`'s Open-Meteo fallback caches a failed lookup as permanent "no data" for that hour — a 429 there doesn't just delay a frame, it could poison its label — so this pacer protects the existing ga511 weather cache too, not just this run.
- Descriptive User-Agent everywhere: `floodmark-research/0.1 (+https://github.com/joshualamiy/floodmark)`.

## Labeling tool

Works unmodified:
```
cd flood-ml && PYTHONPATH=src ../my_env/bin/python -m prep.label_tool --port 8000 --ga511-root data/othercams/iowa_rwis
```

## Collector commands

```
cd flood-ml
# one-shot camera-table build (matches + writes data/othercams/iowa_rwis/cameras.csv)
PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis cameras

# run a collection pass in the foreground (useful for a small --max-groups smoke test)
PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis run --max-groups 5 --wet-target 1 --dry-target 1

# start/stop/status: detached, PID file at data/othercams/iowa_rwis/collect.pid,
# log at logs/jobs/othercams_iowa_rwis_run.log
PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis start
PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis status
PYTHONPATH=src ../my_env/bin/python -m othercams.iowa_rwis stop
```

This is a **one-shot batch job** (it terminates once every selected site has
tried its candidate days), not a continuously-running daemon like
`ga511.daemon` — there's no "live" component here because the whole point of
this source is archived past rain, not waiting for current rain. See the
phase report for whether re-running with a live Caltrans-style source later
makes sense.

## Known limitations / approximations

- **Camera-to-site geo matching is name-based, not exact.** Town names in
  the camera dropdown are matched to `IA_RWIS` station names by word overlap;
  76/84 matched at score >= 0.5. This is good enough for a regional weather
  lookup (Open-Meteo/NWS grid resolution) but not surveyed-coordinate
  precision. A handful of matches were spot-checked by eye against the
  fetched images (see phase report) and matched their real Iowa location.
- **Daytime window is a fixed UTC band (13:00-22:00Z)**, approximating
  daylight across both CST and CDT rather than computing exact sunrise/
  sunset per site/date.
- **No placeholder-image catalog for this source.** If IEM ever serves a
  "camera offline" card as a real image (not observed in this run), it would
  currently only be caught by the low-variance check, not a phash match.
