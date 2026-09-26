# Interface Contract — DRAFT

## 1. ML → Backend

The ML workstream provides a Python function that the backend calls:

```python
predict(image) -> Prediction
```

**Input**
- `image`: a single camera frame (PIL Image or RGB numpy array; *TBD*).

**Output**

| Field                 | Type                          | Description                                           |
|-----------------------|-------------------------------|-------------------------------------------------------|
| `status`              | `"dry" \| "wet" \| "flooded"` | Most likely class                                     |
| `confidence`          | float, 0–1                    | Probability of `status`                               |
| `stage_probabilities` | object                        | Probability for each class; the values add up to 1    |
| `heatmap_png`         | bytes                         | PNG heatmap overlay showing where the model sees water |

**Example** (`heatmap_png` shortened):

```json
{
  "status": "flooded",
  "confidence": 0.87,
  "stage_probabilities": { "dry": 0.03, "wet": 0.10, "flooded": 0.87 },
  "heatmap_png": "<bytes>"
}
```

**Open questions**
- Input format, and does the model resize the image itself?
- Heatmap resolution: same as the input frame, or fixed?
- Minimum confidence before the backend reports `flooded`?

## 2. Backend → Frontend

The API returns JSON. Timestamps use ISO 8601 in UTC.

### `GET /cameras`

Returns a list of all cameras with their latest status.

```json
[
  {
    "id": "GA-I75-0123",
    "lat": 33.7490,
    "lon": -84.3880,
    "road": "I-75 NB @ 10th St",
    "status": "wet",
    "confidence": 0.72,
    "updated_at": "2026-09-23T21:15:00Z"
  }
]
```

| Field        | Type                                     | Description                    |
|--------------|------------------------------------------|--------------------------------|
| `id`         | string                                   | Camera ID                      |
| `lat`, `lon` | float                                    | Camera location                |
| `road`       | string                                   | Road or location description   |
| `status`     | `"dry" \| "wet" \| "flooded"`            | Latest prediction              |
| `confidence` | float, 0–1                               | Confidence of latest prediction |
| `updated_at` | string (ISO 8601)                        | Time of the latest prediction  |

### `GET /cameras/{id}`

Returns one camera with more detail.

```json
{
  "id": "GA-I75-0123",
  "lat": 33.7490,
  "lon": -84.3880,
  "road": "I-75 NB @ 10th St",
  "status": "flooded",
  "confidence": 0.87,
  "updated_at": "2026-09-23T21:15:00Z",
  "frame_url": "/frames/GA-I75-0123/latest.jpg",
  "heatmap_url": "/heatmaps/GA-I75-0123/latest.png",
  "depth_bin": "unknown",
  "history": [
    { "status": "wet",     "confidence": 0.70, "timestamp": "2026-09-23T21:00:00Z" },
    { "status": "flooded", "confidence": 0.87, "timestamp": "2026-09-23T21:15:00Z" }
  ]
}
```

Includes all the fields from `GET /cameras`, plus:

| Field         | Type                  | Description                                                |
|---------------|-----------------------|------------------------------------------------------------|
| `frame_url`   | string                | URL of the latest camera frame                             |
| `heatmap_url` | string                | URL of the heatmap PNG for the latest frame                |
| `depth_bin`   | string                | Estimated water depth bin, or `"unknown"` (bins *TBD*)     |
| `history`     | array                 | Recent statuses, oldest first: `{status, confidence, timestamp}` |

Returns `404` if the camera ID does not exist.

**Open questions**
- What are the depth bins (for example `"<6in"`, `"6-12in"`, `">12in"`)?
- How much history: last N frames or last N hours?
- How often is the data refreshed? Polling interval for the frontend?
