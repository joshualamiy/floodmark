# Class counts

## Per split x label

| split | dry | wet | flooded | not_flooded | total |
|---|---|---|---|---|---|
| train | 6922 | 92 | 2530 | 579 | 10123 |
| val | 1796 | 22 | 542 | 134 | 2494 |
| test | 957 | 31 | 542 | 112 | 1642 |

## Per source x label

| source | dry | wet | flooded | not_flooded | total |
|---|---|---|---|---|---|
| alleyfloodnet | 0 | 0 | 601 | 509 | 1110 |
| eu_flood_2013 | 0 | 0 | 1894 | 236 | 2130 |
| flood_master_test | 0 | 0 | 62 | 0 | 62 |
| fred | 1294 | 0 | 631 | 0 | 1925 |
| ga511 | 8126 | 1 | 0 | 80 | 8207 |
| iowa_rwis | 226 | 114 | 0 | 0 | 340 |
| nysdot_road_surface | 29 | 30 | 0 | 0 | 59 |
| roadway_flooding | 0 | 0 | 426 | 0 | 426 |

## Per split x source

| split | alleyfloodnet | eu_flood_2013 | flood_master_test | fred | ga511 | iowa_rwis | nysdot_road_surface | roadway_flooding |
|---|---|---|---|---|---|---|---|---|
| train | 774 | 1494 | 0 | 1104 | 6296 | 244 | 28 | 183 |
| val | 169 | 317 | 0 | 206 | 1585 | 48 | 15 | 154 |
| test | 167 | 319 | 62 | 615 | 326 | 48 | 16 | 89 |

## label_source counts

| label_source | count |
|---|---|
| weak_precip | 7377 |
| dataset_label | 3299 |
| sequence_condition | 1236 |
| mask | 1177 |
| manual | 1168 |
| ai_review | 2 |

## water_frac_road stats by assigned label

| label | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| dry | 58 | 0.000 | 0.000 | 0.000 | 0.000 |
| wet | 0 | - | - | - | - |
| flooded | 1119 | 0.812 | 1.000 | 0.334 | 1.000 |

## Sources excluded wholesale (not in the per-source table above)

| source | n images | reason |
|---|---|---|
| flood_area_segmentation | 290 | Kaggle Flood Area Segmentation: a 12-image random spot-check found the source is elevated drone/aerial flood photography (whole towns/fields from above), not the ground-level road view this classifier targets. CLIP margins were also weakly negative on average across all 290 images (mean pos-neg margin -0.017, max +0.057), so a per-image CLIP threshold would keep only obviously-aerial images or exclude nearly everything. Excluded wholesale rather than row-by-row; never gathered by build_manifest.gather_rows. |
| flood_master_test (Italian video) | 1406 | Flood Master Database Italian test video: 24 frames spread across the full video are drone/news b-roll (on-screen NYTimes credit, panning and tilting aerial shots of vineyards and rooftops), not a fixed road-camera view. Excluded; the Greek video (567 frames, a fixed elevated camera over a flooded street) was reinstated and kept -- see the spot-check table below. |

## CLIP road-scene filter (removed per source)

| source | kept | removed | removed frac |
|---|---|---|---|
| alleyfloodnet | 1110 | 0 | 0.0% |
| eu_flood_2013 | 2130 | 0 | 0.0% |
| flood_master_test | 567 | 0 | 0.0% |
| fred | 4680 | 0 | 0.0% |
| iowa_rwis | 340 | 0 | 0.0% |
| nysdot_road_surface | 59 | 0 | 0.0% |
| roadway_flooding | 426 | 3 | 0.7% |

## Dedup / temporal thinning

- thinned_frames: 3260
- thinning_rule: keep a frame if phash Hamming > 6 from the last *kept* frame, or if 15 consecutive frames have been dropped (resample at least every 16th frame even during a long static stretch)
- cross_source_duplicates_collapsed: 0
- n_after: 15869

| video/sequence group | frames before | frames after |
|---|---|---|
| flood_master_test:fmd_greek_video | 567 | 62 |
| fred:cambogan | 1079 | 615 |
| fred:dairycreek | 858 | 323 |
| fred:holmview | 816 | 209 |
| fred:mountcotton | 1379 | 572 |
| fred:pullenvale | 548 | 206 |

## Per-(source, label) spot-check table

| source | label | rule | population | n viewed | agreement | verdict |
|---|---|---|---|---|---|---|
| roadway_flooding | flooded | mask f>=0.10 | 429 | 20 | 20/20 (100%) | kept |
| roadway_flooding | wet | mask wet-band (unverified) | 8 | 8 | ~6/8 (75%) | EXCLUDED (n<20) |
| roadway_flooding | dry | mask dry-at-zero (unverified) | 1 | 1 | 0/1 (0%; road-closed/flood-adjacent scene, not dry) | EXCLUDED (n<20) |
| fred | flooded | mask road+water-hazard, f>=0.10 | 1865 | 20 | 20/20 (100%) | kept |
| fred | wet | mask wet-band (verified) | 164 | 20 | ~17/20 (85%; dry near-field, flooded road visible ahead) | kept |
| fred | dry | mask dry-at-zero, within flooded sequences (verified) | 199 | 20 | 20/20 (100%) | kept |
| fred | dry | sequence_condition (always allowed) | 2616 | 20 | 20/20 (100%) | kept |
| flood_master_test | flooded | mask bottom-band (Greek video only) | 567 | 20 | 20/20 (100%) | kept (Italian video excluded, aerial/drone news footage) |
| nysdot_road_surface | dry | dataset_label, agreement>=0.67, cropped | 29 | 15 | ~13/15 (87%) | kept |
| nysdot_road_surface | wet | dataset_label, agreement>=0.67, cropped | 30 | 15 | ~14/15 (93%) | kept |
| ga511 | dry/wet/unusable | ai_review (visual, test-split cameras) | 274 | 274 | n/a -- this IS the label (167 in the first pass, 107 more after the 511GA sweep finished) | recorded to ai_review_labels.csv |
| iowa_rwis | dry/wet | manual (user, labels.csv; not a mask rule) | 340 | 340 | n/a -- this IS the label (weak precip label never used) | kept |
| eu_flood_2013 | flooded/not_flooded | dataset_label (hydrologist relevance lists) + CLIP road_score>0 | None | None | n/a -- see the road-filter contact sheets in docs/phase_reports/v3_retrain.md | kept |
| alleyfloodnet | flooded/not_flooded | dataset_label (folder name; not a mask rule) | 1110 | None | n/a -- this IS the label | kept (INCLUDE_ALLEYFLOODNET switch) |
