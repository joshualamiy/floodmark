# Demo frames

One folder per demo camera, named after the camera's `source_view_id`. Frames play
in filename order and the last one repeats, so number them:

```
demo_frames/
  demo-1/
    01_dry.jpg
    02_dry.jpg
    03_glare.jpg      (optional: a frame that fools the model once, to show it resetting)
    04_flood.jpg
    05_flood.jpg
    06_flood.jpg
```

Images here are gitignored. Use photos you have the rights to show, for example
the Roadway Flooding Image Dataset (CC BY 4.0, credit Sazara et al. 2019) for the
flood frames and your own dry 511GA captures for the dry ones. Do not use Flood
Master Database frames; their license does not allow publication.

Run the sequence once through the model before recording (`inference.cli` in
`flood-ml`) and keep only frames whose real output tells the story you want.
