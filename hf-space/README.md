---
title: SAM2 Encoder
emoji: 🐧
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
---

# SAM2 Hiera Large — encoder Space

Same encoder as the [Replicate model](../replicate/) and the
[Colab notebook](../notebooks/sam2_encode.ipynb), packaged as a Gradio Space.
Drop an image, get back a `bundle.zip` with `manifest.json` + `embedding.bin`
that the browser-side decoder consumes directly.

Companion code for
[Making AI feel realtime with hybrid segmentation](https://jeanrojas.com/blog/splitting-sam2-encoder-decoder).

## Hardware

- **T4 small ($0.40/hr)** — recommended. ~5s per encode.
- **CPU upgrade** — works on the free tier but each encode takes ~30s.

## Setup before pushing

The encoder weights aren't checked into git (850 MB). Place them at
`weights/sam2.1_hiera_large.encoder.onnx` (and the matching `.onnx.data`
file beside it) before pushing the Space, or have the Space pull them
from a Hugging Face dataset on first boot.
