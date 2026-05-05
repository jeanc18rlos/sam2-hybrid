# Replicate encoder

Encoder microservice for Replicate. Same interface as `notebooks/sam2_encode.ipynb` produces.

## Setup

1. Place the exported encoder weights at `weights/sam2.1_hiera_large.encoder.onnx`
   (and the matching `.onnx.data` file beside it). Run the notebook once to get them.
2. `cog login`
3. `cog push r8.im/<your-replicate-username>/sam2-encoder`
4. Copy the version hash printed by Replicate.
5. Set the web app's `NEXT_PUBLIC_REPLICATE_MODEL` env var to
   `<your-replicate-username>/sam2-encoder:<hash>`.

## Local test

```bash
cog predict -i image=@my_photo.jpg
# → /tmp/bundle.zip with manifest.json + embedding.bin inside
```

## Notes

- This intentionally does *not* include weights in git. They're 850 MB and
  Replicate accepts them as part of the build context. Drop them under
  `weights/` before `cog push`.
- The web app uses **the user's own Replicate API token**, not yours. No
  shared GPU bill.
