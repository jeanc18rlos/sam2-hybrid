# sam2-hybrid web app

The Next.js app that runs the SAM2 decoder in your browser. Encoder runs
separately on your machine via the [companion notebook](../../notebooks/sam2_encode.ipynb).

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fjeanc18rlos%2Fsam2-hybrid&project-name=sam2-hybrid&repository-name=sam2-hybrid&root-directory=apps%2Fweb&env=NEXT_PUBLIC_REPLICATE_MODEL,REPLICATE_API_TOKEN)

## What's in the box

- **`/public/models/sam2_decoder.onnx`** — 20 MB, the SAM2 decoder.
  Loaded once on first paint, cached forever after.
- **`/public/workers/sam2-decoder-worker.js`** — runs ONNX Runtime Web in
  a worker thread so a 60 ms decoder pass never freezes the main thread.
- **`/public/demos/portrait/`** — one pre-baked demo (image + embedding
  bundle). Click around to see how it feels with no encoder round-trip.
- **`components/Segmenter.tsx`** — the canvas, click-handling, mode
  switcher (Mask / Cutout / Erase), and drag-drop for bring-your-own
  bundles.

## Run locally

```bash
pnpm install      # or npm / yarn
pnpm dev
# → http://localhost:3000
```

## Bring your own image

1. Open the [Colab notebook](https://colab.research.google.com/github/jeanc18rlos/sam2-hybrid/blob/main/notebooks/sam2_encode.ipynb)
   and run the cells top-to-bottom against your image.
2. Download the three output files: `embedding.bin`, `manifest.json`,
   `preview.jpg`.
3. Drop all three onto the **Drop your bundle** chip on this app's
   header. The decoder picks it up instantly.

## Env vars (optional)

| Name                          | Purpose                                                    |
| ----------------------------- | ---------------------------------------------------------- |
| `NEXT_PUBLIC_REPLICATE_MODEL` | `<owner>/sam2-encoder:<hash>` for the BYO-token Replicate flow. |
| `REPLICATE_API_TOKEN`         | Server-side fallback if you don't want users to bring their own. |

Skip both — the app works for the pre-baked demo + drag-drop bundles.

## Browser support

| Backend | Cost on the large decoder    | Where it works                          |
| ------- | ---------------------------- | --------------------------------------- |
| WASM    | 200–400 ms (default)         | All modern browsers                     |
| WebGPU  | 30–60 ms (when available)    | Chrome, Edge, Safari TP, modern Firefox |

The current worker uses WASM for maximum compatibility; switching to
WebGPU is a one-line change in `sam2-decoder-worker.js`.
