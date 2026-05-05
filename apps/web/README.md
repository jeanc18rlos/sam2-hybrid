# sam2-hybrid web app

Next.js app that runs the SAM2 decoder in the browser.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fjeanc18rlos%2Fsam2-hybrid&project-name=sam2-hybrid&repository-name=sam2-hybrid&env=NEXT_PUBLIC_REPLICATE_MODEL,REPLICATE_API_TOKEN)

## Status

🚧 **Stub** — the live in-browser demo currently runs on
[jeanrojas.com](https://jeanrojas.com/blog/splitting-sam2-encoder-decoder)
inside the blog post. This directory is reserved for the standalone
Vercel template that the article's "Deploy to Vercel" button clones.

If you want the working code today, the relevant files in
[`jeanrojas.com`](https://github.com/jeanc18rlos/jeanrojas.com) are:

- `components/sections/segmentationTile.tsx` — the canvas + click handling
- `public/workers/sam2-decoder-worker.js` — the worker that runs ORT
- `public/models/sam2_decoder.onnx` — the decoder weights
- `public/demo-data/portrait/` — pre-baked embedding bundle

## Env vars (optional)

| Name                          | Purpose                                                           |
| ----------------------------- | ----------------------------------------------------------------- |
| `NEXT_PUBLIC_REPLICATE_MODEL` | `<owner>/sam2-encoder:<version-hash>` for the BYO-token flow.     |
| `REPLICATE_API_TOKEN`         | Server-side fallback if you don't want users to bring their own.  |

Skip both and the app works for pre-baked demos + drag-drop notebook bundles.
