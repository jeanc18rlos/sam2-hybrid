# sam2-hybrid

Companion code for **["Making AI feel realtime with hybrid segmentation"](https://jeanrojas.com/blog/splitting-sam2-encoder-decoder)** on [jeanrojas.com](https://jeanrojas.com).

The full architecture, with the notebook on the user's hardware and the
decoder in their browser:

```
┌──────────────────────────────────────────────┐
│  USER MACHINE  (Colab / local Jupyter)       │
│   image.jpg ──► sam2.1 encoder.onnx          │
│                       │                      │
│                       ▼                      │
│                 embedding.bin                │
│                 + manifest.json              │
│                 (~16 MB float16)             │
└──────────────────────────────────────────────┘
                       │
                       ▼  drag-drop / upload / pre-bake
┌──────────────────────────────────────────────┐
│  VERCEL APP  (Next.js + onnxruntime-web)     │
│   embedding.bin ──► decoder.onnx (WebGPU)    │
│                          ▲                   │
│                          │                   │
│                    user clicks               │
│                          │                   │
│                          ▼                   │
│                       masks                  │
└──────────────────────────────────────────────┘
```

## What's in this repo

| Path                      | What it is                                                    |
| ------------------------- | ------------------------------------------------------------- |
| `notebooks/sam2_encode.ipynb` | The 5-cell encoder notebook. Runs on Colab T4 in ~2 min. |
| `replicate/`              | `cog.yaml` + `predict.py` — encoder as a hosted Replicate model. |
| `hf-space/`               | Gradio app variant for a Hugging Face Space.                  |
| `apps/web/`               | The Next.js app (decoder in the browser). The "Deploy to Vercel" button on the blog post points at this directory. |

## Quick start

### Run the encoder on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeanc18rlos/sam2-hybrid/blob/main/notebooks/sam2_encode.ipynb)

Notebook does:

1. Pin a working torch + ONNX toolchain.
2. Download `sam2.1_hiera_large.pt` (~900 MB).
3. Split into encoder + decoder ONNX via `samexporter`.
4. Encode any image you upload.
5. Pack the output as a flat `embedding.bin` + `manifest.json` ready for the
   browser.

### Run the encoder on Replicate

```bash
cd replicate/
cog login
cog push r8.im/jrojastechnology/sam2-encoder
```

After the push, copy the version hash and set it on the web app:

```env
NEXT_PUBLIC_REPLICATE_MODEL=jrojastechnology/sam2-encoder:<version-hash>
```

The web app uses the user's own Replicate API token (entered in a settings
dialog) — there's no token of yours exposed to the client.

### Run the encoder as a Hugging Face Space

```bash
huggingface-cli login
huggingface-cli repo create sam2-encoder --type space --space_sdk gradio
git clone https://huggingface.co/spaces/jrojastechnology/sam2-encoder
cp hf-space/* sam2-encoder/
cd sam2-encoder && git add . && git commit -m "init" && git push
```

### Deploy the web app

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fjeanc18rlos%2Fsam2-hybrid&project-name=sam2-hybrid&repository-name=sam2-hybrid&env=NEXT_PUBLIC_REPLICATE_MODEL,REPLICATE_API_TOKEN)

The two env vars are optional. If you skip them the app still works for
pre-baked demo images and notebook-uploaded files.

## Performance reference (sam2.1 hiera large)

| Operation        | Cost                              | Frequency           |
| ---------------- | --------------------------------- | ------------------- |
| Encode image     | 8–10s on Apple Silicon GPU        | Once per image      |
| Encode image     | ~5s on a Colab T4                 | Once per image      |
| Decode w/ prompt | 30–60ms on WebGPU                 | Once per click/drag |
| Decode w/ prompt | 200–400ms on WASM SIMD            | Once per click/drag |
| Embedding size   | ~16 MB float16                    | Transferred once    |

## License

MIT for the code in this repository. The pretrained SAM2 weights are released
by Meta under their own license — see
[facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
for the exact terms.

The cover demo images used in the article are from
[ClickSEG](https://github.com/XavierCHEN34/ClickSEG) (Apache 2.0).

## Article

Full write-up with mermaid diagrams, the live in-browser demo, and the
deployment story:
**[jeanrojas.com/blog/splitting-sam2-encoder-decoder](https://jeanrojas.com/blog/splitting-sam2-encoder-decoder)**
