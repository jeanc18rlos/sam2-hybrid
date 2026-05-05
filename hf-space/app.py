"""
SAM2 encoder as a Hugging Face Space.

Same predict() as replicate/predict.py, wrapped in a Gradio interface.
HF Spaces hardware: T4 small ($0.40/hr) is the sweet spot. Free tier
works on CPU but is much slower.
"""

import io
import json
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 1024

# Pick the fastest provider available — CUDA on a paid GPU Space, CPU on free.
SESSION = ort.InferenceSession(
    "weights/sam2.1_hiera_large.encoder.onnx",
    providers=ort.get_available_providers(),
)


def encode(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    ow, oh = img.size

    x = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = (np.array(x, dtype=np.float32) / 255.0 - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None].astype(np.float32)

    high_res_0, high_res_1, image_embed = SESSION.run(None, {"image": arr})

    tensors = {
        "image_embed": image_embed.astype(np.float16),
        "high_res_feats_0": high_res_0.astype(np.float16),
        "high_res_feats_1": high_res_1.astype(np.float16),
    }

    manifest = {"originalWidth": ow, "originalHeight": oh, "tensors": {}}
    bin_buf = io.BytesIO()
    offset = 0
    for name, t in tensors.items():
        manifest["tensors"][name] = {
            "offset": offset,
            "shape": list(t.shape),
            "dtype": "float16",
        }
        bin_buf.write(t.tobytes())
        offset += t.nbytes
    manifest["totalBytes"] = offset

    out = Path("/tmp/bundle.zip")
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("embedding.bin", bin_buf.getvalue())
    return str(out)


demo = gr.Interface(
    fn=encode,
    inputs=gr.Image(type="filepath", label="Image to encode"),
    outputs=gr.File(label="bundle.zip — embedding.bin + manifest.json"),
    title="SAM2 Hiera Large — encoder",
    description=(
        "Drop an image. The Space encodes it with sam2.1_hiera_large and "
        "returns a ~16 MB bundle the in-browser decoder can consume directly. "
        "See the [write-up](https://jeanrojas.com/blog/splitting-sam2-encoder-decoder)."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
