"""
Encode an image with the SAM2 Hiera Large encoder and produce the
notebook-format bundle (manifest.json + embedding.bin + preview.jpg).

Usage (from sam2-hybrid root):
    python scripts/encode_demo.py IMG_3069.jpeg apps/web/public/demos/jean

Caches the SAM2 checkpoint and exported ONNX under .cache/ so re-encoding
new images is fast.
"""

import io
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
INPUT_SIZE = 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PREVIEW_LONG_EDGE = 2000
PREVIEW_QUALITY = 88


def ensure_checkpoint(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cache_dir / "sam2.1_hiera_large.pt"
    if not ckpt.exists():
        print(f"Downloading {CHECKPOINT_URL}  →  {ckpt}  (~900 MB, one-time)…")
        urllib.request.urlretrieve(CHECKPOINT_URL, ckpt)
    print(f"checkpoint: {ckpt}  ({ckpt.stat().st_size / 1e6:.1f} MB)")
    return ckpt


def ensure_encoder_onnx(ckpt: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = out_dir / "sam2.1_hiera_large.encoder.onnx"
    decoder = out_dir / "sam2.1_hiera_large.decoder.onnx"
    if encoder.exists():
        print(f"encoder onnx (cached): {encoder}  ({encoder.stat().st_size / 1e6:.1f} MB)")
        return encoder
    print("Exporting encoder + decoder via samexporter (this can take a couple of minutes)…")
    cmd = [
        sys.executable, "-m", "samexporter.export_sam2",
        "--checkpoint",     str(ckpt),
        "--output_encoder", str(encoder),
        "--output_decoder", str(decoder),
        "--model_type",     "sam2.1_hiera_large",
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(f"samexporter failed (exit {res.returncode}). See output above.")
    return encoder


def preprocess(image_path: Path) -> tuple[np.ndarray, tuple[int, int], Image.Image]:
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    original_size = img.size  # (width, height) — already EXIF-corrected
    print(f"image: {image_path.name}  original={original_size}")

    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None].astype(np.float32)
    return arr, original_size, img


def write_bundle(
    out_dir: Path,
    image_embed: np.ndarray,
    high_res_0: np.ndarray,
    high_res_1: np.ndarray,
    original_size: tuple[int, int],
    full_image: Image.Image,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        "image_embed":      image_embed.astype(np.float16),
        "high_res_feats_0": high_res_0.astype(np.float16),
        "high_res_feats_1": high_res_1.astype(np.float16),
    }

    manifest = {
        "preview": "preview.jpg",
        "originalWidth":  original_size[0],
        "originalHeight": original_size[1],
        "tensors": {},
        "totalBytes": 0,
    }

    bin_path = out_dir / "embedding.bin"
    with bin_path.open("wb") as f:
        offset = 0
        for name, arr in tensors.items():
            manifest["tensors"][name] = {
                "offset": offset,
                "shape":  list(arr.shape),
                "dtype":  "float16",
            }
            f.write(arr.tobytes())
            offset += arr.nbytes
        manifest["totalBytes"] = offset

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    preview = full_image.copy()
    preview.thumbnail((PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE), Image.LANCZOS)
    preview.save(out_dir / "preview.jpg", quality=PREVIEW_QUALITY, optimize=True)

    print(
        f"wrote: {out_dir}/{{preview.jpg, embedding.bin ({bin_path.stat().st_size / 1e6:.1f} MB), manifest.json}}"
    )


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <input-image> <output-dir>")

    img_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if not img_path.exists():
        sys.exit(f"input image not found: {img_path}")

    cache = Path(".cache")
    ckpt = ensure_checkpoint(cache / "checkpoints")
    encoder_onnx = ensure_encoder_onnx(ckpt, cache / "onnx")

    print(f"loading encoder onnx via providers={ort.get_available_providers()}")
    sess = ort.InferenceSession(str(encoder_onnx), providers=ort.get_available_providers())

    arr, original_size, full_img = preprocess(img_path)

    print("encoding…")
    high_res_0, high_res_1, image_embed = sess.run(None, {"image": arr})
    print(
        "  image_embed",      image_embed.shape, image_embed.dtype,
        "\n  high_res_feats_0", high_res_0.shape,
        "\n  high_res_feats_1", high_res_1.shape,
    )

    write_bundle(out_dir, image_embed, high_res_0, high_res_1, original_size, full_img)
    print(f"\n✓ done. drop these into the app or commit to {out_dir}")


if __name__ == "__main__":
    main()
