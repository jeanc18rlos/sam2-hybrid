"""
SAM2 encoder microservice for Replicate.

Wraps the ONNX-exported sam2.1_hiera_large encoder. Takes one image, returns
a zip containing:
  - manifest.json  (offsets/shapes/dtypes for the three feature tensors)
  - embedding.bin  (raw float16 buffer; ~12-18 MB)

The browser-side decoder (in apps/web) consumes this exact format.

Build: `cog build`
Push:  `cog push r8.im/<owner>/sam2-encoder`
"""

from cog import BasePredictor, Input, Path
from PIL import Image
import numpy as np
import onnxruntime as ort
import json
import io
import zipfile

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 1024


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the ONNX session once at container start."""
        self.session = ort.InferenceSession(
            "weights/sam2.1_hiera_large.encoder.onnx",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(
        self,
        image: Path = Input(description="Image to encode"),
    ) -> Path:
        img = Image.open(image).convert("RGB")
        original_width, original_height = img.size

        # Preprocess to the model's fixed 1024x1024 input.
        x = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = (np.array(x, dtype=np.float32) / 255.0 - MEAN) / STD
        arr = arr.transpose(2, 0, 1)[None].astype(np.float32)

        high_res_0, high_res_1, image_embed = self.session.run(
            None, {"image": arr}
        )

        # Cast to fp16 — decoder is robust to it, halves payload size.
        tensors = {
            "image_embed": image_embed.astype(np.float16),
            "high_res_feats_0": high_res_0.astype(np.float16),
            "high_res_feats_1": high_res_1.astype(np.float16),
        }

        # Flat binary layout + JSON manifest. Browser-friendly.
        manifest = {
            "originalWidth": original_width,
            "originalHeight": original_height,
            "tensors": {},
        }
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
        return out
