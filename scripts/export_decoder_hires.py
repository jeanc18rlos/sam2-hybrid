"""
Re-export the SAM2 decoder with a 512x512 mask output instead of the
default 256x256.

The vanilla samexporter decoder returns 256x256 logits straight from
predict_masks(). For interactive segmentation in the browser, that
quadruples the upscale factor we have to apply at display time and
amplifies any high-frequency artifacts in the model output. Adding an
in-graph bilinear upsample to 512x512 doubles the source grid before
the browser ever sees it, halves the upscale factor on display, and
the rest of our pipeline (logit Gaussian, sigmoid, closing, border)
runs on a cleaner signal.

Usage:
    python scripts/export_decoder_hires.py 512
    # writes to .cache/onnx/sam2.1_hiera_large.decoder.onnx (replacing the 256 one)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from samexporter.export_sam2 import SAM2ImageDecoder
from sam2.build_sam import build_sam2

CHECKPOINT = Path(".cache/checkpoints/sam2.1_hiera_large.pt")
OUT = Path(".cache/onnx/sam2.1_hiera_large.decoder.onnx")
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class HiResDecoder(SAM2ImageDecoder):
    """SAM2ImageDecoder + an in-graph bilinear upsample on the mask output."""

    def __init__(self, sam_model, multimask_output: bool, output_size: int) -> None:
        super().__init__(sam_model, multimask_output=multimask_output)
        self.output_size = int(output_size)

    def forward(self, *args, **kwargs):
        masks, iou = super().forward(*args, **kwargs)
        if self.output_size != masks.shape[-1]:
            masks = F.interpolate(
                masks,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return masks, iou


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <output-size>  (e.g. 512)")
    out_size = int(sys.argv[1])

    print(f"Building SAM2 hiera-large from {CHECKPOINT}…")
    model = build_sam2(CONFIG, str(CHECKPOINT), device="cpu").cpu()

    decoder = HiResDecoder(model, multimask_output=True, output_size=out_size).cpu()

    embed_size = (model.image_size // model.backbone_stride,) * 2  # 64x64 for 1024 input
    image_embed = torch.randn(1, model.sam_prompt_encoder.embed_dim, *embed_size)
    high_res_feats_0 = torch.randn(1, 32, embed_size[0] * 4, embed_size[1] * 4)
    high_res_feats_1 = torch.randn(1, 64, embed_size[0] * 2, embed_size[1] * 2)

    point_coords = torch.randint(0, 1024, (1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(0, 1, (1, 5), dtype=torch.float)
    mask_input = torch.randn(1, 1, embed_size[0] * 4, embed_size[1] * 4)
    has_mask_input = torch.tensor([1], dtype=torch.float)

    with torch.no_grad():
        masks, iou = decoder(
            image_embed, high_res_feats_0, high_res_feats_1,
            point_coords, point_labels, mask_input, has_mask_input,
        )
        print(f"smoke test  masks={tuple(masks.shape)}  iou={tuple(iou.shape)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.utils.export(
        decoder,
        (image_embed, high_res_feats_0, high_res_feats_1,
         point_coords, point_labels, mask_input, has_mask_input),
        str(OUT),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            "image_embed", "high_res_feats_0", "high_res_feats_1",
            "point_coords", "point_labels", "mask_input", "has_mask_input",
        ],
        output_names=["masks", "iou_predictions"],
        dynamic_axes={
            "point_coords": {0: "num_labels", 1: "num_points"},
            "point_labels": {0: "num_labels", 1: "num_points"},
            "mask_input":   {0: "num_labels"},
            "has_mask_input": {0: "num_labels"},
        },
    )
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
