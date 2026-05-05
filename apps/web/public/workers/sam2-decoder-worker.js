// SAM2 Decoder Web Worker (ES module)
//
// Two embedding formats are supported, auto-detected by the URL passed in:
//
//   1. "msgpack"   — single .bin produced by the original jeanrojas.com demo.
//                    Detected when the URL ends in .bin and no manifest is
//                    given. Tensors live as Float32 inside the msgpack body.
//
//   2. "manifest"  — manifest.json + embedding.bin (raw float16 buffer)
//                    produced by the SAM2 encoder notebook. Detected when
//                    the URL ends in .json (manifest is fetched first; bin
//                    is fetched alongside).
//
// Both paths converge on the same in-memory shape:
//   { tensors: { name: { data: Float32Array, shape: number[] } }, originalSize }

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.bundle.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

let decoderSession = null;
let currentEmbeddings = null;

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "load-decoder") {
    try {
      self.postMessage({ type: "status", data: "Loading SAM2 decoder model..." });
      const buffer = await fetch(data.modelUrl).then((r) => r.arrayBuffer());
      decoderSession = await ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
      });
      self.postMessage({ type: "decoder-ready" });
    } catch (err) {
      self.postMessage({ type: "error", data: "Decoder load failed: " + err.message });
    }
    return;
  }

  if (type === "load-embeddings") {
    try {
      self.postMessage({ type: "status", data: "Loading pre-computed embeddings..." });
      const { url, imageWidth, imageHeight } = data;

      if (/manifest\.json($|\?)/i.test(url) || url.endsWith(".json")) {
        currentEmbeddings = await loadManifestFormat(url);
      } else {
        currentEmbeddings = await loadMsgpackFormat(url, imageWidth, imageHeight);
      }

      self.postMessage({
        type: "embeddings-ready",
        data: { tensorNames: Object.keys(currentEmbeddings.tensors) },
      });
    } catch (err) {
      self.postMessage({ type: "error", data: "Embeddings load failed: " + err.message });
    }
    return;
  }

  if (type === "load-embeddings-from-files") {
    // Bring-your-own — files come from a FileList; never via URL fetch.
    try {
      self.postMessage({ type: "status", data: "Reading bundle from disk..." });
      const { manifestText, embeddingBuffer } = data;
      currentEmbeddings = parseManifestBundle(manifestText, embeddingBuffer);
      self.postMessage({
        type: "embeddings-ready",
        data: { tensorNames: Object.keys(currentEmbeddings.tensors) },
      });
    } catch (err) {
      self.postMessage({ type: "error", data: "Bundle parse failed: " + err.message });
    }
    return;
  }

  if (type === "decode") {
    if (!decoderSession || !currentEmbeddings) {
      self.postMessage({ type: "error", data: "Model or embeddings not loaded" });
      return;
    }
    try {
      const { points, imageWidth, imageHeight } = data;
      const N = points.length;
      const feeds = {};

      for (const name of ["image_embed", "high_res_feats_0", "high_res_feats_1"]) {
        const t = currentEmbeddings.tensors[name];
        if (t) {
          feeds[name] = new ort.Tensor("float32", new Float32Array(t.data), t.shape);
        }
      }

      const coordsData = new Float32Array(N * 2);
      const labelsData = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        coordsData[i * 2]     = (points[i].x / imageWidth)  * 1024;
        coordsData[i * 2 + 1] = (points[i].y / imageHeight) * 1024;
        labelsData[i]         = points[i].label;
      }
      feeds["point_coords"]   = new ort.Tensor("float32", coordsData, [1, N, 2]);
      feeds["point_labels"]   = new ort.Tensor("float32", labelsData, [1, N]);
      feeds["mask_input"]     = new ort.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]);
      feeds["has_mask_input"] = new ort.Tensor("float32", new Float32Array([0.0]), [1]);

      const results = await decoderSession.run(feeds);

      const masksOutput = results["masks"] || results["output_masks"];
      const iouOutput   = results["iou_predictions"] || results["scores"];

      if (!masksOutput) {
        self.postMessage({
          type: "error",
          data: "No masks in output. Keys: " + Object.keys(results).join(", "),
        });
        return;
      }

      const dims = masksOutput.dims;
      const maskData = masksOutput.cpuData || masksOutput.data;
      const numMasks = dims[1];
      const maskH = dims[2];
      const maskW = dims[3];
      const maskSize = maskH * maskW;
      const iouScores = iouOutput ? (iouOutput.cpuData || iouOutput.data) : null;

      // Always pick the candidate with the largest area.
      //
      // SAM2 returns three candidates that span (part / sub-object / whole).
      // IoU scores are noisy at low N. A largest-area pick matches the
      // user's expectation: a click selects the whole thing, additional
      // positive clicks only ever grow the mask, negative clicks shrink it
      // (because all three candidates contract to honor the exclude point).
      let bestIdx = 0;
      let bestArea = -1;
      for (let m = 0; m < numMasks; m++) {
        let area = 0;
        const start = m * maskSize;
        for (let i = 0; i < maskSize; i++) {
          if (maskData[start + i] > 0) area++;
        }
        if (area > bestArea) { bestArea = area; bestIdx = m; }
      }

      // Quality pipeline. The exporter clamps logits to [-32, 32]; after
      // sigmoid that's essentially a binary 0/1 mask with a 1-2 pixel
      // boundary band. To get a clean upsampled boundary we have to do
      // the heavy lifting in LOGIT space before sigmoid kills the gradient.
      //
      //   1. Extract the chosen candidate's raw 256x256 logits.
      //   2. Bilinear upsample LOGITS to 1024x1024. The boundary band
      //      grows from ~1-2 source pixels to ~4-8 hi-res pixels and
      //      stays a smooth signed-value gradient.
      //   3. Sigmoid the hi-res logits. The wide boundary band yields
      //      a wide soft-alpha ramp.
      //   4. Light 5-tap Gaussian on the hi-res probability map to
      //      polish any remaining sub-pixel jitter.
      const HI = 1024;
      const logitsBest = new Float32Array(maskSize);
      const offset = bestIdx * maskSize;
      for (let i = 0; i < maskSize; i++) {
        logitsBest[i] = maskData[offset + i];
      }

      // Smooth the LOGITS at source resolution. SAM2 occasionally emits
      // adjacent source pixels with opposite-sign logits in textured
      // regions (knit fabrics, foliage, etc.); without this they upsample
      // into a visible "checker" of binary holes. The blur averages
      // neighbouring logits so the boundary becomes a single smooth
      // signed gradient rather than a high-frequency stripe.
      const logitsLo = gaussianBlur5(logitsBest, maskW, maskH);

      const logitsHi = upsampleBilinear(logitsLo, maskW, maskH, HI, HI);

      const probsHi = new Float32Array(HI * HI);
      for (let i = 0; i < probsHi.length; i++) {
        const x = logitsHi[i];
        probsHi[i] = x >= 0
          ? 1 / (1 + Math.exp(-x))
          : Math.exp(x) / (1 + Math.exp(x));
      }
      const finalProbs = gaussianBlur5(probsHi, HI, HI);

      // Transfer the buffer zero-copy — 4 MB is too big to structured-clone.
      self.postMessage(
        {
          type: "mask-result",
          data: {
            mask: finalProbs.buffer,
            width: HI,
            height: HI,
            score: iouScores ? iouScores[bestIdx] : 0,
          },
        },
        [finalProbs.buffer]
      );
    } catch (err) {
      self.postMessage({ type: "error", data: "Decode failed: " + err.message });
    }
  }
};

/* ---------------- format loaders ---------------- */

// Manifest format: notebook output. Two HTTP requests (manifest + bin).
async function loadManifestFormat(manifestUrl) {
  const manifestText = await fetch(manifestUrl).then((r) => r.text());
  const binUrl = manifestUrl.replace(/manifest\.json($|\?)/i, "embedding.bin$1");
  const buffer = await fetch(binUrl).then((r) => r.arrayBuffer());
  return parseManifestBundle(manifestText, buffer);
}

function parseManifestBundle(manifestText, buffer) {
  const manifest = typeof manifestText === "string"
    ? JSON.parse(manifestText)
    : manifestText;
  const tensors = {};
  for (const [name, t] of Object.entries(manifest.tensors)) {
    const count = t.shape.reduce((a, b) => a * b, 1);
    if (t.dtype === "float16") {
      const f16 = new Uint16Array(buffer, t.offset, count);
      const f32 = new Float32Array(count);
      for (let i = 0; i < count; i++) f32[i] = f16ToF32(f16[i]);
      tensors[name] = { data: f32, shape: t.shape };
    } else if (t.dtype === "float32") {
      const f32 = new Float32Array(buffer.slice(t.offset, t.offset + count * 4));
      tensors[name] = { data: f32, shape: t.shape };
    } else {
      throw new Error(`unsupported dtype ${t.dtype} for tensor ${name}`);
    }
  }
  return {
    tensors,
    originalSize: [
      Number(manifest.originalHeight) || 0,
      Number(manifest.originalWidth)  || 0,
    ],
  };
}

// Msgpack format: legacy demo bundle.
async function loadMsgpackFormat(url, imageWidth, imageHeight) {
  const buffer = await fetch(url).then((r) => r.arrayBuffer());
  const { decode } = await import(
    "https://cdn.jsdelivr.net/npm/@msgpack/msgpack@3.0.0-beta2/+esm"
  );
  const decoded = decode(new Uint8Array(buffer));
  const tensors = {};
  for (const [name, tensor] of Object.entries(decoded.tensors)) {
    const bytes = tensor.data instanceof Uint8Array
      ? tensor.data
      : new Uint8Array(tensor.data);
    const aligned = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(aligned).set(bytes);
    tensors[name] = { data: new Float32Array(aligned), shape: tensor.shape };
  }
  return {
    tensors,
    originalSize: decoded.original_size || [imageHeight, imageWidth],
  };
}

// Bilinear upsample of a Float32 grid. ~1M output pixels in ~10 ms.
function upsampleBilinear(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const sx = (sw - 1) / (dw - 1);
  const sy = (sh - 1) / (dh - 1);
  for (let y = 0; y < dh; y++) {
    const fy = y * sy;
    const y0 = Math.floor(fy);
    const y1 = y0 + 1 < sh ? y0 + 1 : sh - 1;
    const wy = fy - y0;
    const r0 = y0 * sw;
    const r1 = y1 * sw;
    const dy = y * dw;
    for (let x = 0; x < dw; x++) {
      const fx = x * sx;
      const x0 = Math.floor(fx);
      const x1 = x0 + 1 < sw ? x0 + 1 : sw - 1;
      const wx = fx - x0;
      const v00 = src[r0 + x0];
      const v01 = src[r0 + x1];
      const v10 = src[r1 + x0];
      const v11 = src[r1 + x1];
      const v0 = v00 + wx * (v01 - v00);
      const v1 = v10 + wx * (v11 - v10);
      out[dy + x] = v0 + wy * (v1 - v0);
    }
  }
  return out;
}

// Separable 5-tap Gaussian blur ([1,4,6,4,1] / 16). Edges clamp.
function gaussianBlur5(src, w, h) {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  // horizontal pass
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const xm2 = Math.max(0, x - 2);
      const xm1 = Math.max(0, x - 1);
      const xp1 = Math.min(w - 1, x + 1);
      const xp2 = Math.min(w - 1, x + 2);
      tmp[row + x] =
        (src[row + xm2] + 4 * src[row + xm1] + 6 * src[row + x] +
         4 * src[row + xp1] + src[row + xp2]) / 16;
    }
  }
  // vertical pass
  for (let y = 0; y < h; y++) {
    const ym2 = Math.max(0, y - 2) * w;
    const ym1 = Math.max(0, y - 1) * w;
    const yc  = y * w;
    const yp1 = Math.min(h - 1, y + 1) * w;
    const yp2 = Math.min(h - 1, y + 2) * w;
    for (let x = 0; x < w; x++) {
      out[yc + x] =
        (tmp[ym2 + x] + 4 * tmp[ym1 + x] + 6 * tmp[yc + x] +
         4 * tmp[yp1 + x] + tmp[yp2 + x]) / 16;
    }
  }
  return out;
}

// IEEE-754 half → single precision.
function f16ToF32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0)    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
