// SAM2 Decoder Web Worker (ES Module)
// Pre-encoded embeddings + browser-side decoder for instant click-to-segment

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.bundle.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

let decoderSession = null;
let currentEmbeddings = null; // { tensors: { name: { data: Float32Array, shape: number[] } }, originalSize }

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "load-decoder") {
    try {
      self.postMessage({ type: "status", data: "Loading SAM2 decoder model..." });
      const modelUrl = data.modelUrl;
      const response = await fetch(modelUrl);
      const buffer = await response.arrayBuffer();

      decoderSession = await ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
      });

      self.postMessage({ type: "decoder-ready" });
    } catch (err) {
      self.postMessage({ type: "error", data: "Decoder load failed: " + err.message });
    }
  } else if (type === "load-embeddings") {
    try {
      self.postMessage({ type: "status", data: "Loading pre-computed embeddings..." });

      const { url, imageWidth, imageHeight } = data;
      const response = await fetch(url);
      const buffer = await response.arrayBuffer();

      // Decode msgpack
      const { decode } = await import("https://cdn.jsdelivr.net/npm/@msgpack/msgpack@3.0.0-beta2/+esm");
      const decoded = decode(new Uint8Array(buffer));

      // Parse tensors
      const tensors = {};
      for (const [name, tensor] of Object.entries(decoded.tensors)) {
        const bytes = tensor.data instanceof Uint8Array ? tensor.data : new Uint8Array(tensor.data);
        const aligned = new ArrayBuffer(bytes.byteLength);
        new Uint8Array(aligned).set(bytes);
        tensors[name] = { data: new Float32Array(aligned), shape: tensor.shape };
      }

      currentEmbeddings = {
        tensors,
        originalSize: decoded.original_size || [imageHeight, imageWidth],
      };

      self.postMessage({
        type: "embeddings-ready",
        data: { tensorNames: Object.keys(tensors) },
      });
    } catch (err) {
      self.postMessage({ type: "error", data: "Embeddings load failed: " + err.message });
    }
  } else if (type === "decode") {
    if (!decoderSession || !currentEmbeddings) {
      self.postMessage({ type: "error", data: "Model or embeddings not loaded" });
      return;
    }

    try {
      const { points, imageWidth, imageHeight } = data;
      const N = points.length;

      // Build decoder inputs — all float32
      const feeds = {};

      // Encoder features
      for (const name of ["image_embed", "high_res_feats_0", "high_res_feats_1"]) {
        const tensor = currentEmbeddings.tensors[name];
        if (tensor) {
          feeds[name] = new ort.Tensor("float32", new Float32Array(tensor.data), tensor.shape);
        }
      }

      // Point coords — scaled to 1024x1024
      const coordsData = new Float32Array(N * 2);
      const labelsData = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        coordsData[i * 2] = (points[i].x / imageWidth) * 1024;
        coordsData[i * 2 + 1] = (points[i].y / imageHeight) * 1024;
        labelsData[i] = points[i].label;
      }
      feeds["point_coords"] = new ort.Tensor("float32", coordsData, [1, N, 2]);
      feeds["point_labels"] = new ort.Tensor("float32", labelsData, [1, N]);

      // Mask input (no previous mask)
      feeds["mask_input"] = new ort.Tensor("float32", new Float32Array(256 * 256), [1, 1, 256, 256]);
      feeds["has_mask_input"] = new ort.Tensor("float32", new Float32Array([0.0]), [1]);

      // Run inference
      const results = await decoderSession.run(feeds);

      // Parse masks output — shape [1, numMasks, H, W]
      const masksOutput = results["masks"] || results["output_masks"];
      const iouOutput = results["iou_predictions"] || results["scores"];

      if (!masksOutput) {
        self.postMessage({ type: "error", data: "No masks in output. Keys: " + Object.keys(results).join(", ") });
        return;
      }

      const dims = masksOutput.dims;
      const maskData = masksOutput.cpuData || masksOutput.data;
      const numMasks = dims[1];
      const maskH = dims[2];
      const maskW = dims[3];
      const maskSize = maskH * maskW;

      // Find best mask by IoU score
      let bestIdx = 0;
      if (iouOutput) {
        const scores = iouOutput.cpuData || iouOutput.data;
        let bestScore = -Infinity;
        for (let i = 0; i < numMasks; i++) {
          if (scores[i] > bestScore) {
            bestScore = scores[i];
            bestIdx = i;
          }
        }
      }

      // Extract best mask
      const bestMask = new Float32Array(maskSize);
      const offset = bestIdx * maskSize;
      for (let i = 0; i < maskSize; i++) {
        bestMask[i] = maskData[offset + i] > 0 ? 1.0 : 0.0;
      }

      self.postMessage({
        type: "mask-result",
        data: {
          mask: Array.from(bestMask),
          width: maskW,
          height: maskH,
          score: iouOutput ? (iouOutput.cpuData || iouOutput.data)[bestIdx] : 0,
        },
      });
    } catch (err) {
      self.postMessage({ type: "error", data: "Decode failed: " + err.message });
    }
  }
};
