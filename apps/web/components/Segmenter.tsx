"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { DragEvent as ReactDragEvent } from "react";

type Point = { x: number; y: number; label: 0 | 1 };
type ViewMode = "edit" | "cutout" | "erase";

type Demo = {
  id: string;
  label: string;
  imageUrl: string;
  /** msgpack .bin URL OR manifest.json URL — worker auto-detects. */
  embeddingsUrl: string;
  imageWidth: number;
  imageHeight: number;
};

const DEMOS: Demo[] = [
  {
    id: "portrait",
    label: "Portrait",
    imageUrl: "/demos/portrait/image.jpg",
    embeddingsUrl: "/demos/portrait/embeddings.bin",
    imageWidth: 800,
    imageHeight: 1200,
  },
  {
    id: "jean",
    label: "Jean",
    imageUrl: "/demos/jean/preview.jpg",
    embeddingsUrl: "/demos/jean/manifest.json",
    // EXIF-corrected portrait orientation. Worker uses the manifest's
    // originalWidth/originalHeight to scale click coords, so these only
    // need to match the preview.jpg aspect for the canvas sizing.
    imageWidth: 1500,
    imageHeight: 2000,
  },
];

const DECODER_URL = "/models/sam2_decoder.onnx";

type Source =
  | { kind: "demo"; demo: Demo }
  | {
      kind: "byo";
      objectImageUrl: string;
      objectEmbeddingsUrl: string;
      width: number;
      height: number;
    };

export default function Segmenter() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const maskOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const maskBinaryRef = useRef<HTMLCanvasElement | null>(null);

  const [source, setSource] = useState<Source>({ kind: "demo", demo: DEMOS[0] });
  const [status, setStatus] = useState("Loading…");
  const [decoderReady, setDecoderReady] = useState(false);
  const [embeddingsReady, setEmbeddingsReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);
  const [score, setScore] = useState<number | null>(null);
  const [view, setView] = useState<ViewMode>("edit");
  const [dragOver, setDragOver] = useState(false);
  const viewRef = useRef<ViewMode>("edit");
  useEffect(() => {
    viewRef.current = view;
  }, [view]);

  const sourceParams = useMemo(() => {
    if (source.kind === "demo") {
      return {
        imageUrl: source.demo.imageUrl,
        embeddingsUrl: source.demo.embeddingsUrl,
        imageWidth: source.demo.imageWidth,
        imageHeight: source.demo.imageHeight,
      };
    }
    return {
      imageUrl: source.objectImageUrl,
      embeddingsUrl: source.objectEmbeddingsUrl,
      imageWidth: source.width,
      imageHeight: source.height,
    };
  }, [source]);

  const draw = useCallback(
    (
      pts: Point[],
      maskOverlay: HTMLCanvasElement | null,
      maskBinary: HTMLCanvasElement | null,
      mode: ViewMode
    ) => {
      const canvas = canvasRef.current;
      const img = imgRef.current;
      if (!canvas || !img) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const hasMask = maskOverlay && maskBinary;

      if (!hasMask || mode === "edit") {
        ctx.drawImage(img, 0, 0);
        if (maskOverlay) {
          ctx.save();
          ctx.globalAlpha = 0.5;
          ctx.drawImage(
            maskOverlay,
            0,
            0,
            maskOverlay.width,
            maskOverlay.height,
            0,
            0,
            canvas.width,
            canvas.height
          );
          ctx.restore();
        }
        for (const pt of pts) {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 9, 0, Math.PI * 2);
          ctx.fillStyle =
            pt.label === 1
              ? "rgba(96, 165, 250, 0.95)"
              : "rgba(244, 63, 94, 0.95)";
          ctx.fill();
          ctx.strokeStyle = "white";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        return;
      }

      if (mode === "cutout" && maskBinary) {
        ctx.drawImage(img, 0, 0);
        ctx.save();
        ctx.globalCompositeOperation = "destination-in";
        ctx.drawImage(
          maskBinary,
          0,
          0,
          maskBinary.width,
          maskBinary.height,
          0,
          0,
          canvas.width,
          canvas.height
        );
        ctx.restore();
        return;
      }

      if (mode === "erase" && maskBinary) {
        ctx.drawImage(img, 0, 0);
        ctx.save();
        ctx.globalCompositeOperation = "destination-out";
        ctx.drawImage(
          maskBinary,
          0,
          0,
          maskBinary.width,
          maskBinary.height,
          0,
          0,
          canvas.width,
          canvas.height
        );
        ctx.restore();
        return;
      }
    },
    []
  );

  // Boot the worker once.
  useEffect(() => {
    const worker = new Worker("/workers/sam2-decoder-worker.js", {
      type: "module",
    });
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const { type, data } = e.data;
      if (type === "status") {
        setStatus(data);
      } else if (type === "decoder-ready") {
        setDecoderReady(true);
      } else if (type === "embeddings-ready") {
        setEmbeddingsReady(true);
        setStatus("Click anywhere on the image");
      } else if (type === "mask-result") {
        const { mask, width, height, score: s } = data;
        const overlay = document.createElement("canvas");
        const binary = document.createElement("canvas");
        overlay.width = binary.width = width;
        overlay.height = binary.height = height;
        const octx = overlay.getContext("2d");
        const bctx = binary.getContext("2d");
        if (octx && bctx) {
          const overlayData = new ImageData(width, height);
          const binaryData = new ImageData(width, height);
          for (let i = 0; i < mask.length; i++) {
            const idx = i * 4;
            if (mask[i] > 0) {
              overlayData.data[idx] = 167;
              overlayData.data[idx + 1] = 139;
              overlayData.data[idx + 2] = 250;
              overlayData.data[idx + 3] = 220;
              binaryData.data[idx] = 255;
              binaryData.data[idx + 1] = 255;
              binaryData.data[idx + 2] = 255;
              binaryData.data[idx + 3] = 255;
            }
          }
          octx.putImageData(overlayData, 0, 0);
          bctx.putImageData(binaryData, 0, 0);
          maskOverlayRef.current = overlay;
          maskBinaryRef.current = binary;
        }
        setScore(s);
        setProcessing(false);
        setPoints((p) => {
          draw(p, maskOverlayRef.current, maskBinaryRef.current, viewRef.current);
          return p;
        });
      } else if (type === "error") {
        setProcessing(false);
        setStatus(`Error: ${data}`);
      }
    };

    setStatus("Loading SAM2 decoder…");
    worker.postMessage({
      type: "load-decoder",
      data: { modelUrl: DECODER_URL },
    });

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [draw]);

  // Re-load image + embeddings whenever the source changes.
  useEffect(() => {
    setEmbeddingsReady(false);
    setPoints([]);
    setScore(null);
    maskOverlayRef.current = null;
    maskBinaryRef.current = null;

    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = sourceParams.imageUrl;
    img.onload = () => {
      imgRef.current = img;
      draw([], null, null, "edit");
      setStatus("Loading embeddings…");
      workerRef.current?.postMessage({
        type: "load-embeddings",
        data: {
          url: sourceParams.embeddingsUrl,
          imageWidth: sourceParams.imageWidth,
          imageHeight: sourceParams.imageHeight,
        },
      });
    };
  }, [sourceParams, draw]);

  useEffect(() => {
    draw(points, maskOverlayRef.current, maskBinaryRef.current, view);
  }, [view, points, draw]);

  const addPoint = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>, label: 0 | 1) => {
      if (!decoderReady || !embeddingsReady || processing) return;
      if (view !== "edit") setView("edit");
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const sx = canvas.width / rect.width;
      const sy = canvas.height / rect.height;
      const x = Math.round((e.clientX - rect.left) * sx);
      const y = Math.round((e.clientY - rect.top) * sy);
      const next = [...points, { x, y, label }];
      setPoints(next);
      draw(next, maskOverlayRef.current, maskBinaryRef.current, "edit");
      setProcessing(true);
      setStatus("Decoding…");
      workerRef.current?.postMessage({
        type: "decode",
        data: {
          points: next,
          imageWidth: sourceParams.imageWidth,
          imageHeight: sourceParams.imageHeight,
        },
      });
    },
    [decoderReady, embeddingsReady, processing, points, draw, view, sourceParams]
  );

  const clear = useCallback(() => {
    setPoints([]);
    setScore(null);
    maskOverlayRef.current = null;
    maskBinaryRef.current = null;
    setView("edit");
    draw([], null, null, "edit");
    setStatus(embeddingsReady ? "Click anywhere on the image" : status);
  }, [draw, embeddingsReady, status]);

  const download = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sam2-${view}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, "image/png");
  }, [view]);

  // Bring-your-own bundle: drop image + embeddings file.
  const handleFiles = useCallback(async (files: FileList) => {
    const list = Array.from(files);
    const imageFile = list.find((f) => /\.(jpe?g|png|webp)$/i.test(f.name));
    const embFile = list.find((f) => /\.bin$/i.test(f.name));
    const manifestFile = list.find((f) => /manifest\.json$/i.test(f.name));

    if (!imageFile || !embFile) {
      setStatus(
        "BYO needs a preview image (.jpg/.png) and an embedding file (.bin)."
      );
      return;
    }

    let width = 0;
    let height = 0;

    if (manifestFile) {
      try {
        const text = await manifestFile.text();
        const m = JSON.parse(text);
        width = Number(m.originalWidth) || 0;
        height = Number(m.originalHeight) || 0;
      } catch {
        // ignore — we'll fall back to image natural size
      }
    }

    if (!width || !height) {
      const dims = await new Promise<{ w: number; h: number }>((resolve) => {
        const probe = new Image();
        probe.onload = () =>
          resolve({ w: probe.naturalWidth, h: probe.naturalHeight });
        probe.src = URL.createObjectURL(imageFile);
      });
      width = dims.w;
      height = dims.h;
    }

    const objectImageUrl = URL.createObjectURL(imageFile);
    const objectEmbeddingsUrl = URL.createObjectURL(embFile);

    setSource({
      kind: "byo",
      objectImageUrl,
      objectEmbeddingsUrl,
      width,
      height,
    });
  }, []);

  const onDrop = useCallback(
    (e: ReactDragEvent<HTMLElement>) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const ready = decoderReady && embeddingsReady;
  const hasMask = !!maskBinaryRef.current;
  const showCheckerBg = view === "cutout" || view === "erase";

  const modes = [
    { id: "edit", label: "Mask" },
    { id: "cutout", label: "Cutout" },
    { id: "erase", label: "Erase" },
  ] as const;

  return (
    <div className="w-full">
      {/* Source picker */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <p className="mr-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">
          Source
        </p>
        {DEMOS.map((d) => {
          const active =
            source.kind === "demo" && source.demo.id === d.id;
          return (
            <button
              key={d.id}
              type="button"
              onClick={() => setSource({ kind: "demo", demo: d })}
              className={`rounded-full border px-3 py-1 text-[11px] font-semibold transition-colors ${
                active
                  ? "border-violet-500 bg-violet-500/15 text-violet-300"
                  : "border-stone-700 bg-stone-900/60 text-stone-300 hover:bg-stone-800"
              }`}
            >
              {d.label}
            </button>
          );
        })}
        <span className="text-stone-700">·</span>
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={`cursor-pointer rounded-full border px-3 py-1 text-[11px] font-semibold transition-colors ${
            dragOver
              ? "border-violet-400 bg-violet-500/20 text-violet-200"
              : source.kind === "byo"
                ? "border-violet-500 bg-violet-500/15 text-violet-300"
                : "border-dashed border-stone-600 bg-stone-900/40 text-stone-300 hover:bg-stone-800"
          }`}
        >
          <input
            type="file"
            multiple
            accept=".bin,.json,image/*"
            className="hidden"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
          />
          {source.kind === "byo" ? "✓ Your bundle" : "Drop your bundle"}
        </label>
        {source.kind === "byo" && (
          <button
            type="button"
            onClick={() => setSource({ kind: "demo", demo: DEMOS[0] })}
            className="text-[10px] text-stone-500 hover:text-stone-300"
          >
            clear
          </button>
        )}
      </div>

      <div className="overflow-hidden rounded-2xl border border-stone-800 bg-stone-900/40 backdrop-blur-sm">
        <div className="grid grid-cols-[170px_1fr] sm:grid-cols-[200px_1fr]">
          {/* Controls */}
          <aside className="relative flex flex-col gap-4 border-r border-stone-800 bg-stone-900/40 p-4">
            <div>
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">
                SAM2
              </p>
              <p className="mt-0.5 text-[11px] leading-snug text-stone-400">
                Left-click → include · Right-click → exclude
              </p>
            </div>

            <div className="flex flex-col gap-1.5">
              <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-stone-500">
                View
              </p>
              {modes.map((m) => {
                const disabled = !hasMask && m.id !== "edit";
                const isActive = view === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    disabled={disabled}
                    onClick={() => setView(m.id)}
                    className={`rounded-md border px-2.5 py-1.5 text-left text-[12px] font-semibold transition-colors ${
                      isActive
                        ? "border-white bg-white text-stone-900"
                        : "border-stone-700 bg-stone-900/70 text-stone-200 hover:bg-stone-800"
                    } disabled:cursor-not-allowed disabled:opacity-40`}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>

            <div className="mt-auto flex flex-col gap-1.5">
              <button
                type="button"
                onClick={download}
                disabled={!ready}
                className="inline-flex items-center justify-center gap-1.5 rounded-md border border-stone-700 bg-stone-900/70 px-2.5 py-1.5 text-[12px] font-semibold text-stone-200 transition-colors hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Download PNG
              </button>
              <button
                type="button"
                onClick={clear}
                disabled={points.length === 0 && !hasMask}
                className="rounded-md border border-stone-700 bg-stone-900/70 px-2.5 py-1.5 text-[12px] font-semibold text-stone-200 transition-colors hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Reset
              </button>
            </div>

            <div className="-mx-4 -mb-4 border-t border-stone-800 bg-stone-950/40 px-4 pb-3 pt-3 text-[10px] leading-snug text-stone-500">
              <div className="flex items-center gap-1.5">
                <span
                  className={`inline-block h-1.5 w-1.5 rounded-full ${
                    ready
                      ? processing
                        ? "animate-pulse bg-violet-400"
                        : "bg-emerald-400"
                      : "animate-pulse bg-stone-500"
                  }`}
                />
                <span className="truncate font-semibold text-stone-300">
                  {processing ? "Decoding…" : status}
                </span>
              </div>
              {score !== null && !processing && (
                <div className="mt-1.5 tabular-nums text-violet-300">
                  confidence {(score * 100).toFixed(1)}%
                </div>
              )}
            </div>
          </aside>

          {/* Canvas */}
          <div className="relative p-3">
            <div
              className={`relative w-full overflow-hidden rounded-xl ${
                showCheckerBg ? "bg-checker" : "bg-stone-950/40"
              }`}
              style={{
                aspectRatio: `${sourceParams.imageWidth} / ${sourceParams.imageHeight}`,
              }}
            >
              <canvas
                ref={canvasRef}
                onClick={(e) => addPoint(e, 1)}
                onContextMenu={(e) => {
                  e.preventDefault();
                  addPoint(e, 0);
                }}
                className={`absolute inset-0 h-full w-full select-none ${
                  ready && view === "edit"
                    ? "cursor-crosshair"
                    : "cursor-default"
                }`}
              />
              {!ready && (
                <div className="absolute inset-0 grid place-items-center bg-stone-950/60 backdrop-blur-sm">
                  <div className="flex flex-col items-center gap-2 text-sm text-stone-400">
                    <svg
                      className="h-5 w-5 animate-spin text-violet-400"
                      viewBox="0 0 24 24"
                      fill="none"
                    >
                      <circle
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeOpacity="0.2"
                        strokeWidth="3"
                      />
                      <path
                        d="M22 12a10 10 0 0 1-10 10"
                        stroke="currentColor"
                        strokeWidth="3"
                        strokeLinecap="round"
                      />
                    </svg>
                    <span className="text-xs">{status}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-stone-500">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-400" />
          left-click — include
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-rose-500" />
          right-click — exclude
        </span>
        <span className="text-stone-700">·</span>
        <span>ONNX · WASM · pre-encoded embeddings</span>
      </div>
    </div>
  );
}
