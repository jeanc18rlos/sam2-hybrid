import Link from "next/link";
import Segmenter from "@/components/Segmenter";

export default function Home() {
  return (
    <main className="mx-auto max-w-5xl px-6 pb-24 pt-12 lg:px-8 lg:pt-20">
      <header className="mb-12">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-red-300">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-400" />
          sam2-hybrid · v0.1
        </div>
        <h1 className="mt-3 text-balance text-4xl font-bold leading-[1.05] tracking-tight text-white sm:text-5xl">
          Click anywhere on the image. The model is in your tab.
        </h1>
        <p className="mt-5 max-w-2xl text-pretty text-lg text-stone-400">
          The full <code className="rounded bg-stone-800 px-1.5 py-0.5 text-[0.92em] text-stone-200">sam2.1_hiera_large</code> decoder
          (224M parameters), running in the browser via{" "}
          <code className="rounded bg-stone-800 px-1.5 py-0.5 text-[0.92em] text-stone-200">onnxruntime-web</code>. Encoder
          ran separately on a notebook, embedding shipped down once. Every
          click after that is local inference, no server round-trip.
        </p>

        <div className="mt-7 flex flex-wrap items-center gap-2">
          <a
            href="https://jeanrojas.com/blog/splitting-sam2-encoder-decoder"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg bg-red-500 px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-400"
          >
            Read the full write-up
            <Arrow />
          </a>
          <a
            href="https://github.com/jeanc18rlos/sam2-hybrid"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm font-semibold text-stone-200 transition-colors hover:bg-stone-800"
          >
            <GithubIcon />
            GitHub
          </a>
          <a
            href="https://colab.research.google.com/github/jeanc18rlos/sam2-hybrid/blob/main/notebooks/sam2_encode.ipynb"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm font-semibold text-stone-200 transition-colors hover:bg-stone-800"
          >
            <ColabIcon />
            Encode your image — Colab
          </a>
        </div>
      </header>

      <Segmenter />

      <section className="mt-16 grid grid-cols-1 gap-6 text-sm text-stone-400 sm:grid-cols-3">
        <Card title="One encode, many clicks">
          The encoder is heavy and runs <em>once</em> per image. The
          decoder is tiny (16&nbsp;MB) and runs once per click. Two halves
          of the same model on two halves of the network.
        </Card>
        <Card title="Image stays local">
          You only ship 16&nbsp;MB of float16 features down to the
          browser. The original photo never leaves your machine — useful
          for privacy-sensitive workflows.
        </Card>
        <Card title="Bring your own bundle">
          Run the{" "}
          <Link
            href="https://colab.research.google.com/github/jeanc18rlos/sam2-hybrid/blob/main/notebooks/sam2_encode.ipynb"
            className="underline hover:text-stone-200"
            target="_blank"
            rel="noopener noreferrer"
          >
            Colab notebook
          </Link>{" "}
          to encode any image, then drop the resulting{" "}
          <code className="text-stone-200">embedding.bin</code> +{" "}
          <code className="text-stone-200">manifest.json</code> +{" "}
          <code className="text-stone-200">preview.jpg</code> onto the
          source picker above.
        </Card>
      </section>

      <footer className="mt-16 flex flex-col items-start justify-between gap-4 border-t border-stone-800 pt-8 text-xs text-stone-500 sm:flex-row sm:items-center">
        <p>
          MIT-licensed. Code at{" "}
          <a
            href="https://github.com/jeanc18rlos/sam2-hybrid"
            className="underline hover:text-stone-300"
            target="_blank"
            rel="noopener noreferrer"
          >
            github.com/jeanc18rlos/sam2-hybrid
          </a>
          . Companion article on{" "}
          <a
            href="https://jeanrojas.com/blog/splitting-sam2-encoder-decoder"
            className="underline hover:text-stone-300"
            target="_blank"
            rel="noopener noreferrer"
          >
            jeanrojas.com
          </a>
          .
        </p>
        <p className="tabular-nums">SAM2 weights © Meta · Demo image: ClickSEG (Apache 2.0)</p>
      </footer>
    </main>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-stone-800 bg-stone-900/40 p-5">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-red-300">
        {title}
      </h3>
      <p className="text-stone-300">{children}</p>
    </div>
  );
}

function Arrow() {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 17 17 7" />
      <path d="M9 7h8v8" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor">
      <path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.18-3.37-1.18-.46-1.16-1.12-1.47-1.12-1.47-.91-.62.07-.6.07-.6 1 .07 1.53 1.04 1.53 1.04.9 1.53 2.36 1.09 2.94.83.09-.65.35-1.09.63-1.34-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.99 1.03-2.69-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02a9.6 9.6 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.6 1.03 2.69 0 3.84-2.34 4.69-4.57 4.93.36.31.68.92.68 1.85v2.74c0 .27.18.58.69.48A10 10 0 0 0 12 2Z" />
    </svg>
  );
}

function ColabIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="7" cy="12" r="4" />
      <circle cx="17" cy="12" r="4" />
      <path d="M11 8v8" />
    </svg>
  );
}
