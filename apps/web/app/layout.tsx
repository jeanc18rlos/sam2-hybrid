import type { Metadata, Viewport } from "next";
import "./globals.css";

const SITE_URL = "https://sam2-hybrid.vercel.app";
const TITLE = "SAM2 Hybrid · in-browser segmentation";
const DESCRIPTION =
  "Click anywhere on the image and watch a 224M-parameter vision transformer respond in 30–60 ms. Decoder runs in your tab. Encoder runs separately on your machine via the companion notebook.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: TITLE,
  description: DESCRIPTION,
  applicationName: "sam2-hybrid",
  keywords: [
    "sam2",
    "segment-anything-2",
    "onnx",
    "webgpu",
    "interactive segmentation",
    "browser ml",
  ],
  openGraph: {
    type: "website",
    title: TITLE,
    description: DESCRIPTION,
    url: SITE_URL,
    siteName: "SAM2 Hybrid",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0a0a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
