/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Allow the worker module on jsdelivr CDN to load via dynamic import.
  // (We fetch onnxruntime-web there to keep the main bundle tiny.)
};

export default nextConfig;
