/** @type {import('next').NextConfig} */
const nextConfig = {
  // Next.js 16 removed appIsrStatus/buildActivity. false hides the dev "N" badge.
  devIndicators: false,
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
