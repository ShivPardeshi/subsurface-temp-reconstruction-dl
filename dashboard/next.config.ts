import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/products/:path*",
        destination: "http://127.0.0.1:8000/products/:path*",
      },
    ];
  },
};

export default nextConfig;
