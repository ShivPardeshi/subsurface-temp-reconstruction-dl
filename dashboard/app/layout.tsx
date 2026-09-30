import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';
import {
  Activity,
  Layers,
  Flame,
  Thermometer,
  ShieldCheck,
  Compass,
} from 'lucide-react';

export const metadata: Metadata = {
  title: 'OceanEmbed: Operational Disaster Products Dashboard',
  description:
    '3D Subsurface Ocean Temperature Reconstruction & Multi-Disaster Management System for the North Indian Ocean.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="flex min-h-screen bg-[#090d16] font-sans text-slate-100 antialiased">
        {/* Fixed Uncodixified Left Sidebar (255px) */}
        <aside className="fixed inset-y-0 left-0 z-30 flex w-[255px] flex-col border-r border-slate-800/90 bg-[#0c1220]">
          {/* Brand Header */}
          <div className="flex h-14 items-center gap-2.5 border-b border-slate-800/90 px-4">
            <div className="flex h-7 w-7 items-center justify-center rounded bg-sky-600 font-mono text-sm font-bold text-white">
              OE
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight text-slate-100">
                OceanEmbed
              </h1>
              <p className="text-[10px] font-mono text-slate-400">
                SIH PS26066 • Disaster Track
              </p>
            </div>
          </div>

          {/* Nav Links */}
          <nav className="flex-1 space-y-1 p-3">
            <div className="px-2 py-1 text-[11px] font-medium uppercase tracking-wider text-slate-500">
              Operations & Products
            </div>

            <Link
              href="/"
              className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800/70 hover:text-slate-100 transition-colors"
            >
              <Activity className="h-4 w-4 text-sky-400" />
              <span>Executive Overview</span>
            </Link>

            <Link
              href="/profile-viewer"
              className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800/70 hover:text-slate-100 transition-colors"
            >
              <Layers className="h-4 w-4 text-cyan-400" />
              <span>3D Profile Sounding (0-1000m)</span>
            </Link>

            <Link
              href="/heatwave-map"
              className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800/70 hover:text-slate-100 transition-colors"
            >
              <Thermometer className="h-4 w-4 text-amber-400" />
              <span>Marine Heatwave Map</span>
            </Link>

            <Link
              href="/tchp-gauge"
              className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800/70 hover:text-slate-100 transition-colors"
            >
              <Flame className="h-4 w-4 text-rose-500" />
              <span>Cyclone Heat Potential (TCHP)</span>
            </Link>

            <Link
              href="/validation"
              className="flex items-center gap-2.5 rounded-md px-2.5 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800/70 hover:text-slate-100 transition-colors"
            >
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              <span>In Situ Cruise Validation (GO-SHIP)</span>
            </Link>
          </nav>

          {/* Model Status & System Health Footer */}
          <div className="border-t border-slate-800/90 p-3.5 bg-slate-950/40">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[11px] font-medium text-slate-300">
                Production Backend Online
              </span>
            </div>
            <p className="mt-1 font-mono text-[10px] text-slate-500">
              Model: Phase 8 Calibrated
            </p>
            <p className="font-mono text-[10px] text-slate-500">
              Domain: North Indian Ocean (2–30°N)
            </p>
          </div>
        </aside>

        {/* Main Content Area */}
        <div className="flex flex-1 flex-col pl-[255px]">
          {/* Top Operational Bar */}
          <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-slate-800/90 bg-[#090d16]/95 px-6 backdrop-blur-sm">
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1.5 font-medium text-slate-200">
                <Compass className="h-3.5 w-3.5 text-sky-400" />
                Operational Hydrography System
              </span>
              <span>•</span>
              <span className="font-mono text-slate-400">
                Grid: 0.25° Resolution (112 x 240 x 15)
              </span>
            </div>

            <div className="flex items-center gap-2.5 text-xs">
              <div className="flex items-center gap-1.5 rounded bg-slate-800/80 px-2.5 py-1 font-mono text-[11px] text-slate-300 border border-slate-700/60">
                <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                <span>ECE: 0.0161</span>
              </div>
            </div>
          </header>

          {/* Page Content */}
          <main className="flex-1 p-6">{children}</main>
        </div>
      </body>
    </html>
  );
}
