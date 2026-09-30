'use client';

import React from 'react';
import { Waves, Shield, Thermometer, ArrowDownUp, Info } from 'lucide-react';
import { StratificationResponse } from '../lib/types';

interface StratificationCardProps {
  data: StratificationResponse | null;
  isLoading?: boolean;
}

export function StratificationCard({ data, isLoading = false }: StratificationCardProps) {
  if (isLoading || !data) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-400">
        <div className="h-40 flex items-center justify-center">
          <span className="text-xs">Computing ocean stratification &amp; layer heat content...</span>
        </div>
      </div>
    );
  }

  const { layer_ohc_gj_m2, thermocline, mixed_layer, tchp_kj_cm2 } = data;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-200">
      <div className="mb-3 flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div>
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Waves className="h-4 w-4 text-cyan-400" />
            Upper Ocean Stratification &amp; Multi-Layer Heat Content
          </h3>
          <p className="text-xs text-slate-400">
            Barrier layer dynamics, $N^2$ buoyancy frequency, and depth-integrated heat content
          </p>
        </div>
        <span className="rounded bg-cyan-950/80 px-2 py-0.5 font-mono text-xs text-cyan-400 border border-cyan-800/60">
          Physics Engine
        </span>
      </div>

      {/* Grid of Key Oceanographic Metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* OHC 0-100m */}
        <div className="rounded border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>OHC (0–100m)</span>
            <Thermometer className="h-3.5 w-3.5 text-amber-400" />
          </div>
          <div className="font-mono text-base font-semibold text-amber-300">
            {layer_ohc_gj_m2.ohc_100m} <span className="text-xs font-normal text-slate-400">GJ/m²</span>
          </div>
          <span className="text-[10px] text-slate-500">Surface Mixed Reservoir</span>
        </div>

        {/* OHC 0-300m */}
        <div className="rounded border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>OHC (0–300m)</span>
            <Thermometer className="h-3.5 w-3.5 text-orange-400" />
          </div>
          <div className="font-mono text-base font-semibold text-orange-300">
            {layer_ohc_gj_m2.ohc_300m} <span className="text-xs font-normal text-slate-400">GJ/m²</span>
          </div>
          <span className="text-[10px] text-slate-500">Thermocline Reservoir</span>
        </div>

        {/* Barrier Layer Thickness */}
        <div className="rounded border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Barrier Layer (BLT)</span>
            <Shield className="h-3.5 w-3.5 text-emerald-400" />
          </div>
          <div className="font-mono text-base font-semibold text-emerald-300">
            {mixed_layer.barrier_layer_thickness_m} <span className="text-xs font-normal text-slate-400">m</span>
          </div>
          <span className="text-[10px] text-slate-500">ILD ({mixed_layer.isothermal_layer_depth_m}m) - MLD ({mixed_layer.mld_direct_m.toFixed(1)}m)</span>
        </div>

        {/* Thermocline Gradient */}
        <div className="rounded border border-slate-800 bg-slate-950/70 p-2.5">
          <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
            <span>Max Gradient (dT/dz)</span>
            <ArrowDownUp className="h-3.5 w-3.5 text-sky-400" />
          </div>
          <div className="font-mono text-base font-semibold text-sky-300">
            {thermocline.max_gradient_c_per_m} <span className="text-xs font-normal text-slate-400">°C/m</span>
          </div>
          <span className="text-[10px] text-slate-500">Peak at {thermocline.depth_m.toFixed(0)}m Depth</span>
        </div>
      </div>

      {/* Physics Interpretability Footnote */}
      <div className="mt-3 rounded border border-slate-800/80 bg-slate-950/40 p-2.5 text-xs text-slate-400 flex items-start gap-2">
        <Info className="h-4 w-4 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-medium text-slate-300">Ocean Stability Context: </span>
          {mixed_layer.barrier_layer_thickness_m > 5.0
            ? 'A thick barrier layer is detected. This inhibits cyclone-induced vertical mixing and entrainment of cold sub-thermocline water, sustaining intense sea surface temperatures.'
            : 'Normal or thin barrier layer. Upper-ocean mixing and wind-driven entrainment can cool surface layers during sustained wind stress.'}
        </div>
      </div>
    </div>
  );
}
