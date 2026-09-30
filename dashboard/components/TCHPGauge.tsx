'use client';

import React from 'react';
import { Flame, AlertTriangle, Waves, Layers, Thermometer } from 'lucide-react';
import { TCHPResponse } from '../lib/types';

interface TCHPGaugeProps {
  data: TCHPResponse | null;
  className?: string;
  isLoading?: boolean;
}

export function TCHPGauge({ data, className = '', isLoading = false }: TCHPGaugeProps) {
  if (isLoading || !data) {
    return (
      <div className={`rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-400 ${className}`}>
        <div className="h-[380px] flex items-center justify-center">
          <span className="text-xs">Computing Tropical Cyclone Heat Potential (TCHP)...</span>
        </div>
      </div>
    );
  }

  const tchpMean = data.tchp.mean_kj_cm2 ?? data.tchp.mean;
  const tchpStd = data.tchp.std_kj_cm2 ?? data.tchp.std;
  const risk = data.risk_assessment;
  const d26Mean = data.d26.mean_meters ?? data.d26.mean;
  const d26Std = data.d26.std_meters ?? data.d26.std;
  const mldMean = data.mld_direct.mean_meters ?? data.mld_direct.mean;
  const ohcValGj = (data.ohc_700.mean_jm2 ?? data.ohc_700.mean) / 1e9;


  // Risk styling
  const riskBorderColor =
    risk.badge_color === 'red'
      ? 'border-red-600 bg-red-950/30'
      : risk.badge_color === 'yellow'
      ? 'border-amber-600 bg-amber-950/30'
      : 'border-emerald-600 bg-emerald-950/30';

  const riskTextColor =
    risk.badge_color === 'red'
      ? 'text-red-400'
      : risk.badge_color === 'yellow'
      ? 'text-amber-400'
      : 'text-emerald-400';

  // Percentage for the gauge bar (0 to 120 kJ/cm2 scale)
  const gaugePercent = Math.min(Math.max((tchpMean / 120) * 100, 0), 100);

  return (
    <div
      className={`rounded-lg border border-slate-800 bg-slate-900/90 p-5 text-slate-200 ${className}`}
    >
      <div className="mb-4 flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <Flame className="h-5 w-5 text-rose-500" />
          <h3 className="text-sm font-semibold text-slate-100">
            Tropical Cyclone Heat Potential (TCHP)
          </h3>
        </div>
        <span className="text-xs font-mono text-slate-400">
          Target: D26 Isotherm Integral
        </span>
      </div>

      {/* Main TCHP Big Metric */}
      <div className="mb-5 rounded-lg border border-slate-800 bg-slate-950/60 p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div>
            <span className="text-xs text-slate-400">Heat Reservoir Energy</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-3xl font-bold font-mono text-slate-100">
                {tchpMean.toFixed(1)}
              </span>
              <span className="text-sm font-mono text-slate-400">
                ± {tchpStd.toFixed(1)} kJ/cm²
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="text-[11px] text-slate-500 block">95% Calibrated Interval</span>
            <span className="font-mono text-xs text-slate-300">
              [{data.tchp.ci_lower_95.toFixed(1)} — {data.tchp.ci_upper_95.toFixed(1)} kJ/cm²]
            </span>
          </div>
        </div>

        {/* Visual Progress / Risk Gauge Bar */}
        <div className="mt-3">
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full transition-all duration-500 ${
                tchpMean > 80
                  ? 'bg-rose-500'
                  : tchpMean >= 50
                  ? 'bg-amber-500'
                  : 'bg-emerald-500'
              }`}
              style={{ width: `${gaugePercent}%` }}
            />
          </div>
          <div className="mt-1.5 flex justify-between text-[10px] font-mono text-slate-500">
            <span>0 kJ/cm² (Low)</span>
            <span>50 (Moderate)</span>
            <span>80 (Rapid Intensification)</span>
            <span>120+</span>
          </div>
        </div>
      </div>

      {/* Cyclogenesis Risk Banner */}
      <div className={`mb-5 rounded-lg border p-3.5 ${riskBorderColor}`}>
        <div className="flex items-start gap-2.5">
          <AlertTriangle className={`mt-0.5 h-4 w-4 shrink-0 ${riskTextColor}`} />
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-slate-200">
                Cyclogenesis Intensity Risk:
              </span>
              <span className={`text-xs font-bold ${riskTextColor}`}>
                {risk.level}
              </span>
            </div>
            <p className="mt-1 text-xs text-slate-300 leading-relaxed">
              {risk.description}
            </p>
          </div>
        </div>
      </div>

      {/* Supporting Disaster Metric Cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* D26 Isotherm */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Layers className="h-3.5 w-3.5 text-rose-400" />
            <span>D26 Depth</span>
          </div>
          <p className="mt-1.5 text-base font-bold font-mono text-slate-100">
            {d26Mean.toFixed(1)}m
          </p>
          <span className="text-[10px] text-slate-500 font-mono">
            ±{d26Std.toFixed(1)}m spread
          </span>
        </div>

        {/* Ocean Heat Content 700m */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Flame className="h-3.5 w-3.5 text-amber-400" />
            <span>OHC (0-700m)</span>
          </div>
          <p className="mt-1.5 text-base font-bold font-mono text-slate-100">
            {ohcValGj.toFixed(1)}
          </p>
          <span className="text-[10px] text-slate-500 font-mono">GJ/m²</span>
        </div>

        {/* Mixed Layer Depth */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Waves className="h-3.5 w-3.5 text-sky-400" />
            <span>Direct MLD</span>
          </div>
          <p className="mt-1.5 text-base font-bold font-mono text-slate-100">
            {mldMean.toFixed(1)}m
          </p>
          <span className="text-[10px] text-slate-500 font-mono">ΔT=0.2°C threshold</span>
        </div>

        {/* Sea Surface Temp */}
        <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-400">
            <Thermometer className="h-3.5 w-3.5 text-emerald-400" />
            <span>Surface SST</span>
          </div>
          <p className="mt-1.5 text-base font-bold font-mono text-slate-100">
            {data.sst.toFixed(2)}°C
          </p>
          <span className="text-[10px] text-slate-500 font-mono">Top sounding level</span>
        </div>
      </div>
    </div>
  );
}
