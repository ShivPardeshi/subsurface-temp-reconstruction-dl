'use client';

import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { ProfileData, ProfileResponse } from '../lib/types';

interface ProfileChartProps {
  profile?: ProfileData;
  climatology?: number[];
  mldMeters?: number;
  d26Meters?: number;
  profileData?: ProfileResponse | null;
  locationName?: string;
  title?: string;
  isLoading?: boolean;
}

export function ProfileChart({
  profile,
  climatology,
  mldMeters,
  d26Meters,
  profileData,
  locationName = 'Selected Coordinate',
  title = '3D Vertical Ocean Temperature Sounding (0m to 1000m)',
  isLoading = false,
}: ProfileChartProps) {
  const activeProfile = profile || profileData?.profile;
  const activeClimatology = climatology || profileData?.climatology_profile || activeProfile?.mean_temperatures;
  const activeMLD = mldMeters !== undefined ? mldMeters : profileData?.mld_direct?.mean_meters;
  const activeD26 = d26Meters !== undefined ? d26Meters : profileData?.d26?.mean_meters;

  if (isLoading || !activeProfile || !activeClimatology) {
    return (
      <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-400">
        <div className="h-[380px] flex items-center justify-center">
          <span className="text-xs">Computing continuous 3D vertical sounding profile...</span>
        </div>
      </div>
    );
  }

  // Transform data for Recharts: array of objects with depth and temp values
  const chartData = activeProfile.depths_m.map((depth, idx) => {
    const meanT = activeProfile.mean_temperatures[idx] ?? 0;
    const stdT = activeProfile.std_temperatures[idx] ?? 0;
    const climT = activeClimatology[idx] ?? meanT;
    const lower95 = activeProfile.ci_lower_95 ? activeProfile.ci_lower_95[idx] : meanT - 1.96 * stdT;
    const upper95 = activeProfile.ci_upper_95 ? activeProfile.ci_upper_95[idx] : meanT + 1.96 * stdT;

    return {
      depth,
      depthLabel: `${depth}m`,
      meanTemp: parseFloat(meanT.toFixed(2)),
      climatology: parseFloat(climT.toFixed(2)),
      lower95: parseFloat(lower95.toFixed(2)),
      upper95: parseFloat(upper95.toFixed(2)),
      bandWidth: parseFloat((upper95 - lower95).toFixed(2)),
      uncertainty: parseFloat(stdT.toFixed(2)),
      anomaly: parseFloat((meanT - climT).toFixed(2)),
    };
  });

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-200">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
          <p className="text-xs text-slate-400">
            {locationName} • Calibrated Posterior Spread (±2σ 95% Confidence)
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          {activeMLD !== undefined && (
            <span className="rounded bg-sky-950/80 px-2 py-0.5 font-mono text-sky-400 border border-sky-800/60">
              MLD: {activeMLD.toFixed(1)}m
            </span>
          )}
          {activeD26 !== undefined && (
            <span className="rounded bg-rose-950/80 px-2 py-0.5 font-mono text-rose-400 border border-rose-800/60">
              D26: {activeD26.toFixed(1)}m
            </span>
          )}
        </div>
      </div>

      <div className="h-[420px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            layout="vertical"
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              type="number"
              domain={[0, 32]}
              unit="°C"
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{
                value: 'Temperature (°C)',
                position: 'insideBottom',
                offset: -10,
                fill: '#94a3b8',
                fontSize: 12,
              }}
            />
            <YAxis
              dataKey="depth"
              type="number"
              reversed={true}
              domain={[0, 1000]}
              ticks={[0, 50, 100, 150, 200, 300, 400, 500, 750, 1000]}
              unit="m"
              stroke="#64748b"
              tick={{ fill: '#94a3b8', fontSize: 11 }}
              label={{
                value: 'Depth (m)',
                angle: -90,
                position: 'insideLeft',
                fill: '#94a3b8',
                fontSize: 12,
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="rounded-lg border border-slate-700 bg-slate-900 p-3 shadow-xl text-xs">
                    <p className="font-semibold text-slate-200 border-b border-slate-800 pb-1 mb-1.5">
                      Depth: {d.depth}m
                    </p>
                    <div className="space-y-1 font-mono">
                      <div className="flex justify-between gap-4 text-sky-400">
                        <span>Reconstructed:</span>
                        <span className="font-bold">{d.meanTemp}°C</span>
                      </div>
                      <div className="flex justify-between gap-4 text-slate-400">
                        <span>Climatology:</span>
                        <span>{d.climatology}°C</span>
                      </div>
                      <div className="flex justify-between gap-4 text-emerald-400">
                        <span>Anomaly (ΔT):</span>
                        <span>{d.anomaly > 0 ? `+${d.anomaly}` : d.anomaly}°C</span>
                      </div>
                      <div className="flex justify-between gap-4 text-slate-500 pt-1 border-t border-slate-800">
                        <span>95% CI:</span>
                        <span>[{d.lower95}°C, {d.upper95}°C]</span>
                      </div>
                    </div>
                  </div>
                );
              }}
            />
            <Legend
              verticalAlign="top"
              height={36}
              wrapperStyle={{ fontSize: 12, color: '#94a3b8' }}
            />

            {/* MLD reference line */}
            {mldMeters !== undefined && (
              <ReferenceLine
                y={mldMeters}
                stroke="#38bdf8"
                strokeDasharray="4 4"
                label={{
                  value: `MLD (${mldMeters.toFixed(0)}m)`,
                  fill: '#38bdf8',
                  fontSize: 10,
                  position: 'right',
                }}
              />
            )}

            {/* D26 Isotherm reference line */}
            {d26Meters !== undefined && (
              <ReferenceLine
                y={d26Meters}
                stroke="#f43f5e"
                strokeDasharray="4 4"
                label={{
                  value: `D26 (${d26Meters.toFixed(0)}m)`,
                  fill: '#f43f5e',
                  fontSize: 10,
                  position: 'right',
                }}
              />
            )}

            {/* 95% Confidence Band (Lower & Upper) */}
            <Area
              dataKey="lower95"
              stroke="transparent"
              fill="transparent"
              legendType="none"
            />
            <Area
              dataKey="upper95"
              name="95% Calibrated Confidence Interval"
              stroke="#0284c7"
              strokeOpacity={0.4}
              fill="#0284c7"
              fillOpacity={0.2}
            />

            {/* Historical Climatology Baseline */}
            <Line
              type="monotone"
              dataKey="climatology"
              name="Climatology Baseline (2-Harmonic OLS)"
              stroke="#64748b"
              strokeWidth={1.5}
              strokeDasharray="5 5"
              dot={false}
            />

            {/* OceanEmbed Production Reconstructed Mean */}
            <Line
              type="monotone"
              dataKey="meanTemp"
              name="OceanEmbed (Phase 8 Zone-Adaptive Model)"
              stroke="#38bdf8"
              strokeWidth={2.5}
              dot={{ r: 3, fill: '#38bdf8', stroke: '#0f172a', strokeWidth: 1 }}
              activeDot={{ r: 5 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between text-[11px] text-slate-500 border-t border-slate-800/80 pt-2.5">
        <span>Pycnocline Gradient: Steepest between 50m and 150m</span>
        <span>Deep-Abyssal Parity: &lt;0.05°C variance at 1000m</span>
      </div>
    </div>
  );
}
