'use client';

import React, { useState } from 'react';
import { HeatwaveGridResponse } from '../lib/types';
import { AlertCircle, Eye, Info } from 'lucide-react';

interface HeatwaveMapViewProps {
  gridData: HeatwaveGridResponse;
  selectedLat: number;
  selectedLon: number;
  onSelectCoordinate?: (lat: number, lon: number) => void;
  title?: string;
}

export function HeatwaveMapView({
  gridData,
  selectedLat,
  selectedLon,
  onSelectCoordinate,
  title = 'Spatial Marine Heatwave Event Tracking (Hobday et al., 2016)',
}: HeatwaveMapViewProps) {
  const [hoveredCell, setHoveredCell] = useState<{
    lat: number;
    lon: number;
    cat: number;
  } | null>(null);

  const { rows, cols } = gridData.dimensions;
  const lats = gridData.latitudes;
  const lons = gridData.longitudes;

  // Category labels and color map
  const getCategoryColor = (cat: number, isOcean: boolean) => {
    if (!isOcean) return '#1e293b'; // Slate land
    switch (cat) {
      case 1:
        return '#f59e0b'; // Amber - Moderate
      case 2:
        return '#ea580c'; // Orange-red - Strong
      case 3:
        return '#dc2626'; // Crimson - Severe
      case 4:
        return '#7f1d1d'; // Dark Red - Extreme
      default:
        return '#091e3a'; // Deep Navy - Normal Ocean
    }
  };

  const getCategoryName = (cat: number) => {
    switch (cat) {
      case 1:
        return 'Category I (Moderate)';
      case 2:
        return 'Category II (Strong)';
      case 3:
        return 'Category III (Severe)';
      case 4:
        return 'Category IV (Extreme)';
      default:
        return 'Normal Climatology';
    }
  };

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-200">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
          <p className="text-xs text-slate-400">
            North Indian Ocean (2–30°N, 45–105°E) • Date: {gridData.date}
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="rounded bg-rose-950/80 px-2 py-0.5 font-mono text-rose-400 border border-rose-800/60">
            {gridData.summary.mhw_area_percentage}% Basin Affected
          </span>
          <span className="text-slate-400 font-mono">
            {gridData.summary.active_mhw_cells} Active Cells
          </span>
        </div>
      </div>

      {/* SVG Interactive Map Grid */}
      <div className="relative overflow-hidden rounded border border-slate-800 bg-slate-950 p-2">
        <svg
          viewBox={`0 0 ${cols * 10} ${rows * 10}`}
          className="w-full h-[320px] select-none"
        >
          {gridData.categories.map((row, rIdx) => {
            // Inverted for display (high latitude at the top)
            const rDisplay = rows - 1 - rIdx;
            const lat = lats[rIdx];

            return row.map((cat, cIdx) => {
              const lon = lons[cIdx];
              const isOcean = gridData.ocean_mask[rIdx][cIdx];
              const fill = getCategoryColor(cat, isOcean);

              const isSelected =
                Math.abs(selectedLat - lat) < 1.0 &&
                Math.abs(selectedLon - lon) < 1.0;

              return (
                <rect
                  key={`${rIdx}-${cIdx}`}
                  x={cIdx * 10}
                  y={rDisplay * 10}
                  width={9.5}
                  height={9.5}
                  fill={fill}
                  stroke={isSelected ? '#38bdf8' : 'none'}
                  strokeWidth={isSelected ? 2 : 0}
                  className="cursor-pointer transition-opacity hover:opacity-80"
                  onMouseEnter={() => setHoveredCell({ lat, lon, cat })}
                  onMouseLeave={() => setHoveredCell(null)}
                  onClick={() => onSelectCoordinate && onSelectCoordinate(lat, lon)}
                />
              );
            });
          })}

          {/* Selected Crosshair Pin */}
          {(() => {
            const latIdx = lats.findIndex((l) => Math.abs(l - selectedLat) < 1.0);
            const lonIdx = lons.findIndex((l) => Math.abs(l - selectedLon) < 1.0);
            if (latIdx >= 0 && lonIdx >= 0) {
              const cx = lonIdx * 10 + 5;
              const cy = (rows - 1 - latIdx) * 10 + 5;
              return (
                <g>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={6}
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    className="animate-pulse"
                  />
                  <circle cx={cx} cy={cy} r={2} fill="#38bdf8" />
                </g>
              );
            }
            return null;
          })()}
        </svg>

        {/* Dynamic Tooltip on Hover */}
        {hoveredCell && (
          <div className="absolute top-3 right-3 rounded bg-slate-900/95 border border-slate-700 px-3 py-1.5 text-xs font-mono text-slate-200 shadow-lg">
            <span>
              {hoveredCell.lat.toFixed(1)}°N, {hoveredCell.lon.toFixed(1)}°E
            </span>
            <span className="block font-sans text-[11px] text-slate-400">
              {getCategoryName(hoveredCell.cat)}
            </span>
          </div>
        )}
      </div>

      {/* Legend & Footnote */}
      <div className="mt-3.5 flex flex-wrap items-center justify-between gap-2 border-t border-slate-800/80 pt-2.5 text-xs text-slate-400">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-slate-500 font-medium">Categories:</span>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-[#091e3a] border border-slate-700" />
            <span className="text-[11px]">Normal</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-[#f59e0b]" />
            <span className="text-[11px]">Cat I (Mod)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-[#ea580c]" />
            <span className="text-[11px]">Cat II (Strong)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-[#dc2626]" />
            <span className="text-[11px]">Cat III (Severe)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-sm bg-[#1e293b]" />
            <span className="text-[11px]">Land</span>
          </div>
        </div>

        <span className="text-[11px] text-slate-500">
          Click any grid square to inspect vertical profile sounding
        </span>
      </div>
    </div>
  );
}
