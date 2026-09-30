'use client';

import React, { useState } from 'react';
import { MapPin, Calendar, Compass, RefreshCw, Zap } from 'lucide-react';

interface DateRegionSelectorProps {
  selectedLat: number;
  selectedLon: number;
  selectedDate: string;
  onLocationChange: (lat: number, lon: number) => void;
  onDateChange: (date: string) => void;
  onRefresh?: () => void;
  isLoading?: boolean;
}

const PRESETS = [
  {
    id: 'arabian_sea',
    name: 'Arabian Sea (15.0°N, 65.0°E)',
    lat: 15.0,
    lon: 65.0,
    desc: 'Warm pool & PGW intrusion (200-300m)',
  },
  {
    id: 'bay_of_bengal',
    name: 'Bay of Bengal (14.0°N, 88.0°E)',
    lat: 14.0,
    lon: 88.0,
    desc: 'River plume, barrier layer & cyclone genesis',
  },
  {
    id: 'equatorial_io',
    name: 'Equatorial Indian Ocean (4.0°N, 78.0°E)',
    lat: 4.0,
    lon: 78.0,
    desc: 'Wyrtki jet & equatorial Kelvin waves',
  },
  {
    id: 'somali_upwelling',
    name: 'Somali Upwelling (10.0°N, 52.0°E)',
    lat: 10.0,
    lon: 52.0,
    desc: 'Great Whirl & SW monsoon coastal upwelling',
  },
  {
    id: 'seychelles_ridge',
    name: 'Seychelles-Chagos Ridge (8.0°S, 60.0°E)',
    lat: 8.0,
    lon: 60.0,
    desc: 'Open-ocean upwelling with shallow thermocline',
  },
];

const HISTORICAL_EVENTS = [
  {
    id: 'custom',
    label: '-- Select Oceanographic Event --',
    date: '',
    lat: 15.0,
    lon: 65.0,
  },
  {
    id: 'biparjoy_2023',
    label: 'June 2023: Cyclone Biparjoy (Arabian Sea)',
    date: '2023-06-11',
    lat: 16.5,
    lon: 67.4,
  },
  {
    id: 'mocha_2023',
    label: 'May 2023: Super Cyclone Mocha (Bay of Bengal)',
    date: '2023-05-12',
    lat: 16.0,
    lon: 89.5,
  },
  {
    id: 'mhw_2024',
    label: 'May 2024: Severe Arabian Sea Marine Heatwave',
    date: '2024-05-18',
    lat: 16.0,
    lon: 66.0,
  },
  {
    id: 'upwelling_2024',
    label: 'Aug 2024: Peak SW Monsoon Coastal Upwelling',
    date: '2024-08-15',
    lat: 10.0,
    lon: 52.0,
  },
  {
    id: 'warmpool_2025',
    label: 'May 2025: Pre-Monsoon Peak Warm Pool',
    date: '2025-05-15',
    lat: 15.0,
    lon: 65.0,
  },
  {
    id: 'monsoon_2025',
    label: 'Aug 2025: SW Monsoon Wind-Mixing Deepening',
    date: '2025-08-15',
    lat: 15.0,
    lon: 65.0,
  },
  {
    id: 'forecast_2026_monsoon',
    label: 'June 2026: 2026 SW Monsoon Forecast Baseline',
    date: '2026-06-15',
    lat: 14.0,
    lon: 70.0,
  },
  {
    id: 'forecast_2026_post',
    label: 'Oct 2026: 2026 Post-Monsoon Cyclone Outlook',
    date: '2026-10-20',
    lat: 12.0,
    lon: 86.0,
  },
];

export function DateRegionSelector({
  selectedLat,
  selectedLon,
  selectedDate,
  onLocationChange,
  onDateChange,
  onRefresh,
  isLoading = false,
}: DateRegionSelectorProps) {
  const [customMode, setCustomMode] = useState(false);

  const handlePresetSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === 'custom') {
      setCustomMode(true);
      return;
    }
    const preset = PRESETS.find((p) => p.id === val);
    if (preset) {
      setCustomMode(false);
      onLocationChange(preset.lat, preset.lon);
    }
  };

  const handleEventSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    const ev = HISTORICAL_EVENTS.find((item) => item.id === val);
    if (ev && ev.date) {
      onDateChange(ev.date);
      onLocationChange(ev.lat, ev.lon);
    }
  };

  const currentPresetId =
    PRESETS.find(
      (p) =>
        Math.abs(p.lat - selectedLat) < 0.1 && Math.abs(p.lon - selectedLon) < 0.1
    )?.id || (customMode ? 'custom' : 'custom');

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-200">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Date Selector (2023 - 2026) */}
        <div className="flex items-center gap-2">
          <Calendar className="h-4 w-4 text-sky-400" />
          <label className="text-xs font-medium text-slate-400">Date (2023–2026):</label>
          <input
            type="date"
            value={selectedDate}
            min="2023-01-01"
            max="2026-12-31"
            onChange={(e) => onDateChange(e.target.value)}
            className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs font-mono text-slate-100 focus:border-sky-500 focus:outline-none"
          />
        </div>

        {/* Region Preset */}
        <div className="flex items-center gap-2">
          <Compass className="h-4 w-4 text-emerald-400" />
          <label className="text-xs font-medium text-slate-400">Region:</label>
          <select
            value={currentPresetId}
            onChange={handlePresetSelect}
            className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs text-slate-100 focus:border-sky-500 focus:outline-none"
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
            <option value="custom">Custom Coordinates</option>
          </select>
        </div>

        {/* Quick Oceanographic Events Launcher */}
        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-amber-400" />
          <label className="text-xs font-medium text-slate-400">Event:</label>
          <select
            defaultValue="custom"
            onChange={handleEventSelect}
            className="rounded border border-slate-700 bg-slate-800 px-2.5 py-1 text-xs text-slate-100 focus:border-amber-500 focus:outline-none"
          >
            {HISTORICAL_EVENTS.map((ev) => (
              <option key={ev.id} value={ev.id}>
                {ev.label}
              </option>
            ))}
          </select>
        </div>

        {/* Coordinate Indicator / Controls */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 rounded bg-slate-800/80 px-2.5 py-1 font-mono text-xs text-slate-300 border border-slate-700/60">
            <MapPin className="h-3.5 w-3.5 text-sky-400" />
            <span>
              {selectedLat.toFixed(1)}°N, {selectedLon.toFixed(1)}°E
            </span>
          </div>

          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isLoading}
              className="flex items-center gap-1.5 rounded bg-sky-600 hover:bg-sky-500 disabled:bg-slate-700 px-3 py-1 text-xs font-medium text-white transition-colors"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`}
              />
              <span>{isLoading ? 'Computing...' : 'Recalculate'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Custom Coordinates Sliders (Expanded when in custom mode) */}
      {customMode && (
        <div className="mt-3.5 grid grid-cols-1 gap-4 border-t border-slate-800 pt-3 md:grid-cols-2">
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Latitude (2.0°N to 30.0°N):</span>
              <span className="font-mono text-sky-400">{selectedLat.toFixed(1)}°N</span>
            </div>
            <input
              type="range"
              min="2.0"
              max="30.0"
              step="0.5"
              value={selectedLat}
              onChange={(e) => onLocationChange(parseFloat(e.target.value), selectedLon)}
              className="w-full accent-sky-500"
            />
          </div>
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Longitude (45.0°E to 105.0°E):</span>
              <span className="font-mono text-sky-400">{selectedLon.toFixed(1)}°E</span>
            </div>
            <input
              type="range"
              min="45.0"
              max="105.0"
              step="0.5"
              value={selectedLon}
              onChange={(e) => onLocationChange(selectedLat, parseFloat(e.target.value))}
              className="w-full accent-sky-500"
            />
          </div>
        </div>
      )}
    </div>
  );
}

