'use client';

import React, { useEffect, useState } from 'react';
import { DateRegionSelector } from '../../components/DateRegionSelector';
import { TCHPGauge } from '../../components/TCHPGauge';
import { CalibrationBadge } from '../../components/CalibrationBadge';
import { fetchTCHP } from '../../lib/api';
import { TCHPResponse } from '../../lib/types';
import { Flame, AlertOctagon, BookOpen, ShieldAlert } from 'lucide-react';

export default function TCHPGaugePage() {
  const [lat, setLat] = useState(15.0);
  const [lon, setLon] = useState(65.0);
  const [date, setDate] = useState('2025-11-28');
  const [isLoading, setIsLoading] = useState(true);
  const [tchpData, setTchpData] = useState<TCHPResponse | null>(null);

  const loadTCHP = async (targetLat: number, targetLon: number, targetDate: string) => {
    setIsLoading(true);
    try {
      const data = await fetchTCHP(targetLat, targetLon, targetDate);
      setTchpData(data);
    } catch (err) {
      console.error('Error fetching TCHP:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadTCHP(lat, lon, date);
  }, [lat, lon, date]);

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-2">
          <Flame className="h-5 w-5 text-rose-500" />
          <h2 className="text-xl font-bold tracking-tight text-slate-100">
            Tropical Cyclone Heat Potential (TCHP) & Disaster Warning
          </h2>
        </div>
        <p className="mt-1 text-xs text-slate-400">
          Evaluates upper-ocean thermal energy available to sustain tropical cyclogenesis and rapid cyclone intensification across the Bay of Bengal & Arabian Sea.
        </p>
      </div>

      <DateRegionSelector
        selectedLat={lat}
        selectedLon={lon}
        selectedDate={date}
        onLocationChange={(newLat, newLon) => {
          setLat(newLat);
          setLon(newLon);
        }}
        onDateChange={setDate}
        onRefresh={() => loadTCHP(lat, lon, date)}
        isLoading={isLoading}
      />

      {tchpData && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Main TCHP Disaster Card (2 Cols) */}
          <div className="lg:col-span-2">
            <TCHPGauge data={tchpData} />
          </div>

          {/* Operational Disaster Guidelines (1 Col) */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-5 text-slate-200">
            <div className="mb-3 flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
                <BookOpen className="h-4 w-4 text-sky-400" />
                Operational Risk Matrix
              </h3>
            </div>

            <div className="space-y-3 text-xs">
              <div className="rounded border border-emerald-900/80 bg-emerald-950/40 p-2.5">
                <div className="font-bold text-emerald-400">
                  &lt; 50 kJ/cm²: Low Cyclogenesis Heat
                </div>
                <p className="mt-1 text-[11px] text-slate-300">
                  Upper-ocean heat content is insufficient to fuel rapid cyclone deepening.
                </p>
              </div>

              <div className="rounded border border-amber-900/80 bg-amber-950/40 p-2.5">
                <div className="font-bold text-amber-400">
                  50 – 80 kJ/cm²: Moderate Maintenance
                </div>
                <p className="mt-1 text-[11px] text-slate-300">
                  Sustains tropical depressions and moderate cyclones (Category 1–2).
                </p>
              </div>

              <div className="rounded border border-rose-900/80 bg-rose-950/40 p-2.5">
                <div className="font-bold text-rose-400">
                  &gt; 80 kJ/cm²: Rapid Intensification Alert
                </div>
                <p className="mt-1 text-[11px] text-slate-300">
                  Deep warm layer (&gt;26°C down to 60m+) supports rapid cyclone intensification into Very Severe Cyclonic Storms (VSCS).
                </p>
              </div>
            </div>

            {/* Mathematical Equation Box */}
            <div className="mt-4 rounded bg-slate-950/80 p-3 text-[11px] font-mono text-slate-400 border border-slate-800">
              <span className="text-slate-200 block font-semibold mb-1">
                Physical Formulation:
              </span>
              <p className="text-sky-300">
                TCHP = ρ · c_p · ∫ [T(z) - 26°C] dz (from z=0 to D26)
              </p>
              <p className="mt-1 text-[10px] text-slate-500">
                where ρ = 1025 kg/m³, c_p = 3993 J/(kg·K)
              </p>
            </div>
          </div>
        </div>
      )}

      <CalibrationBadge
        ece={0.0161}
        note={tchpData?.calibration_note}
        config={tchpData?.model_configuration}
      />
    </div>
  );
}
