'use client';

import React, { useEffect, useState } from 'react';
import { DateRegionSelector } from '../../components/DateRegionSelector';
import { DynamicOceanMap } from '../../components/DynamicOceanMap';
import { CalibrationBadge } from '../../components/CalibrationBadge';
import { fetchHeatwaveGrid, fetchHeatwaveStatus } from '../../lib/api';
import { HeatwaveGridResponse, HeatwaveStatusResponse } from '../../lib/types';
import { Thermometer, AlertTriangle, CheckCircle, Flame, Activity, Waves } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';

export default function HeatwaveMapPage() {
  const [lat, setLat] = useState(15.0);
  const [lon, setLon] = useState(65.0);
  const [date, setDate] = useState('2025-05-15');
  const [isLoading, setIsLoading] = useState(true);

  const [mhwGrid, setMhwGrid] = useState<HeatwaveGridResponse | null>(null);
  const [mhwStatus, setMhwStatus] = useState<HeatwaveStatusResponse | null>(null);

  const loadData = async (targetLat: number, targetLon: number, targetDate: string) => {
    setIsLoading(true);
    try {
      const [grid, status] = await Promise.all([
        fetchHeatwaveGrid(targetDate),
        fetchHeatwaveStatus(targetLat, targetLon, targetDate),
      ]);
      setMhwGrid(grid);
      setMhwStatus(status);
    } catch (err) {
      console.error('Error fetching MHW data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData(lat, lon, date);
  }, [lat, lon, date]);

  // Transform recent 7-day series for Recharts
  const timeSeriesData =
    mhwStatus?.recent_series.map((t, idx) => ({
      day: `Day t-${6 - idx}`,
      temp: t,
      threshold: mhwStatus.threshold_90th,
    })) || [];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-amber-400" />
            <h2 className="text-xl font-bold tracking-tight text-slate-100">
              Marine Heatwave Spatial Tracking (Hobday et al., 2016)
            </h2>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Identifies extreme upper-ocean thermal anomalies exceeding the 90th percentile climatology for 5+ consecutive days across the North Indian Ocean.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="rounded bg-rose-950/80 px-2.5 py-1 text-rose-300 border border-rose-800/60">
            {mhwGrid ? `${mhwGrid.summary.mhw_area_percentage}% Basin Affected` : 'Loading...'}
          </span>
          <span className="rounded bg-slate-900 px-2.5 py-1 text-slate-400 border border-slate-800">
            {mhwGrid?.summary.active_mhw_cells || 0} Active Cells
          </span>
        </div>
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
        onRefresh={() => loadData(lat, lon, date)}
        isLoading={isLoading}
      />

      {/* Main Dynamic Map View */}
      <DynamicOceanMap
        selectedLat={lat}
        selectedLon={lon}
        selectedDate={date}
        gridData={mhwGrid || undefined}
        onSelectCoordinate={(newLat, newLon) => {
          setLat(newLat);
          setLon(newLon);
        }}
        title="Interactive Marine Heatwave Categories &amp; Ocean Disaster GIS"
        height="540px"
      />

      {/* Row 2: Selected Point Status & 7-Day Time Series */}
      {mhwStatus && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Card 1: Point Status */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
              Sounding Location Status
            </h3>

            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs text-slate-400">Event Status:</span>
                <span
                  className={`inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium border ${
                    mhwStatus.is_active_heatwave
                      ? 'bg-rose-950 text-rose-300 border-rose-800'
                      : 'bg-emerald-950 text-emerald-300 border-emerald-800'
                  }`}
                >
                  {mhwStatus.is_active_heatwave ? (
                    <>
                      <AlertTriangle className="h-3 w-3" /> Active Marine Heatwave
                    </>
                  ) : (
                    <>
                      <CheckCircle className="h-3 w-3" /> Normal Climatology
                    </>
                  )}
                </span>
              </div>

              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs text-slate-400">Hobday Category:</span>
                <span className="font-mono text-xs font-semibold text-slate-200">
                  {mhwStatus.category_label}
                </span>
              </div>

              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs text-slate-400">Current SST:</span>
                <span className="font-mono text-xs text-slate-200">
                  {mhwStatus.current_sst}°C
                </span>
              </div>

              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs text-slate-400">90th Percentile Climatology:</span>
                <span className="font-mono text-xs text-slate-200">
                  {mhwStatus.threshold_90th}°C
                </span>
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Thermal Anomaly:</span>
                <span
                  className={`font-mono text-xs font-bold ${
                    mhwStatus.anomaly_c > 0 ? 'text-rose-400' : 'text-emerald-400'
                  }`}
                >
                  {mhwStatus.anomaly_c > 0 ? `+${mhwStatus.anomaly_c}°C` : `${mhwStatus.anomaly_c}°C`}
                </span>
              </div>
            </div>
          </div>

          {/* Card 2 & 3: 7-Day Temperature vs Threshold Chart */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 lg:col-span-2">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
              7-Day SST Progression vs 90th Percentile Threshold
            </h3>

            <div className="h-48 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData} margin={{ top: 10, right: 20, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="day" tick={{ fill: '#94a3b8', fontSize: 10 }} stroke="#334155" />
                  <YAxis domain={['auto', 'auto']} tick={{ fill: '#94a3b8', fontSize: 10 }} stroke="#334155" unit="°C" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0f172a',
                      borderColor: '#334155',
                      borderRadius: '6px',
                      fontSize: '11px',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                  <ReferenceLine
                    y={mhwStatus.threshold_90th}
                    stroke="#dc2626"
                    strokeDasharray="4 4"
                    label={{
                      value: '90th % Threshold',
                      fill: '#f87171',
                      fontSize: 10,
                      position: 'top',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="temp"
                    name="Observed / Model SST"
                    stroke="#38bdf8"
                    strokeWidth={2.5}
                    dot={{ fill: '#38bdf8', r: 3.5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      <CalibrationBadge />
    </div>
  );
}
