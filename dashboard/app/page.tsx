'use client';

import React, { useEffect, useState } from 'react';
import { DateRegionSelector } from '../components/DateRegionSelector';
import { ProfileChart } from '../components/ProfileChart';
import { TCHPGauge } from '../components/TCHPGauge';
import { DynamicOceanMap } from '../components/DynamicOceanMap';
import { StratificationCard } from '../components/StratificationCard';
import { CalibrationBadge } from '../components/CalibrationBadge';
import {
  fetchProfile,
  fetchTCHP,
  fetchHeatwaveStatus,
  fetchHeatwaveGrid,
  fetchStratification,
} from '../lib/api';
import {
  ProfileResponse,
  TCHPResponse,
  HeatwaveStatusResponse,
  HeatwaveGridResponse,
  StratificationResponse,
} from '../lib/types';
import { Flame, Waves, Thermometer, ShieldCheck, Compass, Anchor } from 'lucide-react';

export default function OverviewPage() {
  const [lat, setLat] = useState(15.0);
  const [lon, setLon] = useState(65.0);
  const [date, setDate] = useState('2025-05-15');
  const [isLoading, setIsLoading] = useState(true);

  const [profileData, setProfileData] = useState<ProfileResponse | null>(null);
  const [tchpData, setTchpData] = useState<TCHPResponse | null>(null);
  const [mhwStatus, setMhwStatus] = useState<HeatwaveStatusResponse | null>(null);
  const [mhwGrid, setMhwGrid] = useState<HeatwaveGridResponse | null>(null);
  const [stratData, setStratData] = useState<StratificationResponse | null>(null);

  const loadData = async (targetLat: number, targetLon: number, targetDate: string) => {
    setIsLoading(true);
    try {
      const [prof, tchp, mhw, grid, strat] = await Promise.all([
        fetchProfile(targetLat, targetLon, targetDate),
        fetchTCHP(targetLat, targetLon, targetDate),
        fetchHeatwaveStatus(targetLat, targetLon, targetDate),
        fetchHeatwaveGrid(targetDate),
        fetchStratification(targetLat, targetLon, targetDate),
      ]);
      setProfileData(prof);
      setTchpData(tchp);
      setMhwStatus(mhw);
      setMhwGrid(grid);
      setStratData(strat);
    } catch (err) {
      console.error('Error loading dashboard data:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData(lat, lon, date);
  }, [lat, lon, date]);

  const handleLocationChange = (newLat: number, newLon: number) => {
    setLat(newLat);
    setLon(newLon);
  };

  const handleDateChange = (newDate: string) => {
    setDate(newDate);
  };

  return (
    <div className="space-y-6">
      {/* Title & Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center gap-2.5">
            <h2 className="text-xl font-bold tracking-tight text-slate-100">
              OceanEmbed: Disaster Operations Center
            </h2>
            <span className="rounded bg-sky-950 px-2 py-0.5 font-mono text-xs text-sky-400 border border-sky-800">
              Phase 8 Operational
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Continuous 3D thermal soundings (0–1000m), TCHP cyclone risk tracking, and marine heatwave detection for the North Indian Ocean.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="rounded bg-slate-900 px-2.5 py-1 text-slate-300 border border-slate-800">
            Lat: {lat.toFixed(1)}°N
          </span>
          <span className="rounded bg-slate-900 px-2.5 py-1 text-slate-300 border border-slate-800">
            Lon: {lon.toFixed(1)}°E
          </span>
          <span className="rounded bg-sky-950 px-2.5 py-1 text-sky-300 border border-sky-800/60">
            {date}
          </span>
        </div>
      </div>

      {/* Global Interactive Selector */}
      <DateRegionSelector
        selectedLat={lat}
        selectedLon={lon}
        selectedDate={date}
        onLocationChange={handleLocationChange}
        onDateChange={handleDateChange}
        onRefresh={() => loadData(lat, lon, date)}
        isLoading={isLoading}
      />

      {/* Quick Summary Key Metrics Bar */}
      {tchpData && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-slate-800 bg-slate-900/80 p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>TCHP Energy</span>
              <Flame className="h-4 w-4 text-rose-400" />
            </div>
            <div className="mt-1 font-mono text-lg font-bold text-rose-300">
              {tchpData.tchp.mean.toFixed(1)} <span className="text-xs font-normal text-slate-400">{tchpData.tchp.unit}</span>
            </div>
            <span className="text-[10px] text-slate-500">±{tchpData.tchp.std.toFixed(1)} (Calibrated 95% CI)</span>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/80 p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Mixed Layer (MLD)</span>
              <Waves className="h-4 w-4 text-cyan-400" />
            </div>
            <div className="mt-1 font-mono text-lg font-bold text-cyan-300">
              {tchpData.mld_direct.mean.toFixed(1)} <span className="text-xs font-normal text-slate-400">m</span>
            </div>
            <span className="text-[10px] text-slate-500">Direct Physics Estimate</span>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/80 p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>26°C Isotherm (D26)</span>
              <Compass className="h-4 w-4 text-amber-400" />
            </div>
            <div className="mt-1 font-mono text-lg font-bold text-amber-300">
              {tchpData.d26.mean.toFixed(1)} <span className="text-xs font-normal text-slate-400">m</span>
            </div>
            <span className="text-[10px] text-slate-500">Upper Thermocline Depth</span>
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/80 p-3.5">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Marine Heatwave</span>
              <Thermometer className="h-4 w-4 text-purple-400" />
            </div>
            <div className="mt-1 font-mono text-lg font-bold text-purple-300">
              {mhwStatus?.category_label || 'Normal'}
            </div>
            <span className="text-[10px] text-slate-500">
              {mhwStatus?.is_active_heatwave ? `Anomaly: +${mhwStatus.anomaly_c}°C` : 'Within 90th percentile'}
            </span>
          </div>
        </div>
      )}

      {/* Dynamic Interactive GIS Map */}
      <DynamicOceanMap
        selectedLat={lat}
        selectedLon={lon}
        selectedDate={date}
        gridData={mhwGrid || undefined}
        onSelectCoordinate={handleLocationChange}
        height="500px"
      />

      {/* 3D Vertical Profile & Disaster Gauge Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left: Depth Sounding Chart */}
        <ProfileChart profileData={profileData} isLoading={isLoading} />

        {/* Right: TCHP & Cyclogenesis Risk Card */}
        <TCHPGauge data={tchpData} isLoading={isLoading} />
      </div>

      {/* Stratification & Ocean Physics Engine Card */}
      <StratificationCard data={stratData} isLoading={isLoading} />

      {/* Transparent Calibration Disclosure */}
      <CalibrationBadge />
    </div>
  );
}
