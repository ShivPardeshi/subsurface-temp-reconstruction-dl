'use client';

import React, { useEffect, useState } from 'react';
import { DateRegionSelector } from '../../components/DateRegionSelector';
import { ProfileChart } from '../../components/ProfileChart';
import { CalibrationBadge } from '../../components/CalibrationBadge';
import { fetchProfile } from '../../lib/api';
import { ProfileResponse } from '../../lib/types';
import { Layers, ArrowDownUp, CheckCircle2 } from 'lucide-react';

export default function ProfileViewerPage() {
  const [lat, setLat] = useState(15.0);
  const [lon, setLon] = useState(65.0);
  const [date, setDate] = useState('2025-11-28');
  const [isLoading, setIsLoading] = useState(true);
  const [profileData, setProfileData] = useState<ProfileResponse | null>(null);

  const loadProfile = async (targetLat: number, targetLon: number, targetDate: string) => {
    setIsLoading(true);
    try {
      const data = await fetchProfile(targetLat, targetLon, targetDate);
      setProfileData(data);
    } catch (err) {
      console.error('Error fetching profile:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadProfile(lat, lon, date);
  }, [lat, lon, date]);

  return (
    <div className="space-y-6">
      <div>
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-cyan-400" />
          <h2 className="text-xl font-bold tracking-tight text-slate-100">
            3D Vertical Sounding Explorer (0m to 1000m)
          </h2>
        </div>
        <p className="mt-1 text-xs text-slate-400">
          Reconstructs high-resolution continuous temperature profiles across all 15 canonical physical depths with calibrated posterior uncertainty bounds.
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
        onRefresh={() => loadProfile(lat, lon, date)}
        isLoading={isLoading}
      />

      {profileData && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Main Visual Sounding Chart (2 Columns) */}
          <div className="lg:col-span-2">
            <ProfileChart
              profile={profileData.profile}
              climatology={profileData.climatology_profile}
              mldMeters={profileData.mld_direct.mean_meters}
              d26Meters={profileData.d26.mean_meters}
              locationName={`${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E`}
            />
          </div>

          {/* Depth Soundings Tabular Breakdown (1 Column) */}
          <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-200">
            <div className="mb-3 flex items-center justify-between border-b border-slate-800 pb-2">
              <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-1.5">
                <ArrowDownUp className="h-4 w-4 text-sky-400" />
                Depth Soundings Table
              </h3>
              <span className="text-[11px] font-mono text-slate-400">15 Depths</span>
            </div>

            <div className="max-h-[380px] overflow-y-auto pr-1">
              <table className="w-full text-left font-mono text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-[11px] text-slate-400">
                    <th className="pb-1.5 font-medium">Depth</th>
                    <th className="pb-1.5 font-medium">Pred (°C)</th>
                    <th className="pb-1.5 font-medium">Clim (°C)</th>
                    <th className="pb-1.5 font-medium">±1σ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {profileData.profile.depths_m.map((depth, idx) => {
                    const pred = profileData.profile.mean_temperatures[idx];
                    const clim = profileData.climatology_profile[idx];
                    const std = profileData.profile.std_temperatures[idx];
                    const isThermocline = depth >= 75 && depth <= 150;

                    return (
                      <tr
                        key={depth}
                        className={isThermocline ? 'bg-sky-950/30 text-sky-200' : ''}
                      >
                        <td className="py-1.5 text-slate-400">{depth}m</td>
                        <td className="py-1.5 font-bold text-slate-100">
                          {pred.toFixed(2)}
                        </td>
                        <td className="py-1.5 text-slate-400">{clim.toFixed(2)}</td>
                        <td className="py-1.5 text-slate-500">±{std.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="mt-3.5 rounded bg-slate-950/60 p-2.5 text-[11px] text-slate-400 border border-slate-800">
              <div className="flex items-center gap-1 text-sky-400 font-medium">
                <CheckCircle2 className="h-3 w-3" />
                <span>Thermocline Core Superiority:</span>
              </div>
              <p className="mt-0.5 leading-relaxed text-slate-400">
                Blue highlighted rows (75m–150m) represent the steep pycnocline layer where OceanEmbed outperforms linear baselines (+3.89% skill).
              </p>
            </div>
          </div>
        </div>
      )}

      <CalibrationBadge
        ece={0.0161}
        note={profileData?.calibration_note}
        config={profileData?.model_configuration}
      />
    </div>
  );
}
