'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Layers, MapPin, Wind, Compass, Radio, Activity } from 'lucide-react';
import { CycloneTracksResponse, ValidationStationsResponse, HeatwaveGridResponse } from '../lib/types';
import { fetchCycloneTracks, fetchValidationStations } from '../lib/api';

interface DynamicOceanMapProps {
  selectedLat: number;
  selectedLon: number;
  selectedDate: string;
  gridData?: HeatwaveGridResponse;
  onSelectCoordinate?: (lat: number, lon: number) => void;
  title?: string;
  height?: string;
}

export function DynamicOceanMap({
  selectedLat,
  selectedLon,
  selectedDate,
  gridData,
  onSelectCoordinate,
  title = 'North Indian Ocean Geospatial Explorer & Disaster Mapping',
  height = '520px',
}: DynamicOceanMapProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const layersRef = useRef<{ [key: string]: any }>({});
  const markerRef = useRef<any>(null);

  // Layer Toggles
  const [showMHW, setShowMHW] = useState(true);
  const [showTCHP, setShowTCHP] = useState(true);
  const [showGoship, setShowGoship] = useState(true);
  const [showRama, setShowRama] = useState(true);
  const [showCyclones, setShowCyclones] = useState(true);

  // Hover coordinate HUD
  const [cursorCoords, setCursorCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [cycloneData, setCycloneData] = useState<CycloneTracksResponse | null>(null);
  const [validationData, setValidationData] = useState<ValidationStationsResponse | null>(null);

  // Load ancillary datasets on mount
  useEffect(() => {
    let mounted = true;
    Promise.all([fetchCycloneTracks(), fetchValidationStations()])
      .then(([cData, vData]) => {
        if (mounted) {
          setCycloneData(cData);
          setValidationData(vData);
        }
      })
      .catch((err) => console.error('Failed to load ancillary map data:', err));
    return () => {
      mounted = false;
    };
  }, []);

  // Initialize Leaflet Map
  useEffect(() => {
    if (typeof window === 'undefined' || !mapContainerRef.current) return;

    let isSubscribed = true;

    // Dynamically import Leaflet
    import('leaflet').then((L) => {
      if (!isSubscribed || !mapContainerRef.current) return;

      // Clean up previous instance if exists
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }

      // Initialize map centered on North Indian Ocean
      const map = L.map(mapContainerRef.current, {
        center: [14.0, 72.0],
        zoom: 5,
        minZoom: 3,
        maxZoom: 9,
        maxBounds: [
          [-15.0, 35.0],
          [35.0, 115.0],
        ],
        zoomControl: true,
      });

      mapInstanceRef.current = map;

      // CartoDB Dark Matter Basemap
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19,
      }).addTo(map);

      // Mousemove event for HUD
      map.on('mousemove', (e: any) => {
        setCursorCoords({
          lat: parseFloat(e.latlng.lat.toFixed(2)),
          lon: parseFloat(e.latlng.lng.toFixed(2)),
        });
      });

      // Click event for selecting coordinates
      map.on('click', (e: any) => {
        const lat = parseFloat(e.latlng.lat.toFixed(1));
        const lon = parseFloat(e.latlng.lng.toFixed(1));
        if (lat >= 2.0 && lat <= 30.0 && lon >= 45.0 && lon <= 105.0) {
          if (onSelectCoordinate) onSelectCoordinate(lat, lon);
        }
      });

      // Initialize layer groups
      layersRef.current.mhw = L.layerGroup().addTo(map);
      layersRef.current.tchp = L.layerGroup().addTo(map);
      layersRef.current.goship = L.layerGroup().addTo(map);
      layersRef.current.rama = L.layerGroup().addTo(map);
      layersRef.current.cyclones = L.layerGroup().addTo(map);

      // Selected Location Marker (Glowing circle beacon)
      markerRef.current = L.circleMarker([selectedLat, selectedLon], {
        radius: 8,
        color: '#38bdf8',
        weight: 3,
        fillColor: '#0284c7',
        fillOpacity: 0.9,
      })
        .addTo(map)
        .bindPopup(`<b>Active Sounding Coordinate</b><br/>Lat: ${selectedLat}°N, Lon: ${selectedLon}°E`);

      // Trigger map resize after container layout
      setTimeout(() => map.invalidateSize(), 200);
    });

    return () => {
      isSubscribed = false;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update Selected Coordinate Marker
  useEffect(() => {
    if (!mapInstanceRef.current || !markerRef.current) return;
    markerRef.current.setLatLng([selectedLat, selectedLon]);
    markerRef.current.setPopupContent(
      `<div class="text-xs font-sans"><b>Active Sounding Point</b><br/>${selectedLat.toFixed(1)}°N, ${selectedLon.toFixed(1)}°E<br/><span class="text-slate-500">Date: ${selectedDate}</span></div>`
    );
  }, [selectedLat, selectedLon, selectedDate]);

  // Update Layers when data or toggles change
  useEffect(() => {
    if (!mapInstanceRef.current) return;

    import('leaflet').then((L) => {
      const { mhw, tchp, goship, rama, cyclones } = layersRef.current;
      if (!mhw) return;

      // 1. MHW Layer
      mhw.clearLayers();
      if (showMHW && gridData) {
        const { rows, cols } = gridData.dimensions;
        const lats = gridData.latitudes;
        const lons = gridData.longitudes;

        // Render clusters of MHW
        for (let r = 0; r < rows; r += 2) {
          for (let c = 0; c < cols; c += 2) {
            const cat = gridData.categories[r]?.[c] || 0;
            const isOcean = gridData.ocean_mask[r]?.[c];
            if (isOcean && cat > 0) {
              const lat = lats[r];
              const lon = lons[c];
              const color = cat === 1 ? '#f59e0b' : cat === 2 ? '#ea580c' : cat === 3 ? '#dc2626' : '#7f1d1d';
              const catLabel = cat === 1 ? 'Cat I (Moderate)' : cat === 2 ? 'Cat II (Strong)' : cat === 3 ? 'Cat III (Severe)' : 'Cat IV (Extreme)';

              L.rectangle(
                [
                  [lat - 0.5, lon - 0.5],
                  [lat + 0.5, lon + 0.5],
                ],
                {
                  color: 'none',
                  fillColor: color,
                  fillOpacity: 0.55,
                }
              )
                .addTo(mhw)
                .bindTooltip(`<b>MHW Event: ${catLabel}</b><br/>${lat.toFixed(1)}°N, ${lon.toFixed(1)}°E<br/>Anomaly: +1.4°C`);
            }
          }
        }
      }

      // 2. TCHP Cyclone Thermal Potential Zones
      tchp.clearLayers();
      if (showTCHP) {
        // Arabian Sea Warm Pool TCHP Zone (>80 kJ/cm^2)
        L.polygon(
          [
            [12.0, 62.0],
            [18.0, 64.0],
            [17.5, 72.0],
            [11.0, 71.0],
          ],
          {
            color: '#ef4444',
            weight: 1.5,
            dashArray: '4, 4',
            fillColor: '#dc2626',
            fillOpacity: 0.15,
          }
        )
          .addTo(tchp)
          .bindTooltip(`<b>Arabian Sea Cyclone Genesis Reservoir</b><br/>TCHP &gt; 80 kJ/cm² (Rapid Intensification Alert)`);

        // Bay of Bengal High TCHP Zone (>100 kJ/cm^2)
        L.polygon(
          [
            [10.0, 84.0],
            [18.0, 86.0],
            [19.0, 92.0],
            [12.0, 91.0],
          ],
          {
            color: '#f97316',
            weight: 1.5,
            dashArray: '4, 4',
            fillColor: '#ea580c',
            fillOpacity: 0.18,
          }
        )
          .addTo(tchp)
          .bindTooltip(`<b>Bay of Bengal Cyclone Fuel Zone</b><br/>TCHP &gt; 100 kJ/cm² • Thick Barrier Layer`);
      }

      // 3. GO-SHIP / CCHDO Hydrographic Transects
      goship.clearLayers();
      if (showGoship && validationData?.goship_transects) {
        validationData.goship_transects.forEach((transect) => {
          const points: [number, number][] = transect.stations.map((s) => [s.lat, s.lon]);
          L.polyline(points, {
            color: '#38bdf8',
            weight: 2,
            opacity: 0.85,
          }).addTo(goship);

          transect.stations.forEach((s) => {
            L.circleMarker([s.lat, s.lon], {
              radius: 4,
              color: '#38bdf8',
              weight: 1.5,
              fillColor: '#0369a1',
              fillOpacity: 0.9,
            })
              .addTo(goship)
              .bindPopup(
                `<b>GO-SHIP Line ${transect.line} - ${s.station_id}</b><br/>` +
                  `Coordinates: ${s.lat}°N, ${s.lon}°E<br/>` +
                  `Depth Range: ${s.depth_range}<br/>` +
                  `Validation RMSE: <span class="font-mono text-emerald-400">${s.rmse_c}°C</span><br/>` +
                  `Overall Line R²: <span class="font-mono text-sky-400">${transect.r2}</span>`
              );
          });
        });
      }

      // 4. RAMA Moored Ocean Buoys
      rama.clearLayers();
      if (showRama && validationData?.rama_buoys) {
        validationData.rama_buoys.forEach((b) => {
          L.circleMarker([b.lat, b.lon], {
            radius: 5,
            color: '#10b981',
            weight: 2,
            fillColor: '#065f46',
            fillOpacity: 0.9,
          })
            .addTo(rama)
            .bindPopup(
              `<b>RAMA Moored Buoy: ${b.id}</b><br/>` +
                `Name: ${b.name}<br/>` +
                `Coordinates: ${b.lat}°N, ${b.lon}°E<br/>` +
                `Telemetry: ${b.sensors}<br/>` +
                `Subsurface RMSE: <span class="font-mono text-emerald-400">${b.rmse_c}°C</span>`
            );
        });
      }

      // 5. Historical Cyclone Tracks
      cyclones.clearLayers();
      if (showCyclones && cycloneData?.cyclones) {
        const cycloneColors: { [key: string]: string } = {
          biparjoy_2023: '#a855f7',
          mocha_2023: '#ec4899',
          tauktae_2021: '#f59e0b',
        };

        cycloneData.cyclones.forEach((cyc) => {
          const color = cycloneColors[cyc.id] || '#cbd5e1';
          const points: [number, number][] = cyc.track.map((wp) => [wp.lat, wp.lon]);

          // Draw Track Polyline
          L.polyline(points, {
            color,
            weight: 2.5,
            opacity: 0.9,
          }).addTo(cyclones);

          // Draw Waypoints
          cyc.track.forEach((wp) => {
            L.circleMarker([wp.lat, wp.lon], {
              radius: 4.5,
              color,
              weight: 1.5,
              fillColor: '#0f172a',
              fillOpacity: 0.95,
            })
              .addTo(cyclones)
              .bindPopup(
                `<b>${cyc.name} (${cyc.year})</b><br/>` +
                  `Stage: ${wp.status}<br/>` +
                  `Date: ${wp.date}<br/>` +
                  `Max Wind: ${wp.wind_kts} kts<br/>` +
                  `Ocean TCHP: <span class="font-mono text-amber-400">${wp.tchp} kJ/cm²</span>`
              );
          });
        });
      }
    });
  }, [showMHW, showTCHP, showGoship, showRama, showCyclones, gridData, validationData, cycloneData]);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/90 p-4 text-slate-200">
      {/* Header & Controls */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Compass className="h-4 w-4 text-sky-400" />
            {title}
          </h3>
          <p className="text-xs text-slate-400">
            Interactive North Indian Ocean Domain (2–30°N, 45–105°E) • Date: <span className="font-mono text-sky-400">{selectedDate}</span>
          </p>
        </div>

        {/* Dynamic Layer Toggles */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button
            onClick={() => setShowMHW(!showMHW)}
            className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors border ${
              showMHW
                ? 'bg-rose-950/80 text-rose-300 border-rose-800'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Activity className="h-3 w-3" />
            <span>Heatwaves (MHW)</span>
          </button>

          <button
            onClick={() => setShowTCHP(!showTCHP)}
            className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors border ${
              showTCHP
                ? 'bg-amber-950/80 text-amber-300 border-amber-800'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Wind className="h-3 w-3" />
            <span>TCHP Risk Zones</span>
          </button>

          <button
            onClick={() => setShowGoship(!showGoship)}
            className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors border ${
              showGoship
                ? 'bg-sky-950/80 text-sky-300 border-sky-800'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3 w-3" />
            <span>GO-SHIP Transects</span>
          </button>

          <button
            onClick={() => setShowRama(!showRama)}
            className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors border ${
              showRama
                ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Radio className="h-3 w-3" />
            <span>RAMA Buoys</span>
          </button>

          <button
            onClick={() => setShowCyclones(!showCyclones)}
            className={`flex items-center gap-1.5 rounded px-2.5 py-1 font-medium transition-colors border ${
              showCyclones
                ? 'bg-purple-950/80 text-purple-300 border-purple-800'
                : 'bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200'
            }`}
          >
            <Wind className="h-3 w-3 text-purple-400" />
            <span>Cyclone Tracks</span>
          </button>
        </div>
      </div>

      {/* Real Dynamic Map Container */}
      <div className="relative overflow-hidden rounded border border-slate-800 bg-slate-950">
        <div ref={mapContainerRef} style={{ width: '100%', height }} className="z-0" />

        {/* Live Coordinate HUD Overlay */}
        <div className="absolute top-3 left-3 z-[500] pointer-events-none flex flex-col gap-1 rounded bg-slate-950/90 px-3 py-1.5 border border-slate-700 font-mono text-[11px] text-slate-300 backdrop-blur-sm">
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Cursor:</span>
            <span>{cursorCoords ? `${cursorCoords.lat}°N, ${cursorCoords.lon}°E` : 'Hover map'}</span>
          </div>
          <div className="flex items-center gap-2 text-sky-400">
            <MapPin className="h-3 w-3" />
            <span>Selected: {selectedLat.toFixed(1)}°N, {selectedLon.toFixed(1)}°E</span>
          </div>
        </div>

        {/* Action Hint */}
        <div className="absolute bottom-3 left-3 z-[500] pointer-events-none rounded bg-slate-950/90 px-2.5 py-1 border border-slate-800 text-[11px] text-slate-400 backdrop-blur-sm">
          Click any ocean coordinate to compute 3D Subsurface Profile, TCHP &amp; MLD
        </div>
      </div>

      {/* Map Legend */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-slate-800 pt-2.5 text-xs text-slate-400">
        <div className="flex flex-wrap items-center gap-4">
          <span className="text-slate-500 font-medium">Map Layers:</span>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-sky-500" />
            <span className="text-[11px]">Active Sounding Pin</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-4 bg-[#f59e0b]/70 border border-amber-500/50" />
            <span className="text-[11px]">MHW Heatwave Footprint</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-4 bg-[#dc2626]/40 border border-red-500 border-dashed" />
            <span className="text-[11px]">TCHP Cyclone Fuel (&gt;80 kJ/cm²)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-4 bg-sky-400" />
            <span className="text-[11px]">GO-SHIP Lines (I01, I08N, I09N)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
            <span className="text-[11px]">RAMA Buoys</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-4 bg-purple-500" />
            <span className="text-[11px]">Cyclones (Biparjoy/Mocha/Tauktae)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
