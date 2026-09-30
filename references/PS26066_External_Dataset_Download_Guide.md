# PS26066 — External Dataset Download Guide

Covers every **external** dataset identified across our planning (beyond the 7 SIH-mandated core variables you've already downloaded: SST, SSS, SSH/SLA, winds, currents, and GLORYS/ARGO). All links below were verified via live search just before writing this, not pulled from memory — but data portal UIs do change, so if a specific link 404s, the parent domain listed alongside it is your fallback starting point.

**Legend:** 🔓 = no registration needed, direct download. 🔒 = free registration required (noted with approval-time expectations where known).

---

## TIER 1 — Essential (directly targets Bay of Bengal / Arabian Sea priority zones)

### 1. River discharge (Ganges-Brahmaputra-Meghna) — Bay of Bengal barrier layer
**Primary source — no registration, satellite-derived: 🔓 ESA CCI River Discharge (RD_cci)**
- Portal: `https://data.ceda.ac.uk/neodc/esacci/river_discharge/data/RD/RD-multi/v1.2/`
- Steps:
  1. Navigate to the URL above — it's a plain directory listing (CEDA archive), no login needed for browsing/download.
  2. Look for station/location files covering the Ganges and Brahmaputra — filenames are organized by gauge location, browse or use the CEDA search interface at `data.ceda.ac.uk` if the direct path structure has changed.
  3. Download the relevant multi-mission merged (CM) time series in NetCDF/CSV.
- Why this one first: it's satellite-derived, so it doesn't depend on Indian government portal approval — a good hedge while other registrations are pending.

**Secondary source — cross-check, free registration: 🔒 GRDC (Global Runoff Data Centre)**
- Portal: `https://grdc.bafg.de/`
- Steps:
  1. Go to the portal, use "Data Download" → search by river name ("Ganges", "Brahmaputra") or by country (India, Bangladesh).
  2. Register for a free account if the specific stations require it (some GRDC data is open, some requires the free data-request form).
  3. Download daily/monthly discharge as CSV.

**Tertiary/dam-specific source — India's own, free registration: 🔒 India-WRIS**
- Portal: `https://indiawris.gov.in/`
- Steps:
  1. Register for an account.
  2. Navigate to the real-time data / reservoir dashboard section, search for Farakka Barrage specifically, or the relevant CWC monitoring stations on the Ganges/Brahmaputra.
  3. Data can typically be viewed as graphs and exported (Excel/CSV) from the dashboard.
- **Honest caveat, worth knowing before you invest time here:** we found a documented real user report describing friction actually extracting raw time-series data through this portal for research use. Treat this as a "try it, timebox it, fall back to ESA CCI / GRDC if it's not working smoothly" source, not your only plan.

### 2. Precipitation — Bay of Bengal freshwater forcing, both basins' E−P balance
**🔒 GPM IMERG via NASA Earthdata / GES DISC**
- Portal: `https://gpm.nasa.gov/data/imerg` (info) → actual data via `https://disc.gsfc.nasa.gov/`
- Steps:
  1. You should already have a NASA Earthdata login from registering for SSS access — same account works here.
  2. On GES DISC, search "IMERG" — select the daily (or half-hourly, if you want to aggregate yourself) Final Run product.
  3. Use the GES DISC subsetting tool to constrain to our domain (2°N–30°N, 45°E–105°E) and date range before downloading, to avoid pulling global data unnecessarily.

**Alternative/supplement — India's own: 🔒 MOSDAC MT-SAPHIR rainfall**
- Portal: `https://mosdac.gov.in/`
- Steps: see the general MOSDAC registration process in Tier 2 below (applies to all MOSDAC products) — search the catalog for "MT-SAPHIR" or "rainfall" once logged in, order via cart, retrieve via SFTP.

### 3. Latent heat flux (and related surface meteorology) — E−P balance, Arabian Sea winter convection
**🔓 OAFlux (WHOI) — genuinely open, no registration**
- Portal: `https://oaflux.whoi.edu/data-access`
- Steps:
  1. Go to the data-access page above.
  2. Method 1 (simplest): use an ordinary browser or FTP client to go to `ftp://ftp.whoi.edu/pub/science/oaflux/data_v3` and browse directly.
  3. Download the daily 1° gridded NetCDF files (latent heat flux, sensible heat flux, evaporation, wind speed, SST, humidity all included) for your training period.
  4. Note: 1° native resolution is coarser than our 0.25° target grid — you'll need to regrid/interpolate this into our pipeline, consistent with the PS's own allowance for using an available product with appropriate interpolation.

### 4. Wind stress curl / Ekman pumping — Arabian Sea upwelling
**🔒 ISRO EOS-06 (Oceansat-3) Scatterometer, via MOSDAC**
- Portal: `https://mosdac.gov.in/`
- Steps:
  1. Register (see general MOSDAC steps below).
  2. In the catalog, search for the scatterometer wind product (look for "L4AW" or "Analysed Winds" under Oceansat-3/EOS-06 satellite products) — this product includes wind divergence and wind stress curl as value-added outputs, meaning you don't have to derive curl yourself from raw wind fields.
  3. Order via cart, retrieve via the SFTP link MOSDAC provides once ready.
- **Fallback if this specific derived product is hard to locate:** download raw scatterometer or reanalysis wind U/V (ERA5, or the PS-mandated wind product you already have) and compute wind stress curl yourself (∇×τ) — every input needed for that (wind U, V) you already have.

### 5. Chlorophyll-a / ocean color — Arabian Sea upwelling proxy, river plume tracing
**🔒 NASA Ocean Color Web**
- Portal: `https://oceancolor.gsfc.nasa.gov/`
- Steps:
  1. Same NASA Earthdata login as IMERG/SSS.
  2. Use the Level-3 browser to select MODIS-Aqua or VIIRS chlorophyll-a, daily or 8-day composite, and download for our domain/date range.

**Alternative — India's own: 🔒 MOSDAC Oceansat-3 OCM-3**
- Same MOSDAC registration/ordering process, search catalog for "OCM-3" ocean color chlorophyll product.

---

## TIER 2 — Structural / static (needed regardless of region, low effort, mostly no registration)

### 6. Bathymetry — land masking, shelf-depth context
**🔓 GEBCO — no registration, and there's a subset-download tool so you don't need the full global file**
- Portal: `https://download.gebco.net`
- Steps:
  1. Go to the URL above (GEBCO's dedicated regional-subset download app).
  2. Draw/enter our bounding box (2°N–30°N, 45°E–105°E) directly in the tool.
  3. Select the GEBCO_2026 grid, choose NetCDF format, submit — you'll get an email with a download link once your (small, regional) extract is ready. Much smaller than the ~7GB global file.

### 7. Land/ocean mask (if you want a dedicated shapefile rather than deriving purely from GEBCO)
**🔓 Natural Earth**
- Portal: `https://www.naturalearthdata.com/downloads/` → "Physical Vectors" → "Land" (10m or 50m resolution)
- Steps: direct download, no registration, standard shapefile format, load with any GIS/Python geopandas workflow.

### 8. Climate indices (ONI, IOD/DMI) — basin-wide interannual conditioning
**🔓 ONI (El Niño/ENSO), NOAA CPC**
- Portal: `https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v4.shtml`
- Steps: this is a plain HTML table of monthly values going back to 1950 — copy/paste or scrape directly, no download file needed.

**🔓 IOD/DMI, NOAA PSL**
- Portal: `https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/`
- Steps: direct plain-text/CSV time series download, no registration.

### 9. IBTrACS — cyclone tracks for extreme-event evaluation windows
**🔓 NOAA NCEI — no registration**
- Direct CSV access: `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/`
- Steps:
  1. Go to the URL above.
  2. Download the **North Indian** basin-specific file (much smaller than the global file, and it's exactly our region) rather than the full global archive.
  3. This gives cyclone position, intensity, and timing at 3-hour intervals — use this to identify real historical cyclone windows in your test period for the extreme-event evaluation slice.

---

## TIER 3 — Optional / cross-validation (useful, not blocking)

### 10. EN4 quality-controlled ocean profiles — an independent accuracy cross-check, same reference dataset kanishkaanarang used
**🔒 Met Office Hadley Centre**
- Portal: `https://www.metoffice.gov.uk/hadobs/en4/`
- Steps: free registration, then direct download of quality-controlled profile data by year — useful as a secondary independent validation source alongside real ARGO.

### 11. RAMA moored buoy data — for the validation-independence investigation we flagged earlier
**🔓/🔒 PMEL TAO/RAMA (registration requirements vary by exact data request)**
- Portal: `https://www.pmel.noaa.gov/tao/drupal/disdel/`
- Steps:
  1. Use the data delivery interface, select the RAMA array, choose Indian Ocean buoy locations within our domain.
  2. Download temperature/salinity time series.
  3. **Remember the actual purpose of this one:** before using it as a training input, check GLORYS's own documented assimilation sources (Copernicus Marine's product documentation) to determine whether RAMA data is already assimilated into GLORYS — if it isn't, this becomes our genuinely independent validation set, the TS-Cast-style check we planned.

---

## General MOSDAC registration process (applies to items 4, 2-alt, 5-alt above)
1. Go to `https://mosdac.gov.in/` and click "Sign Up."
2. Fill the registration form completely and submit.
3. Wait for an approval email (real user reports suggest this can take some time — **start this registration today, in parallel with everything else**, since it's the one step with an unpredictable wait).
4. Verify your email via the link sent.
5. Log in, use the catalog search to find the specific product, add to cart, place the order.
6. MOSDAC processes the order server-side and emails you when ready.
7. Retrieve data via `sftp://ftp.mosdac.gov.in` using your account credentials.

---

## Quick-reference summary table

| # | Dataset | Registration | Direct link |
|---|---|---|---|
| 1a | River discharge (satellite) | 🔓 None | data.ceda.ac.uk/neodc/esacci/river_discharge |
| 1b | River discharge (gauge, global) | 🔒 GRDC | grdc.bafg.de |
| 1c | Farakka/dam-specific | 🔒 India-WRIS | indiawris.gov.in |
| 2a | Precipitation (global) | 🔒 NASA Earthdata | disc.gsfc.nasa.gov (search IMERG) |
| 2b | Precipitation (India) | 🔒 MOSDAC | mosdac.gov.in |
| 3 | Latent heat flux | 🔓 None | oaflux.whoi.edu/data-access |
| 4 | Wind stress curl | 🔒 MOSDAC | mosdac.gov.in |
| 5a | Chlorophyll-a (global) | 🔒 NASA Earthdata | oceancolor.gsfc.nasa.gov |
| 5b | Chlorophyll-a (India) | 🔒 MOSDAC | mosdac.gov.in |
| 6 | Bathymetry | 🔓 None | download.gebco.net |
| 7 | Land mask | 🔓 None | naturalearthdata.com/downloads |
| 8a | ONI index | 🔓 None | origin.cpc.ncep.noaa.gov/.../ONI_v4.shtml |
| 8b | IOD/DMI index | 🔓 None | psl.noaa.gov/gcos_wgsp/Timeseries/DMI |
| 9 | IBTrACS cyclones | 🔓 None | ncei.noaa.gov/data/.../ibtracs/v04r01/access/csv |
| 10 | EN4 profiles | 🔒 Met Office | metoffice.gov.uk/hadobs/en4 |
| 11 | RAMA buoys | 🔓/🔒 varies | pmel.noaa.gov/tao/drupal/disdel |

## Suggested order of operations
1. **Start MOSDAC registration today** — it has the least predictable wait time and gates three separate items (2b, 4, 5b).
2. Grab everything marked 🔓 immediately — GEBCO, OAFlux, Natural Earth, ONI, IOD, IBTrACS, and the ESA CCI river discharge. Zero blockers, can be done this afternoon.
3. In parallel, register for GRDC and India-WRIS if you want the river discharge cross-checks/dam-specific data.
4. Confirm your NASA Earthdata access (likely already set up from SSS) covers IMERG and Ocean Color — usually the same account, no new registration needed.
5. EN4 and RAMA last — they're Tier 3, useful for validation rigor but not blocking the core build.
