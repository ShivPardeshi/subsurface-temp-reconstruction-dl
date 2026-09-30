import os
import glob
import xarray as xr
import zarr
import pandas as pd
import numpy as np

print('=== DETAILED DATE OVERLAP ANALYSIS ===')
raw = 'data/raw'
datasets = {
    'GLORYS (Target T/S)': ('glorys/*.nc', 'time'),
    'ARGO (In-situ Valid)': ('argo/*.nc', 'time'),
    'SST (OSTIA)': ('sst_sss_ssh_currents_winds/india_sst/*.nc', 'time'),
    'SSS (SMOS/SMAP)': ('sst_sss_ssh_currents_winds/india_sss/*.nc', 'time'),
    'SSH (DUACS)': ('sst_sss_ssh_currents_winds/india_ssh/*.nc', 'time'),
    'Precipitation (GPM)': ('precipitation/**/*.nc4', 'time'),
    'Heat Flux (ERA5 SLHF)': ('heat_flux/**/*.nc', 'valid_time'),
    'Wind Stress Curl': ('wind_curl/**/*.nc', 'time'),
}

for name, (pat, t_var) in datasets.items():
    fs = sorted(glob.glob(os.path.join(raw, pat), recursive=True))
    if not fs:
        print(f'{name}: NO FILES')
        continue
    try:
        if len(fs) == 1:
            ds = xr.open_dataset(fs[0])
            t = pd.to_datetime(ds[t_var].values)
            start_str = t[0].strftime('%Y-%m-%d')
            end_str = t[-1].strftime('%Y-%m-%d')
            print(f'{name}: {len(t)} steps | {start_str} to {end_str}')
            ds.close()
        else:
            ds0 = xr.open_dataset(fs[0])
            ds1 = xr.open_dataset(fs[-1])
            t0 = pd.to_datetime(ds0[t_var].values[0]).strftime('%Y-%m-%d')
            t1 = pd.to_datetime(ds1[t_var].values[-1]).strftime('%Y-%m-%d')
            print(f'{name}: {len(fs)} daily files | {t0} to {t1}')
            ds0.close()
            ds1.close()
    except Exception as e:
        print(f'{name}: error {e}')
print('======================================\n')


def check_nc(path, name):
    try:
        ds = xr.open_dataset(path, decode_times=False)
        t_var = 'time' if 'time' in ds.coords or 'time' in ds.dims else ('valid_time' if 'valid_time' in ds.coords else None)
        t_info = 'No time coord'
        if t_var:
            t_len = ds.dims.get(t_var, len(ds[t_var]))
            try:
                ds_dec = xr.open_dataset(path)
                t_min = str(ds_dec[t_var].values[0])[:19]
                t_max = str(ds_dec[t_var].values[-1])[:19]
                t_info = f'{t_var}: {t_len} steps ({t_min} to {t_max})'
                ds_dec.close()
            except Exception as e:
                t_info = f'{t_var}: {t_len} steps (raw min={ds[t_var].values[0]}, max={ds[t_var].values[-1]})'
        vars_list = list(ds.data_vars.keys())
        dims_dict = dict(ds.dims)
        print(f'{name}: dims={dims_dict}, {t_info}, vars={vars_list[:6]}')
        ds.close()
    except Exception as e:
        print(f'{name}: ERROR {e}')

print('--- RAW DATASETS ---')
for cat in ['glorys', 'argo', 'heat_flux', 'wind_curl', 'climate_indices', 'bathymetry', 'ibtracs']:
    fs = glob.glob(os.path.join('data/raw', cat, '**', '*.nc*'), recursive=True)
    if fs:
        check_nc(fs[0], cat.upper())
    else:
        print(f'{cat.upper()}: No NC files')

print('\n--- SST / SSS / SSH / WINDS / CURRENTS SUBDIRS ---')
base_winds = 'data/raw/sst_sss_ssh_currents_winds'
for item in os.listdir(base_winds):
    ip = os.path.join(base_winds, item)
    if os.path.isdir(ip):
        nc_files = glob.glob(os.path.join(ip, '*.nc*'))
        print(f'  {item}: {len(nc_files)} files')
        if nc_files:
            check_nc(nc_files[0], f'    sample_{item}')
    else:
        print(f'  file: {item}')

print('\n--- PRECIPITATION ---')
precip_files = sorted(glob.glob('data/raw/precipitation/**/*.nc*', recursive=True))
print(f'Total precipitation files: {len(precip_files)}')
if precip_files:
    check_nc(precip_files[0], 'Precip First')
    check_nc(precip_files[-1], 'Precip Last')

print('\n--- CHLOROPHYLL ---')
chla_files = sorted(glob.glob('data/raw/chlorophyll/**/*.nc*', recursive=True))
print(f'Total chlorophyll files: {len(chla_files)}')
if chla_files:
    check_nc(chla_files[0], 'Chl First')
    check_nc(chla_files[-1], 'Chl Last')

print('\n--- RIVER DISCHARGE ---')
rd_files = sorted(glob.glob('data/raw/river_discharge/**/*', recursive=True))
rd_files = [f for f in rd_files if os.path.isfile(f)]
print(f'Total river discharge files: {len(rd_files)}')
for f in rd_files[:5]:
    print('  ', os.path.basename(f))

print('\n--- PROCESSED DATASETS ---')
for pz in ['data/processed/oceanembed_datacube.zarr', 'data/processed/toy_datacube.zarr']:
    if os.path.exists(pz):
        try:
            ds = xr.open_zarr(pz)
            print(f'{pz}: dims={dict(ds.dims)}, time=({str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}), channels={len(ds.channel)}')
            print(f'  channel names: {list(ds.channel.values)}')
            ds.close()
        except Exception as e:
            print(f'{pz}: ERROR {e}')

for p2 in ['data/processed/phase2_dataset', 'data/processed/phase2_toy']:
    if os.path.exists(p2):
        print(f'{p2} contents: {os.listdir(p2)}')
        # Check sub Zarrs
        for sub in os.listdir(p2):
            sp = os.path.join(p2, sub)
            if os.path.isdir(sp) and (sp.endswith('.zarr') or 'zarr' in sub):
                try:
                    ds = xr.open_zarr(sp)
                    t_str = f'time=({str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]})' if 'time' in ds.coords else 'no time'
                    print(f'  {sub}: dims={dict(ds.dims)}, {t_str}, vars={list(ds.data_vars.keys())}')
                    ds.close()
                except Exception as e:
                    print(f'  {sub}: zarr open err: {e}')
