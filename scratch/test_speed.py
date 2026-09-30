import os, sys, time
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.data.harmonize.build_datacube import DataCubeBuilder

t0 = time.time()
builder = DataCubeBuilder(toy_mode=False)
t_init = time.time()
print(f"Builder init took {t_init - t0:.2f}s")

# Profile breakdown of single day
d = "2025-01-01"
dt = pd.to_datetime(d)

steps = [
    ("SST", lambda: builder._load_sst(dt)),
    ("SSS", lambda: builder._load_sss(dt)),
    ("SSH", lambda: builder._load_ssh(dt)),
    ("Winds", lambda: builder._load_winds(dt)),
    ("Currents", lambda: builder._load_currents(dt)),
    ("Wind Curl", lambda: builder._load_wind_curl(dt)),
    ("Precip", lambda: builder._load_precipitation(dt)),
    ("Heat Flux", lambda: builder._load_heat_flux(dt)),
    ("Chlorophyll", lambda: builder._load_chlorophyll(dt)),
]

for name, fn in steps:
    t_step = time.time()
    res = fn()
    print(f"  {name}: {time.time() - t_step:.3f}s")

t1 = time.time()
dates = ["2025-01-01", "2025-01-02", "2025-01-03"]
for d in dates:
    td0 = time.time()
    res = builder.assemble_day(d)
    print(f"Day {d} total: {time.time() - td0:.3f}s")
print(f"Total for 3 days: {time.time() - t1:.2f}s")
