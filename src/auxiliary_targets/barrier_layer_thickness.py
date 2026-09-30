"""
Bay of Bengal Barrier Layer Thickness (BLT) computation module.
BLT = Isothermal Layer Depth (ILD) - Mixed Layer Depth (density-based).
Masked to Bay of Bengal (BoB region membership > 0.5).
"""

import numpy as np
from src.auxiliary_targets.mixed_layer_depth import compute_mld_profile

# Seawater equation of state constants (linear approximation)
RHO_0 = 1025.0              # kg/m^3
ALPHA = 2.5e-4              # 1/K (thermal expansion coefficient)
BETA = 7.5e-4               # 1/psu (haline contraction coefficient)

def compute_potential_density(temp_celsius: np.ndarray, salinity_psu: np.ndarray) -> np.ndarray:
    """
    Computes approximate potential density anomaly sigma_theta (kg/m^3) using linear equation of state:
      sigma_theta = rho_0 * (-alpha * (T - 15) + beta * (S - 35))
    """
    return RHO_0 * (-ALPHA * (temp_celsius - 15.0) + BETA * (salinity_psu - 35.0))

def compute_barrier_layer_profile(
    temp_profile: np.ndarray,
    salinity_profile: np.ndarray,
    depths: np.ndarray,
    ref_depth: float = 10.0,
    delta_t: float = 0.2
) -> float:
    """
    Computes Barrier Layer Thickness (BLT = ILD - MLD_rho) for a 1D vertical profile.
    """
    if len(temp_profile) < 2 or np.all(np.isnan(temp_profile)):
        return 0.0

    # 1. Isothermal Layer Depth (ILD)
    ild = compute_mld_profile(temp_profile, depths, delta_t_threshold=delta_t, ref_depth=ref_depth)

    # 2. Density-equivalent threshold: delta_sigma = rho_0 * alpha * delta_t
    delta_sigma_threshold = RHO_0 * ALPHA * delta_t # ~0.05125 kg/m^3

    # Density profile
    density_prof = compute_potential_density(temp_profile, salinity_profile)

    # Find 10m density
    ref_idx = int(np.argmin(np.abs(depths - ref_depth)))
    rho_ref = density_prof[ref_idx]

    # Find MLD_density (continuous interpolation)
    mld_rho = depths[-1]
    for i in range(ref_idx, len(depths) - 1):
        d_curr = density_prof[i]
        d_next = density_prof[i + 1]
        z_curr = depths[i]
        z_next = depths[i + 1]

        diff_curr = d_curr - rho_ref
        diff_next = d_next - rho_ref

        if diff_next >= delta_sigma_threshold:
            if np.abs(diff_next - diff_curr) < 1e-6:
                mld_rho = z_curr
            else:
                frac = (delta_sigma_threshold - diff_curr) / (diff_next - diff_curr)
                frac = np.clip(frac, 0.0, 1.0)
                mld_rho = z_curr + frac * (z_next - z_curr)
            break

    # BLT = max(0, ILD - MLD_density)
    blt = max(0.0, float(ild - mld_rho))
    return blt

def compute_barrier_layer_field(
    temp_3d: np.ndarray,
    salinity_3d: np.ndarray,
    depths: np.ndarray,
    bob_mask: np.ndarray
) -> np.ndarray:
    """
    Computes 2D BLT field, masked to Bay of Bengal (bob_mask > 0.5).
    """
    D, H, W = temp_3d.shape
    blt_field = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            if bob_mask[i, j] > 0.5:
                t_prof = temp_3d[:, i, j]
                s_prof = salinity_3d[:, i, j]
                blt_field[i, j] = compute_barrier_layer_profile(t_prof, s_prof, depths)

    return blt_field
