"""2D Fourier Spectral Analysis and Oversmoothing Diagnostic.

Computes radially averaged power spectra (PSD) of spatial fields per depth to
detect whether the model loses fine-scale mesoscale eddies and features
(oversmoothing diagnostic, per Asefi et al. and PS26066 Phase 5 Spec).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from .basic_metrics import _to_numpy


def compute_radial_power_spectrum(
    field: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radially averaged 1D Fourier Power Spectral Density (PSD).

    Args:
        field: 2D array of shape (H, W).
        mask: Optional 2D boolean mask (True = valid ocean).

    Returns:
        wavenumbers: 1D array of radial wavenumber indices.
        radial_psd: 1D array of radially integrated power at each wavenumber.
    """
    f = _to_numpy(field).copy()
    h, w = f.shape

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        valid = m & np.isfinite(f)
        ocean_mean = float(np.nanmean(f[valid])) if np.any(valid) else 0.0
        f[~valid] = ocean_mean
        f = f - ocean_mean
        # Apply 2D cosine bell (Hanning window) to ocean boundary to reduce spectral leakage
        win_y = np.hanning(h)[:, None]
        win_x = np.hanning(w)[None, :]
        f = f * (win_y * win_x)
    else:
        f = f - np.nanmean(f)
        f = np.nan_to_num(f)
        win_y = np.hanning(h)[:, None]
        win_x = np.hanning(w)[None, :]
        f = f * (win_y * win_x)

    # 2D Fast Fourier Transform
    fft2 = np.fft.fftshift(np.fft.fft2(f))
    power_2d = np.abs(fft2) ** 2 / (h * w)

    # Center coordinates
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    # Maximum valid radius
    max_r = min(cy, cx)
    radial_psd = np.zeros(max_r, dtype=np.float64)

    for radius in range(max_r):
        ring_mask = r == radius
        if np.any(ring_mask):
            radial_psd[radius] = np.mean(power_2d[ring_mask])

    wavenumbers = np.arange(max_r)
    return wavenumbers, radial_psd


def compare_spectra(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    depths: Optional[List[float]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Dict[str, Union[np.ndarray, float]]]:
    """Compare Fourier power spectra of predicted and ground truth fields across depths.

    Computes:
    - 'oversmoothing_index': Energy deficit in upper 25% wavenumbers.
      Values > 0 indicate model loses high-wavenumber fine structure.
    - Full radial spectra arrays for plotting.
    """
    p = _to_numpy(pred)
    t = _to_numpy(target)

    if p.ndim == 4:
        p = np.mean(p, axis=0)
        t = np.mean(t, axis=0)

    num_depths = p.shape[0]
    results = {}

    for d in range(num_depths):
        p_d = p[d]
        t_d = t[d]
        depth_key = f"{depths[d]}m" if depths and d < len(depths) else f"depth_{d}"

        k, psd_p = compute_radial_power_spectrum(p_d, mask=mask)
        _, psd_t = compute_radial_power_spectrum(t_d, mask=mask)

        # High-wavenumber range (top 25% of spectrum)
        cutoff = int(0.75 * len(k))
        energy_p_high = np.sum(psd_p[cutoff:])
        energy_t_high = np.sum(psd_t[cutoff:])

        if energy_t_high > 1e-9:
            oversmoothing = float(1.0 - (energy_p_high / energy_t_high))
        else:
            oversmoothing = 0.0

        results[depth_key] = {
            "wavenumbers": k,
            "psd_pred": psd_p,
            "psd_true": psd_t,
            "oversmoothing_index": oversmoothing,
        }

    return results
