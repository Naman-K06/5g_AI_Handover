'''
import numpy as np
import sys
import os

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FC_GHZ, H_BS, H_UT, TX_POWER_DBM, NOISE_FLOOR_DBM, C_LIGHT,
    ANTENNA_GAIN_MAX, PHI_3DB, THETA_3DB, A_MAX, DOWNTILT_DEG,
    SHADOWING_STD_LOS, SHADOWING_STD_NLOS, SHADOWING_CORRELATION
)

# FIX: Derive the AR(1) innovation coefficient from config instead of hardcoding 0.57.
# For an AR(1) process: x[t] = rho * x[t-1] + sqrt(1 - rho^2) * noise
# This ensures the process stays correct if SHADOWING_CORRELATION is tuned.
_SHADOWING_INNOVATION = np.sqrt(1.0 - SHADOWING_CORRELATION ** 2)  # ≈ 0.5724 for rho=0.82


# -------------------
# 1. 3D ANTENNA GAIN (3GPP TR 38.901 Table 7.3-1)
# -------------------
def get_3d_antenna_gain(phi_ue_deg, theta_ue_deg):
    """
    3GPP TR 38.901 Table 7.3-1: Combined 3D antenna pattern.
    phi_ue_deg  : Horizontal angle relative to boresight (degrees)
    theta_ue_deg: Vertical elevation angle relative to horizon (degrees)
    """
    # Horizontal Pattern
    phi = (phi_ue_deg + 180) % 360 - 180
    a_phi = -min(12 * (phi / PHI_3DB) ** 2, A_MAX)

    # Vertical Pattern (accounts for mechanical downtilt)
    theta = theta_ue_deg - DOWNTILT_DEG
    a_theta = -min(12 * (theta / THETA_3DB) ** 2, A_MAX)

    # Combined 3D gain
    return ANTENNA_GAIN_MAX - min(-(a_phi + a_theta), A_MAX)


# -------------------
# 2. 5G THROUGHPUT MAPPING (SINR → SE → Mbps)
# -------------------
def get_5g_throughput(sinr_db, bandwidth_mhz=20):
    """
    Coarse MCS mapping to spectral efficiency (bits/s/Hz).
    Wider bins reduce sensitivity to small SINR fluctuations.
    """
    if sinr_db < -8:
        se = 0.0
    elif sinr_db < 2:
        se = 1.0   # QPSK low-order MCS
    elif sinr_db < 12:
        se = 3.0   # 16-QAM / 64-QAM mid-tier
    else:
        se = 6.0   # 256-QAM high-tier

    return se * bandwidth_mhz * 1e6  # bits/s


# -------------------
# 3. 3GPP UMa LOS PROBABILITY (TR 38.901 Table 7.4.2-1)
# -------------------
def get_uma_los_prob(d_2d, h_ut):
    if d_2d <= 18:
        return 1.0
    c_prime = 0 if h_ut <= 13 else ((h_ut - 13) / 10) ** 1.5
    p_los = (18 / d_2d) + np.exp(-d_2d / 63) * (1 - 18 / d_2d)
    if h_ut > 13:
        p_los *= (1 + c_prime * (5 / 4) * (d_2d / 100) ** 3 * np.exp(-d_2d / 150))
    return np.clip(p_los, 0, 1)


# -------------------
# 4. 3GPP UMa PATH LOSS (TR 38.901 Table 7.4.1-1)
# -------------------
def get_3gpp_uma_pl(d_2d, d_3d, is_los):
    h_e = 1.0
    f_c = FC_GHZ
    d_bp_prime = 4 * (H_BS - h_e) * (H_UT - h_e) * (f_c * 1e9) / C_LIGHT

    if is_los:
        if 10 <= d_2d <= d_bp_prime:
            return 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_c)
        return (28.0 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(f_c)
                - 9.0 * np.log10(d_bp_prime ** 2 + (H_BS - H_UT) ** 2))

    pl_los = get_3gpp_uma_pl(d_2d, d_3d, is_los=True)
    pl_nlos = 13.54 + 39.08 * np.log10(d_3d) + 20.0 * np.log10(f_c) - 0.6 * (H_UT - 1.5)
    return max(pl_los, pl_nlos)


# -------------------
# 5. MAIN NETWORK STATE FUNCTION
# -------------------
def get_network_state(user_pos, bs_coords, prev_shadowing=None, cell_loads=None):
    """
    Compute RSRP, SINR, and throughput for all cells at the given UE position.

    Parameters
    ----------
    user_pos       : (2,) array — UE (x, y) in the digital twin grid (metres)
    bs_coords      : (N_sites, 2) array — base station (x, y) positions
    prev_shadowing : (N_cells,) array or None — correlated shadowing from last step
    cell_loads     : (N_cells,) array or None — fractional load [0, 1] per cell

    Returns
    -------
    dict with keys: rsrp, sinr, throughput, shadowing, path_loss
    """
    num_sites = len(bs_coords)
    num_cells = num_sites * 3  # 3 sectors per site

    if cell_loads is None:
        cell_loads = np.full(num_cells, 0.5)

    site_indices   = np.repeat(np.arange(num_sites), 3)
    cell_boresights = np.tile([0, 120, 240], num_sites)  # degrees

    dx = user_pos[0] - bs_coords[site_indices, 0]
    dy = user_pos[1] - bs_coords[site_indices, 1]
    d_2d = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 10.0)
    d_3d = np.sqrt(d_2d ** 2 + (H_BS - H_UT) ** 2)

    # Angles for 3D antenna pattern
    phi_ue_deg   = np.degrees(np.arctan2(dy, dx))           # azimuth
    theta_ue_deg = np.degrees(np.arctan2(H_BS - H_UT, d_2d))  # elevation

    # LOS determination
    p_los  = np.array([get_uma_los_prob(d, H_UT) for d in d_2d])
    is_los = np.random.random(num_cells) < p_los

    # Path loss
    path_loss = np.array([
        get_3gpp_uma_pl(d_2d[i], d_3d[i], is_los[i]) for i in range(num_cells)
    ])

    # 3D antenna gain
    ant_gain = np.array([
        get_3d_antenna_gain(phi_ue_deg[i] - cell_boresights[i], theta_ue_deg[i])
        for i in range(num_cells)
    ])

    # Correlated shadowing — FIX: use derived innovation coefficient, not hardcoded 0.57
    sigma_sf     = np.where(is_los, SHADOWING_STD_LOS, SHADOWING_STD_NLOS)
    fresh_noise  = np.random.normal(0, sigma_sf)
    if prev_shadowing is None:
        current_shadowing = fresh_noise
    else:
        current_shadowing = (SHADOWING_CORRELATION * prev_shadowing
                             + _SHADOWING_INNOVATION * fresh_noise)

    # RSRP (dBm)
    rsrp          = TX_POWER_DBM + ant_gain - path_loss + current_shadowing
    rsrp_linear   = 10 ** ((rsrp - 30) / 10)          # watts
    noise_linear  = 10 ** ((NOISE_FLOOR_DBM - 30) / 10)

    # SINR — interference from all cells except the serving cell
    sinr_db       = np.zeros(num_cells)
    throughput_bps = np.zeros(num_cells)

    for i in range(num_cells):
        # Interference: sum of power from every OTHER cell, weighted by load
        interference = np.sum([
            rsrp_linear[j] * cell_loads[j]
            for j in range(num_cells) if j != i
        ])
        sinr_val   = 10 * np.log10(
            np.maximum(rsrp_linear[i] / (interference + noise_linear), 1e-10)
        )
        sinr_db[i]        = np.clip(sinr_val, -20, 40)
        throughput_bps[i] = get_5g_throughput(sinr_db[i])

    return {
        "rsrp":       rsrp,
        "sinr":       sinr_db,
        "throughput": throughput_bps,
        "shadowing":  current_shadowing,
        "path_loss":  path_loss,
    }

'''

import numpy as np
import sys
import os

# Import configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FC_GHZ, H_BS, H_UT, TX_POWER_DBM, NOISE_FLOOR_DBM, C_LIGHT,
    ANTENNA_GAIN_MAX, PHI_3DB, THETA_3DB, A_MAX, DOWNTILT_DEG,
    SHADOWING_STD_LOS, SHADOWING_STD_NLOS, SHADOWING_CORRELATION
)

# Derive the AR(1) innovation coefficient from config instead of hardcoding.
# For an AR(1) process: x[t] = rho * x[t-1] + sqrt(1 - rho^2) * noise
# Ensures correctness if SHADOWING_CORRELATION is tuned.
_SHADOWING_INNOVATION = np.sqrt(1.0 - SHADOWING_CORRELATION ** 2)  # ≈ 0.5724 for rho=0.82


# -------------------
# 1. 3D ANTENNA GAIN (3GPP TR 38.901 Table 7.3-1)
# -------------------
def get_3d_antenna_gain(phi_ue_deg, theta_ue_deg):
    """
    3GPP TR 38.901 Table 7.3-1: Combined 3D antenna pattern.
    phi_ue_deg  : Horizontal angle relative to boresight (degrees)
    theta_ue_deg: Vertical elevation angle relative to horizon (degrees)
    """
    phi = (phi_ue_deg + 180) % 360 - 180
    a_phi = -min(12 * (phi / PHI_3DB) ** 2, A_MAX)

    theta = theta_ue_deg - DOWNTILT_DEG
    a_theta = -min(12 * (theta / THETA_3DB) ** 2, A_MAX)

    return ANTENNA_GAIN_MAX - min(-(a_phi + a_theta), A_MAX)


# -------------------
# 2. 5G THROUGHPUT MAPPING (SINR → SE → Mbps)
# -------------------
def get_5g_throughput(sinr_db, bandwidth_mhz=20):
    """
    Coarse MCS mapping to spectral efficiency (bits/s/Hz).
    Wider bins reduce sensitivity to small SINR fluctuations.
    """
    if sinr_db < -8:
        se = 0.0
    elif sinr_db < 2:
        se = 1.0   # QPSK low-order MCS
    elif sinr_db < 12:
        se = 3.0   # 16-QAM / 64-QAM mid-tier
    else:
        se = 6.0   # 256-QAM high-tier

    return se * bandwidth_mhz * 1e6  # bits/s


# -------------------
# 3. 3GPP UMa LOS PROBABILITY (TR 38.901 Table 7.4.2-1)
# -------------------
def get_uma_los_prob(d_2d, h_ut):
    if d_2d <= 18:
        return 1.0
    c_prime = 0 if h_ut <= 13 else ((h_ut - 13) / 10) ** 1.5
    p_los = (18 / d_2d) + np.exp(-d_2d / 63) * (1 - 18 / d_2d)
    if h_ut > 13:
        p_los *= (1 + c_prime * (5 / 4) * (d_2d / 100) ** 3 * np.exp(-d_2d / 150))
    return np.clip(p_los, 0, 1)


# -------------------
# 4. 3GPP UMa PATH LOSS (TR 38.901 Table 7.4.1-1)
# -------------------
def get_3gpp_uma_pl(d_2d, d_3d, is_los):
    h_e = 1.0
    f_c = FC_GHZ
    d_bp_prime = 4 * (H_BS - h_e) * (H_UT - h_e) * (f_c * 1e9) / C_LIGHT

    if is_los:
        if 10 <= d_2d <= d_bp_prime:
            return 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_c)
        return (28.0 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(f_c)
                - 9.0 * np.log10(d_bp_prime ** 2 + (H_BS - H_UT) ** 2))

    pl_los = get_3gpp_uma_pl(d_2d, d_3d, is_los=True)
    pl_nlos = 13.54 + 39.08 * np.log10(d_3d) + 20.0 * np.log10(f_c) - 0.6 * (H_UT - 1.5)
    return max(pl_los, pl_nlos)


# -------------------
# 5. MAIN NETWORK STATE FUNCTION
# -------------------
def get_network_state(user_pos, bs_coords, prev_shadowing=None, cell_loads=None):
    """
    Compute RSRP, SINR, and throughput for all cells at the given UE position.

    Parameters
    ----------
    user_pos       : (2,) array — UE (x, y) in the digital twin grid (metres)
    bs_coords      : (N_sites, 2) array — base station (x, y) positions
    prev_shadowing : (N_cells,) array or None — correlated shadowing from last step
    cell_loads     : (N_cells,) array or None — fractional load [0, 1] per cell

    Returns
    -------
    dict with keys: rsrp, sinr, throughput, shadowing, path_loss
    """
    num_sites = len(bs_coords)
    num_cells = num_sites * 3  # 3 sectors per site

    if cell_loads is None:
        cell_loads = np.full(num_cells, 0.5)

    site_indices    = np.repeat(np.arange(num_sites), 3)
    cell_boresights = np.tile([0, 120, 240], num_sites)  # degrees

    dx   = user_pos[0] - bs_coords[site_indices, 0]
    dy   = user_pos[1] - bs_coords[site_indices, 1]
    d_2d = np.maximum(np.sqrt(dx ** 2 + dy ** 2), 10.0)
    d_3d = np.sqrt(d_2d ** 2 + (H_BS - H_UT) ** 2)

    phi_ue_deg   = np.degrees(np.arctan2(dy, dx))
    theta_ue_deg = np.degrees(np.arctan2(H_BS - H_UT, d_2d))

    p_los  = np.array([get_uma_los_prob(d, H_UT) for d in d_2d])
    is_los = np.random.random(num_cells) < p_los

    path_loss = np.array([
        get_3gpp_uma_pl(d_2d[i], d_3d[i], is_los[i]) for i in range(num_cells)
    ])

    ant_gain = np.array([
        get_3d_antenna_gain(phi_ue_deg[i] - cell_boresights[i], theta_ue_deg[i])
        for i in range(num_cells)
    ])

    sigma_sf    = np.where(is_los, SHADOWING_STD_LOS, SHADOWING_STD_NLOS)
    fresh_noise = np.random.normal(0, sigma_sf)
    if prev_shadowing is None:
        current_shadowing = fresh_noise
    else:
        current_shadowing = (SHADOWING_CORRELATION * prev_shadowing
                             + _SHADOWING_INNOVATION * fresh_noise)

    # RSRP (dBm)
    rsrp         = TX_POWER_DBM + ant_gain - path_loss + current_shadowing
    rsrp_linear  = 10 ** ((rsrp - 30) / 10)       # watts
    noise_linear = 10 ** ((NOISE_FLOOR_DBM - 30) / 10)

    # --- VECTORIZED SINR (replaces O(N²) Python loop) ---
    # Total interference power (all cells weighted by load)
    total_interference = np.dot(rsrp_linear, cell_loads)  # scalar

    # Per-cell SINR: subtract own contribution from total interference
    own_interference  = rsrp_linear * cell_loads
    interference_excl = total_interference - own_interference  # (num_cells,)

    sinr_linear  = rsrp_linear / (interference_excl + noise_linear)
    sinr_db      = np.clip(10 * np.log10(np.maximum(sinr_linear, 1e-10)), -20, 40)

    throughput_bps = np.array([get_5g_throughput(s) for s in sinr_db])

    return {
        "rsrp":       rsrp,
        "sinr":       sinr_db,
        "throughput": throughput_bps,
        "shadowing":  current_shadowing,
        "path_loss":  path_loss,
    }