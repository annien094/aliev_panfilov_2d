"""
Tissue heterogeneity, other pacing protocols and dataset generators for the 2D
Aliev-Panfilov solver.

This file builds on solveAP_2D_jax.py (the solver, phie and video code), which
must be in the same folder. It is not needed for planar waves on homogeneous
tissue.

Contents
--------
    1. Other pacing protocols          random pacing sites, spiral-wave (cross-field) stimulus
    2. Patch generators                rectangular and irregular patches
    3. Single runs with patches        run_single_simulation, run_simulations_same_patches
    4. Batch datasets                  generate_batch_simulations, generate_heightened_excitability_batch
    5. Homogeneous reference dataset   generate_no_fibrosis_dataset, build_phie_matrix_dense

A patch is a region whose tissue properties differ from the baseline: fibrotic
(lower D, higher a, lower b) or with heightened excitability (negative a and/or
higher k). Patches reach the solver through the advanced fields of Params
(D_matrix, a_field, b_field, k_field), which all include the ghost border.
"""
#%%
# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations

import gc
import os
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path
from scipy.signal import convolve2d

from solveAP_2D_jax import (
    MS_PER_TU,
    Params,
    _grad_central,
    calc_phie_jax,
    make_electrodes_from_domain,
    save_VW_video,
    simulate,
)


#%%
# =============================================================================
# 1. OTHER PACING PROTOCOLS
# =============================================================================
def generate_random_stim_mask(
    ncells: int,
    stim_size: int = 10,
    seed: Optional[int] = None,
    forbidden_mask: Optional[np.ndarray] = None,
    max_tries: int = 500,
) -> np.ndarray:
    """Generate a random stimulation mask with a square patch, avoiding forbidden areas."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    half_size = stim_size // 2

    if forbidden_mask is not None:
        forbidden_mask = np.asarray(forbidden_mask, dtype=bool)
        if forbidden_mask.shape != (ncells, ncells):
            raise ValueError("forbidden_mask must have shape (ncells, ncells)")

    # Generate random center position, ensuring the stim patch fits within boundaries
    centre_stim_x = None
    centre_stim_y = None
    for _ in range(max_tries):
        cx = rng.integers(half_size, ncells - half_size)
        cy = rng.integers(half_size, ncells - half_size)
        if forbidden_mask is None:
            centre_stim_x, centre_stim_y = cx, cy
            break
        patch = forbidden_mask[cx - half_size:cx + half_size, cy - half_size:cy + half_size]
        if not patch.any():
            centre_stim_x, centre_stim_y = cx, cy
            break

    if centre_stim_x is None:
        raise ValueError("Failed to place stimulus outside fibrotic areas; increase max_tries or reduce stim_size.")

    stim_mask[centre_stim_x - half_size:centre_stim_x + half_size,
              centre_stim_y - half_size:centre_stim_y + half_size] = 1.0

    return stim_mask, (centre_stim_x, centre_stim_y)


def generate_random_stim_masks_for_cycles(
    ncells: int,
    ncyc: int,
    stim_size: int = 10,
    seed: Optional[int] = None,
    forbidden_mask: Optional[np.ndarray] = None,
    max_tries_per_cycle: int = 500,
) -> Tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate one random stimulation mask per cycle with distinct, non-overlapping locations."""
    if ncyc < 1:
        raise ValueError(f"ncyc must be >= 1, got {ncyc}")

    rng = np.random.default_rng(seed)
    stim_masks = np.zeros((ncyc, ncells, ncells), dtype=np.float32)
    stim_centers: list[tuple[int, int]] = []

    if forbidden_mask is not None:
        forbidden_mask = np.asarray(forbidden_mask, dtype=bool)
        if forbidden_mask.shape != (ncells, ncells):
            raise ValueError("forbidden_mask must have shape (ncells, ncells)")
    else:
        forbidden_mask = np.zeros((ncells, ncells), dtype=bool)

    half_size = stim_size // 2
    used_mask = np.zeros((ncells, ncells), dtype=bool)

    for cyc in range(ncyc):
        placed = False
        for _ in range(max_tries_per_cycle):
            cx = int(rng.integers(half_size, ncells - half_size))
            cy = int(rng.integers(half_size, ncells - half_size))

            row_slice = slice(cx - half_size, cx + half_size)
            col_slice = slice(cy - half_size, cy + half_size)
            if forbidden_mask[row_slice, col_slice].any():
                continue
            if used_mask[row_slice, col_slice].any():
                continue

            stim_masks[cyc, row_slice, col_slice] = 1.0
            used_mask[row_slice, col_slice] = True
            stim_centers.append((cx, cy))
            placed = True
            break

        if not placed:
            raise ValueError(
                "Failed to place distinct stimulus masks for all cycles; "
                "increase max_tries_per_cycle or reduce stim_size."
            )

    return stim_masks, stim_centers


def generate_spiral_wave_stim_masks(ncells: int,
                                    planar_width: int = 5,
                                    cross_width: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate stimulation masks for spiral wave generation.

    Two stimuli are used:
    - First stimulus: planar wave from top (stimulates top rows)
    - Cross-field stimulus: wave from left (stimulates left columns)

    Parameters:
    -----------
    ncells : int
        Interior grid size (ncells x ncells)
    planar_width : int
        Number of rows to stimulate at the top for planar wave (default: 5)
    cross_width : int, optional
        Width of cross-field stimulus from left. If None, uses ncells//3

    Returns:
    --------
    stim_mask : (ncells, ncells) ndarray
        Primary stimulation mask for planar wave (top rows)
    cross_stim_mask : (ncells, ncells) ndarray
        Cross-field stimulation mask (left columns)

    Example:
    --------
    >>> stim_mask, cross_stim_mask = generate_spiral_wave_stim_masks(100, planar_width=5)
    >>> params = Params(..., stim_mask=stim_mask, cross_stim_mask=cross_stim_mask,
    ...                 cross_stim_time=42.0, cross_stim_duration=1.0)
    """
    if cross_width is None:
        cross_width = ncells // 3

    # Primary stimulus: top rows (planar wave going downward)
    stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    stim_mask[:planar_width, :] = 1.0

    # Cross-field stimulus: left columns (wave going rightward)
    cross_stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    cross_stim_mask[:, :cross_width] = 1.0

    return stim_mask, cross_stim_mask


#%%
# =============================================================================
# 2. PATCH GENERATORS
# =============================================================================
# -------------------------------------------------------------------------
# Rectangular patches
# -------------------------------------------------------------------------
def make_D_with_rect_patches(ncells: int, D0: float, Dfac: float, npatches: int,
                             patch_w_rng=(15, 25), patch_h_rng=(15, 25),
                             margin=7, min_sep=5, seed: int = 0) -> Tuple[np.ndarray, list]:
    """
    Create D_matrix with ghost border and non-overlapping rectangular patches in the interior.

    Parameters:
    -----------
    ncells : int
        Interior grid size (domain will be ncells+2 to include ghost cells)
    D0 : float
        Baseline diffusion coefficient
    Dfac : float
        Diffusion reduction factor (D_fibrotic = D0 * Dfac)
    npatches : int
        Number of rectangular patches
    patch_w_rng, patch_h_rng : tuple
        (min, max) range for patch width and height
    margin : int
        Minimum distance from domain boundary
    min_sep : int
        Minimum separation between patches
    seed : int
        Random seed

    Returns:
    --------
    D_matrix : (X, X) array where X=ncells+2
        Diffusion coefficient map with ghost border
    fiblocs : list of arrays
        Each element is (N_patch, 2) array of [row, col] coordinates (0-based indices)

    Note:
    -----
    Arrays use (row, col) indexing where:
    - row index increases from 0 to X-1 (bottom to top when plotted)
    - col index increases from 0 to X-1 (left to right when plotted)
    """
    rng = np.random.default_rng(seed)
    X = ncells + 2
    D = np.full((X, X), D0, dtype=np.float32)

    occ = np.zeros((X, X), dtype=bool)
    fiblocs: list[np.ndarray] = []

    n = 0
    while n < npatches:
        w = rng.integers(patch_w_rng[0], patch_w_rng[1] + 1)
        h = rng.integers(patch_h_rng[0], patch_h_rng[1] + 1)

        # choose bottom-left in interior, keep away from boundaries/ghosts
        i0 = rng.integers(margin, ncells - w - margin + 1)
        j0 = rng.integers(margin, ncells - h - margin + 1)

        I = np.arange(i0, i0 + w + 1)  # inclusive range: the patch spans (w+1) x (h+1) cells
        J = np.arange(j0, j0 + h + 1)

        # separation check (in full-grid coords)
        p = min_sep
        r0 = max(0, I[0] - p)
        r1 = min(X, I[-1] + p + 1)
        c0 = max(0, J[0] - p)
        c1 = min(X, J[-1] + p + 1)

        if not occ[r0:r1, c0:c1].any():
            # Mark occupied region
            occ[np.ix_(I, J)] = True
            D[np.ix_(I, J)] = D0 * Dfac

            # Store pixel coordinates [row, col] for this patch
            patch_rows, patch_cols = np.meshgrid(I, J, indexing='ij')
            fiblocs.append(np.stack([patch_rows.ravel(), patch_cols.ravel()], axis=1).astype(np.int32))
            n += 1

    return D, fiblocs


# -------------------------------------------------------------------------
# Irregularly shaped patches
# Caution ⚠️: This easily blows up the simulation with Dfac=0.1, but Dfac>=0.2 seems okay.
# -------------------------------------------------------------------------
def _poly_area(x: np.ndarray, y: np.ndarray) -> float:
    # Shoelace (x,y as 1D arrays, polygon assumed closed implicitly)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _dilate_mask(M: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return M
    k = np.ones((2 * pad + 1, 2 * pad + 1), dtype=np.uint8)
    return convolve2d(M.astype(np.uint8), k, mode="same", boundary="fill", fillvalue=0) > 0


def _estimate_aspect(mask: np.ndarray) -> float:
    pts = np.argwhere(mask)  # (N,2) [x(row), y(col)]
    if pts.shape[0] < 3:
        return 1.0
    pts = pts.astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)
    C = (pts.T @ pts) / pts.shape[0]
    s = np.sqrt(np.linalg.eigvalsh(C))
    s = np.sort(s)[::-1]
    return float(s[0] / max(s[1], 1e-12))


def irregular_shape_generation(ncells: int, D0: float, Dfac: float, npatches: int, seed: int | None = None):
    """
    Generate irregularly shaped fibrotic patches.

    Parameters:
    -----------
    ncells : int
        Interior grid size (domain will be ncells+2 to include ghost cells)
    D0 : float
        Baseline diffusion coefficient
    Dfac : float
        Diffusion reduction factor (D_fibrotic = D0 * Dfac)
    npatches : int
        Number of patches to generate
    seed : int or None
        Random seed

    Returns:
    --------
    D_matrix : (X, X) array where X=ncells+2
        Diffusion coefficient map with ghost border
    fiblocs : list of arrays
        Each element is (N_patch, 2) array of [row, col] coordinates (0-based indices)

    Note:
    -----
    Arrays use (row, col) indexing where:
    - row index increases from 0 to X-1 (bottom to top when plotted)
    - col index increases from 0 to X-1 (left to right when plotted)
    """
    rng = np.random.default_rng(seed)

    X = ncells + 2
    Y = ncells + 2

    # Initialize D_matrix with baseline diffusion
    D_matrix = np.full((X, Y), D0, dtype=np.float32)

    # Patch shape and placement settings
    area_min, area_max = 300, 450
    max_tries = 800
    min_vertices, max_vertices = 6, 12
    p = 10  # minimum separation between patches
    margin = 5  # minimum distance from interior boundaries

    elong_prob = 0.45
    max_aspect = 2.0
    roughness = 0.3

    # Interior boundaries: indices 1 to ncells (in full grid coords)
    # With margin, patches must stay in [1+margin, ncells-margin+1] = [6, ncells-4] for ncells=100
    interior_min = 1 + margin  # e.g., 6
    interior_max = ncells + 1 - margin  # e.g., 97 for ncells=100

    mask_all = np.zeros((X, Y), dtype=bool)
    fiblocs: list[np.ndarray] = []

    # Grid of pixel centers in 1-based coordinates: pixel [i, j] sits at (i+1, j+1)
    # meshgrid with indexing='ij': Xq varies along axis 0 (rows), Yq varies along axis 1 (cols)
    # So pts_grid has points [x, y] where x is row-index, y is col-index
    Xq, Yq = np.meshgrid(np.arange(1, X + 1), np.arange(1, Y + 1), indexing="ij")
    pts_grid = np.stack([Xq.ravel(), Yq.ravel()], axis=1)

    n, tries = 0, 0
    while n < npatches and tries < max_tries:
        tries += 1

        A_tgt = int(rng.integers(area_min, area_max + 1))
        # Choose center within allowed interior region
        cx = int(rng.integers(interior_min + p, interior_max - p + 1))
        cy = int(rng.integers(interior_min + p, interior_max - p + 1))

        nv = int(rng.integers(min_vertices, max_vertices + 1))
        theta = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        r0 = np.sqrt(A_tgt / np.pi)
        r = r0 * (1.0 - roughness / 2.0 + roughness * rng.random(nv))

        # Base polygon around origin
        px = r * np.cos(theta)
        py = r * np.sin(theta)

        # Optional elongation (area-preserving)
        if rng.random() < elong_prob:
            s = float(np.exp(np.log(max_aspect) * rng.random()))  # log-uniform in [1,max_aspect]
            ang = float(2.0 * np.pi * rng.random())
            c, sA = np.cos(ang), np.sin(ang)
            R = np.array([[c, -sA], [sA, c]])
            S = np.array([[s, 0.0], [0.0, 1.0 / s]])              # det=1
            A = R @ S @ R.T
            XY = A @ np.vstack([px, py])
            px, py = XY[0], XY[1]

        # Rescale to hit target area
        A_now = _poly_area(px, py)
        if A_now <= 0:
            continue
        s_area = np.sqrt(A_tgt / A_now)
        px *= s_area
        py *= s_area

        # Translate to center (cx, cy)
        x = px + cx
        y = py + cy

        # Fit inside interior bounds by shrinking about (cx,cy) if needed
        # Must stay within [interior_min, interior_max]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        if xmin < interior_min or ymin < interior_min or xmax > interior_max or ymax > interior_max:
            eps = 1e-12
            sx = min((cx - interior_min) / max(eps, cx - xmin), (interior_max - cx) / max(eps, xmax - cx))
            sy = min((cy - interior_min) / max(eps, cy - ymin), (interior_max - cy) / max(eps, ymax - cy))
            sfit = max(0.0, min(sx, sy, 1.0))
            x = cx + sfit * (x - cx)
            y = cy + sfit * (y - cy)
            if x.min() < interior_min or y.min() < interior_min or x.max() > interior_max or y.max() > interior_max:
                continue

        # Rasterize polygon into mask using matplotlib Path
        poly = Path(np.stack([x, y], axis=1))
        inside = poly.contains_points(pts_grid, radius=1e-9)  # small radius includes boundary-ish
        mask = inside.reshape(X, Y)

        # Buffer check: forbid overlap within p pixels
        if (_dilate_mask(mask_all, p) & mask).any():
            continue

        # Accept patch: modify diffusion tensors inside mask
        D_matrix[mask] = D0 * Dfac

        mask_all |= mask

        # Store patch pixel coordinates: np.where returns (rows, cols)
        # Store as [row, col] coordinates (0-based indices)
        rows, cols = np.where(mask)
        fiblocs.append(np.stack([rows, cols], axis=1).astype(np.int32))

        asp = _estimate_aspect(mask)
        print(f"Patch {n+1}: {mask.sum()} px, aspect~{asp:.2f}, center=[{cx},{cy}]")
        n += 1

    if n < npatches:
        print(f"Warning: placed {n}/{npatches} patches. Consider reducing p/max_aspect or widening area range.")

    return D_matrix, fiblocs


#%%
# =============================================================================
# 3. SINGLE RUNS WITH PATCHES
# =============================================================================
def run_single_simulation(ncells: int,
                          patch_type: str,
                          npatches: int,
                          seed: int,
                          mode: str = 'fibrosis',
                          D0: float = 0.1,
                          Dfac_range: Tuple[float, float] = (0.2, 0.4),
                          a_factor_range: Tuple[float, float] = (1.10, 3.00),
                          b_factor_range: Tuple[float, float] = (0.80, 0.90),
                          neg_a_value: float = -0.025,
                          k_increase_range: Tuple[float, float] = (1.2, 2.0),
                          save_dir: Optional[str] = None,
                          return_data: bool = True) -> Dict:
    """
    Run a single AP simulation with either 'irregular' or 'rectangular' patches.

    Modes:
    ------
    mode='fibrosis' (default):
        Patches are fibrotic: D is reduced by Dfac, a is increased by a_factor,
        b is decreased by b_factor. Uses npatches as given.
    mode='heightened_excitability_neg_a'
       | 'heightened_excitability_higher_k'
       | 'heightened_excitability_both':
        Patches are heightened-excitability regions: D is unchanged, and
        inside the patch a is set to a fixed negative value (neg_a_value)
        and/or k is increased (factor in k_increase_range). npatches is
        forced to 1.

    Parameters:
    -----------
    ncells : int
        Grid size (interior)
    patch_type : str
        Either 'irregular' or 'rectangular'
    npatches : int
        Number of patches (ignored and forced to 1 for excitability modes)
    seed : int
        Random seed for reproducibility
    mode : str
        One of 'fibrosis', 'heightened_excitability_neg_a',
        'heightened_excitability_higher_k', 'heightened_excitability_both'
    D0 : float
        Baseline diffusion coefficient
    Dfac_range : tuple[float, float]
        Diffusion reduction factor range (fibrosis mode only).
    a_factor_range : tuple[float, float]
        Multiplicative increase of a inside fibrotic regions (fibrosis mode).
    b_factor_range : tuple[float, float]
        Multiplicative decrease of b inside fibrotic regions (fibrosis mode).
    neg_a_value : float
        Fixed (negative) value of a inside excitability patches, e.g. -0.025
        (chosen in [-0.03, -0.02]).
    k_increase_range : tuple[float, float]
        Multiplicative increase of k inside excitability patches (factor in [1.2, 2.0]).
    save_dir : str, optional
        Directory to save results. If None, results are not saved to disk.

    Returns:
    --------
    dict containing: Vsav, Wsav, t_sav, D_matrix, fiblocs, stim_center, params
    """
    valid_modes = (
        'fibrosis',
        'heightened_excitability_neg_a',
        'heightened_excitability_higher_k',
        'heightened_excitability_both',
    )
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    is_excitability = mode != 'fibrosis'
    if is_excitability:
        npatches = 1

    X = ncells + 2
    rng = np.random.default_rng(seed)

    # Baseline cell parameters (also used to build the heterogeneous fields below).
    a0 = 0.01
    b0 = 0.15
    k0 = 8.0

    # Sample per-simulation patch parameters.
    if is_excitability:
        # No D modification for excitability patches: pass Dfac=1.0 so
        # irregular_shape_generation leaves D_matrix == D0 inside the patch.
        Dfac = 1.0
        neg_a = mode in ('heightened_excitability_neg_a',
                         'heightened_excitability_both')
        higher_k = mode in ('heightened_excitability_higher_k',
                            'heightened_excitability_both')
        # Fixed negative a inside the patch. a_factor records the effective
        # multiplier (a_patch = a0 * a_factor = neg_a_value) so downstream
        # plumbing (saving / metadata) is unchanged.
        a_factor = (neg_a_value / a0) if neg_a else 1.0
        k_factor = (float(rng.uniform(k_increase_range[0], k_increase_range[1]))
                    if higher_k else 1.0)
        b_factor = 1.0
    else:
        Dfac = float(rng.uniform(Dfac_range[0], Dfac_range[1]))
        a_factor = float(rng.uniform(a_factor_range[0], a_factor_range[1]))
        b_factor = float(rng.uniform(b_factor_range[0], b_factor_range[1]))
        k_factor = 1.0

    # Generate patches
    if patch_type == 'irregular':
        D_matrix, fiblocs = irregular_shape_generation(ncells, D0=D0, Dfac=Dfac,
                                                       npatches=npatches, seed=seed)
    elif patch_type == 'rectangular':
        D_matrix, fiblocs = make_D_with_rect_patches(ncells, D0=D0, Dfac=Dfac,
                                                     npatches=npatches, seed=seed)
    else:
        raise ValueError(f"patch_type must be 'irregular' or 'rectangular', got {patch_type}")

    # Build a boolean patch mask in full-grid coordinates from fiblocs (works
    # for both fibrosis and excitability modes; D_matrix < D0 fails when Dfac=1).
    patch_mask_full = np.zeros((X, X), dtype=bool)
    for patch_coords in fiblocs:
        if len(patch_coords) > 0:
            patch_mask_full[patch_coords[:, 0], patch_coords[:, 1]] = True
    patch_mask_interior = patch_mask_full[1:-1, 1:-1]

    # Stimulation: avoid the patch region (fibrotic or excitability).
    ncyc = 3
    stim_masks, stim_centers = generate_random_stim_masks_for_cycles(
        ncells=ncells,
        ncyc=ncyc,
        stim_size=10,
        seed=seed + 1000,
        forbidden_mask=patch_mask_interior,
    )

    a_field = np.full((X, X), a0, dtype=np.float32)
    b_field = np.full((X, X), b0, dtype=np.float32)
    k_field = np.full((X, X), k0, dtype=np.float32)
    a_field[patch_mask_full] = a0 * a_factor
    b_field[patch_mask_full] = b0 * b_factor
    k_field[patch_mask_full] = k0 * k_factor

    # Create parameters
    p = Params(
        dt=0.01,
        tend=50*3,
        BCL=50.0,
        ncyc=ncyc,
        stimdur=2.0,
        gathert=10,
        ncells=ncells,
        h=0.1,
        k=k0,
        mu1=0.2,
        mu2=0.3,
        epsi=0.002,
        D_scalar=D0,
        a0=a0,
        b0=b0,
        a_field=a_field,
        b_field=b_field,
        k_field=k_field,
        D_matrix=D_matrix,
        stim_mask=stim_masks[0],
        stim_masks=stim_masks,
        stim_amp_scale=0.1,
        cross_stim_mask=None,
    )

    # Run simulation
    print(f"Running {patch_type} simulation {seed} (mode={mode})...")
    out = simulate(p)

    # Calculate phie (extracellular potential) using JAX
    print(f"  Calculating phie...")
    elecposX, elecposY = make_electrodes_from_domain(
        X=ncells+2, Y=ncells+2, numelec_x=10, numelec_y=10
    )
    phie, elecpos = calc_phie_jax(
        h=p.h,
        D_matrix=D_matrix,
        elecposX=np.array(elecposX),
        elecposY=np.array(elecposY),
        Vsav=out["Vsav"],
        vsav_layout="TXY",
    )

    result = {
        "Vsav": out["Vsav"],
        "Wsav": out["Wsav"],
        "t_sav": out["t_sav"],
        "D_matrix": D_matrix,
        "fiblocs": fiblocs,
        "stim_centers": stim_centers,
        "phie": phie,
        "elecpos": elecpos * p.h,   # same length unit as h, as in run_planar_simulation
        "Dfac": Dfac,
        "a_factor": a_factor,
        "b_factor": b_factor,
        "k_factor": k_factor,
        "params": p,
        "patch_type": patch_type,
        "mode": mode,
        "seed": seed
    }

    # Optionally save to disk
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"sim_{patch_type}_{mode}_{seed}.npz")

        # Convert fiblocs to object array if it's a list (irregular patches)
        # fiblocs is a list of arrays, each (N_patch, 2) with [row, col] coordinates
        fiblocs_save = result["fiblocs"]
        if isinstance(fiblocs_save, list):
            fiblocs_save = np.array(fiblocs_save, dtype=object)

        # savez_compressed: save several arrays into a single file in compressed .npz format.
        np.savez_compressed(
            filename,
            # Vsav=result["Vsav"],            # (T, ncells, ncells) - [time, row, col]
            # Wsav=result["Wsav"],            # (T, ncells, ncells) - [time, row, col]
            t_sav=result["t_sav"],          # (T,) - time values
            D_matrix=result["D_matrix"],    # (X, X) - [row, col] where X=ncells+2
            fiblocs=fiblocs_save,           # object array of (N_patch, 2) arrays - each [row, col]
            phie=result["phie"],            # (E, T) - [electrode, time]
            elecpos=result["elecpos"],      # (E, 2) - [electrode, (row, col)], in units of h, frame of the grid with ghost border
            stim_centers=np.array(result["stim_centers"], dtype=np.int32),  # (ncyc,2) - (row, col) per cycle
            patch_type=patch_type,
            mode=mode,
            seed=seed,
            ncells=ncells,
            D0=D0,
            Dfac=Dfac,
            a_factor=a_factor,
            b_factor=b_factor,
            k_factor=k_factor,
            npatches=npatches
        )
        print(f"  Saved to {filename}")

    if not return_data:
        # Return minimal metadata to avoid retaining large arrays in memory
        return {
            "patch_type": patch_type,
            "mode": mode,
            "seed": seed,
            "ncells": ncells,
            "D0": D0,
            "Dfac": Dfac,
            "a_factor": a_factor,
            "b_factor": b_factor,
            "k_factor": k_factor,
            "npatches": npatches,
            "save_dir": save_dir,
        }

    return result


def run_simulations_same_patches(ncells: int,
                                 patch_type: str,
                                 npatches: int,
                                 patch_seed: int,
                                 n_stim: int = 3,
                                 stim_seeds: Optional[list[int]] = None,
                                 D0: float = 0.1,
                                 Dfac: float = 0.2,
                                 save_dir: Optional[str] = None,
                                 return_data: bool = True) -> list:
    """
    Run multiple simulations with the same fibrotic patch configuration and
    different pacing stimulus locations.

    Parameters:
    -----------
    patch_seed : int
        Random seed used to generate the fibrotic patches (fixed across runs)
    n_stim : int
        Number of pacing locations to simulate (ignored if stim_seeds provided)
    stim_seeds : list[int], optional
        Seeds to generate distinct pacing locations. If None, uses
        [patch_seed + 1, patch_seed + 2, ...].

    Returns:
    --------
    list of dicts, each containing simulation results for a different pacing location
    """
    if n_stim < 1:
        raise ValueError(f"n_stim must be >= 1, got {n_stim}")

    # Generate fibrotic patches once
    if patch_type == "irregular":
        D_matrix, fiblocs = irregular_shape_generation(
            ncells, D0=D0, Dfac=Dfac, npatches=npatches, seed=patch_seed
        )
    elif patch_type == "rectangular":
        D_matrix, fiblocs = make_D_with_rect_patches(
            ncells, D0=D0, Dfac=Dfac, npatches=npatches, seed=patch_seed
        )
    else:
        raise ValueError(f"patch_type must be 'irregular' or 'rectangular', got {patch_type}")

    if stim_seeds is None:
        stim_seeds = [patch_seed + i + 1 for i in range(n_stim)]
    else:
        if len(stim_seeds) != n_stim:
            raise ValueError("stim_seeds length must match n_stim")

    results = []
    for i, stim_seed in enumerate(stim_seeds, start=1):
        fibrotic_mask = D_matrix[1:-1, 1:-1] < D0
        stim_mask, stim_center = generate_random_stim_mask(
            ncells, stim_size=10, seed=stim_seed, forbidden_mask=fibrotic_mask
        )

        p = Params(
            dt=0.01,
            tend=50*3,
            BCL=50.0,
            ncyc=3,
            stimdur=2.0,
            gathert=10,
            ncells=ncells,
            h=0.1,
            k=8.0,
            mu1=0.2,
            mu2=0.3,
            epsi=0.002,
            D_scalar=D0,
            a0=0.01,
            b0=0.15,
            D_matrix=D_matrix,
            stim_mask=stim_mask,
            stim_amp_scale=0.1,
        )

        print(f"Running {patch_type} simulation {i}/{n_stim} (stim_seed={stim_seed})...")
        out = simulate(p)

        print("  Calculating phie...")
        elecposX, elecposY = make_electrodes_from_domain(
            X=ncells+2, Y=ncells+2, numelec_x=10, numelec_y=10
        )
        phie, elecpos = calc_phie_jax(
            h=p.h,
            D_matrix=D_matrix,
            elecposX=np.array(elecposX),
            elecposY=np.array(elecposY),
            Vsav=out["Vsav"],
            vsav_layout="TXY",
        )

        result = {
            "Vsav": out["Vsav"],
            "Wsav": out["Wsav"],
            "t_sav": out["t_sav"],
            "D_matrix": D_matrix,
            "fiblocs": fiblocs,
            "stim_center": stim_center,
            "phie": phie,
            "elecpos": elecpos * p.h,   # same length unit as h, as in run_planar_simulation
            "params": p,
            "patch_type": patch_type,
            "patch_seed": patch_seed,
            "stim_seed": stim_seed,
        }

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(
                save_dir, f"sim_{patch_type}_patch{patch_seed}_stim{stim_seed}.npz"
            )
            fiblocs_save = fiblocs
            if isinstance(fiblocs_save, list):
                fiblocs_save = np.array(fiblocs_save, dtype=object)
            np.savez_compressed(
                filename,
                t_sav=result["t_sav"],
                D_matrix=result["D_matrix"],
                fiblocs=fiblocs_save,
                phie=result["phie"],
                elecpos=result["elecpos"],
                stim_center=np.array(result["stim_center"]),
                patch_type=patch_type,
                patch_seed=patch_seed,
                stim_seed=stim_seed,
                ncells=ncells,
                D0=D0,
                Dfac=Dfac,
                npatches=npatches,
            )
            print(f"  Saved to {filename}")

        if return_data:
            results.append(result)

    if not return_data:
        return [
            {
                "patch_type": patch_type,
                "patch_seed": patch_seed,
                "stim_seed": s,
                "ncells": ncells,
                "D0": D0,
                "Dfac": Dfac,
                "npatches": npatches,
                "save_dir": save_dir,
            }
            for s in stim_seeds
        ]

    return results


#%%
# =============================================================================
# 4. BATCH DATASETS
# =============================================================================
# Many simulations with different patches and pacing locations, saved to disk.
def generate_batch_simulations(n_simulations: int = 10,
                               ncells: int = 100,
                               npatches: int = 5,
                               D0: float = 0.1,
                               Dfac_range: Tuple[float, float] = (0.2, 0.4),
                               a_factor_range: Tuple[float, float] = (1.0, 1.0),
                               b_factor_range: Tuple[float, float] = (1.0, 1.0),
                               save_dir: str = "AP_simulations_more_var",
                               start_seed: int = 0,
                               keep_results: bool = True,
                               save_one_video: bool = True) -> list:
    """
    Generate multiple simulations with varying fibrotic patterns and pacing locations.

    Parameters:
    -----------
    n_simulations : int
        Total number of simulations to generate (all with irregular patches)
    ncells : int
        Grid size (interior)
    npatches : int
        Number of fibrotic patches per simulation
    D0 : float
        Baseline diffusion coefficient
    save_dir : str
        Directory to save simulation results
    start_seed : int
        Starting seed for random number generation

    Returns:
    --------
    list of dicts, each containing simulation results
    """
    results = []

    print(f"\n{'='*70}")
    print(f"GENERATING {n_simulations} SIMULATIONS")
    print(f"  - All with irregular patches")
    print(f"  - Grid size: {ncells}x{ncells}")
    print(f"  - Patches per simulation: {npatches}")
    print(f"  - Fibrosis variability: Dfac~U{Dfac_range}, a_factor~U{a_factor_range}, b_factor~U{b_factor_range}")
    print(f"  - Results will be saved to: {save_dir}/")
    print("  - 3 pacing locations per simulation (ncyc=3 in run_single_simulation)")
    print(f"{'='*70}\n")

    # Generate irregular patch simulations
    for i in range(n_simulations):
        seed = start_seed + i

        print(f"\n[{i+1}/{n_simulations}] Irregular patch simulation (seed={seed})")
        need_result = keep_results or (save_one_video and i == 0)
        result = run_single_simulation(
            ncells=ncells,
            patch_type='irregular',
            npatches=npatches,
            seed=seed,
            D0=D0,
            Dfac_range=Dfac_range,
            a_factor_range=a_factor_range,
            b_factor_range=b_factor_range,
            save_dir=save_dir,
            return_data=need_result,
        )

        if save_one_video and i == 0 and need_result:
            video_filename = os.path.join(save_dir, f"preview_irregular_{seed}.mp4")
            save_VW_video(
                result["Vsav"],
                result["Wsav"],
                result["D_matrix"] < result["params"].D_scalar,
                filename=video_filename,
                fps=20,
                vmin=0.0,
                vmax=1.0,
                dt=result["params"].dt,
                gathert=result["params"].gathert,
            )

        if keep_results:
            results.append(result)

        del result

        # Proactively release memory between simulations
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"BATCH GENERATION COMPLETE")
    if keep_results:
        print(f"Total simulations: {len(results)}")
    else:
        total = n_simulations
        print(f"Total simulations: {total}")
    print(f"Results saved to: {save_dir}/")
    print(f"{'='*70}\n")

    return results


def generate_heightened_excitability_batch(
    n_simulations: int = 20,
    variant: str = 'higher_k',
    ncells: int = 100,
    D0: float = 0.1,
    save_dir: str = "AP_simulations_heightened_excitability",
    start_seed: int = 0,
    neg_a_value: float = -0.025,
    k_increase_range: Tuple[float, float] = (1.2, 2.0),
    keep_results: bool = False,
    save_one_video: bool = True,
) -> list:
    """
    Generate a batch of single-patch heightened-excitability simulations.

    Parameters:
    -----------
    n_simulations : int
        Number of simulations to generate.
    variant : str
        Which parameter(s) to perturb inside the patch:
          'neg_a'    -> mode='heightened_excitability_neg_a'
          'higher_k' -> mode='heightened_excitability_higher_k'
          'both'     -> mode='heightened_excitability_both'
    ncells, D0, start_seed : as in generate_batch_simulations.
    neg_a_value : fixed negative a inside the patch (e.g. -0.025), passed
        through to run_single_simulation.
    k_increase_range : sampling range for k, passed through to
        run_single_simulation.
    keep_results : if False, only the first sim's arrays are retained (for
        the preview video); the rest are dropped after the npz is written.
    save_one_video : write a preview .mp4 for the first simulation.
    """
    variant_to_mode = {
        'neg_a':    'heightened_excitability_neg_a',
        'higher_k': 'heightened_excitability_higher_k',
        'both':     'heightened_excitability_both',
    }
    if variant not in variant_to_mode:
        raise ValueError(
            f"variant must be one of {list(variant_to_mode)}, got {variant!r}"
        )
    mode = variant_to_mode[variant]

    os.makedirs(save_dir, exist_ok=True)
    results = []

    print(f"\n{'='*70}")
    print(f"GENERATING {n_simulations} HEIGHTENED-EXCITABILITY SIMULATIONS")
    print(f"  - variant: {variant}  (mode={mode})")
    print(f"  - Grid size: {ncells}x{ncells}, 1 irregular patch per sim, D unchanged")
    if variant in ('neg_a', 'both'):
        print(f"  - a (patch) = {neg_a_value} (fixed negative)")
    if variant in ('higher_k', 'both'):
        print(f"  - k_factor ~ U[{k_increase_range[0]}, {k_increase_range[1]}]")
    print(f"  - Results -> {save_dir}/")
    print(f"{'='*70}\n")

    for i in range(n_simulations):
        seed = start_seed + i
        print(f"\n[{i+1}/{n_simulations}] seed={seed}")

        need_result = keep_results or (save_one_video and i == 0)
        result = run_single_simulation(
            ncells=ncells,
            patch_type='irregular',
            npatches=1,
            seed=seed,
            mode=mode,
            D0=D0,
            neg_a_value=neg_a_value,
            k_increase_range=k_increase_range,
            save_dir=save_dir,
            return_data=need_result,
        )

        if save_one_video and i == 0 and need_result:
            # Overlay the excitability patch (D == D0 here, so derive mask from fiblocs).
            X = ncells + 2
            patch_overlay = np.zeros((X, X), dtype=bool)
            for patch_coords in result["fiblocs"]:
                if len(patch_coords) > 0:
                    patch_overlay[patch_coords[:, 0], patch_coords[:, 1]] = True

            video_filename = os.path.join(save_dir, f"preview_{variant}_{seed}.mp4")
            save_VW_video(
                result["Vsav"],
                result["Wsav"],
                patch_overlay,
                filename=video_filename,
                fps=20,
                vmin=0.0,
                vmax=1.0,
                dt=result["params"].dt,
                gathert=result["params"].gathert,
            )

        if keep_results:
            results.append(result)

        del result
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    print(f"\n{'='*70}")
    print(f"BATCH GENERATION COMPLETE  ({variant})")
    print(f"Total simulations: {n_simulations}")
    print(f"Results saved to: {save_dir}/")
    print(f"{'='*70}\n")

    return results


#%%
# =============================================================================
# 5. HOMOGENEOUS REFERENCE DATASET
# =============================================================================
# Simulations without patches, each paced once from a random site, saved
# together with a dense matrix A such that phie = A @ V_flat.
def _trapz_weights(x: np.ndarray) -> np.ndarray:
    dx = np.diff(x)
    if dx.size == 0:
        return np.zeros_like(x)
    w = np.empty_like(x)
    w[0] = dx[0] * 0.5
    w[-1] = dx[-1] * 0.5
    if x.size > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def build_phie_matrix_dense(
    D_matrix: np.ndarray,
    h: float,
    elecposX: np.ndarray,
    elecposY: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build explicit dense matrix A such that phie = A @ V_flat.
    Matches calc_phie_jax boundary handling and trapz weights.

    V_flat is row-major flattening of (ncells, ncells).
    Returns A (E, N) and elecpos (E, 2).
    """
    ncells = D_matrix.shape[0] - 2
    if ncells <= 0:
        raise ValueError("D_matrix must include a 1-cell ghost border.")

    D_core = D_matrix[1:-1, 1:-1].astype(np.float32, copy=False)

    Dx_full, Dy_full = _grad_central(jnp.asarray(D_matrix, dtype=jnp.float32), h)
    Dx_core = np.array(Dx_full)[1:-1, 1:-1].astype(np.float32, copy=False)
    Dy_core = np.array(Dy_full)[1:-1, 1:-1].astype(np.float32, copy=False)

    # Match calc_phie_jax electrode nudging
    ex = np.array(elecposX, dtype=np.float32, copy=True)
    ey = np.array(elecposY, dtype=np.float32, copy=True)
    ex[np.isclose(ex % 1.0, 0.0)] += 1e-4
    ey[np.isclose(ey % 1.0, 0.0)] += 1e-4
    elecpos = np.stack([ex, ey], axis=1)

    x_coords = np.arange(0, ncells + 2, dtype=np.float32)
    y_coords = np.arange(0, ncells + 2, dtype=np.float32)
    x_phys = x_coords * h
    y_phys = y_coords * h
    x_core = x_coords[1:-1]
    y_core = y_coords[1:-1]

    w_x = _trapz_weights(x_phys[1:-1])
    w_y = _trapz_weights(y_phys[1:-1])
    w_xy = (w_x[:, None] * w_y[None, :]).astype(np.float32)

    # 1D gradient matrix matching jnp.gradient with uniform spacing
    G = np.zeros((ncells, ncells), dtype=np.float32)
    for i in range(ncells):
        if i == 0:
            G[i, 0] = -1.0 / h
            G[i, 1] = 1.0 / h
        elif i == ncells - 1:
            G[i, ncells - 2] = -1.0 / h
            G[i, ncells - 1] = 1.0 / h
        else:
            G[i, i - 1] = -0.5 / h
            G[i, i + 1] = 0.5 / h

    L = G @ G

    # Precompute nonzero stencils per row
    G_rows = []
    L_rows = []
    for i in range(ncells):
        g_idx = np.nonzero(G[i])[0]
        l_idx = np.nonzero(L[i])[0]
        G_rows.append((g_idx, G[i, g_idx]))
        L_rows.append((l_idx, L[i, l_idx]))

    # Distance weights per electrode
    Xg, Yg = np.meshgrid(x_core, y_core, indexing="ij")
    N = ncells * ncells
    E = ex.shape[0]
    W = np.zeros((E, N), dtype=np.float32)
    for e in range(E):
        distance = np.sqrt((Xg - ex[e]) ** 2 + (Yg - ey[e]) ** 2) * h
        weights = w_xy / distance
        W[e, :] = weights.ravel(order="C")

    # Build A directly: A = -W @ K, without forming K explicitly.
    A = np.zeros((E, N), dtype=np.float32)
    for i in range(ncells):
        lxi_idx, lxi_val = L_rows[i]
        gxi_idx, gxi_val = G_rows[i]
        for j in range(ncells):
            p = i * ncells + j
            w_vec = W[:, p]

            d_ij = D_core[i, j]
            dx_ij = Dx_core[i, j]
            dy_ij = Dy_core[i, j]

            # X-direction contributions (vary row index, same col)
            for k, v in zip(lxi_idx, lxi_val):
                q = k * ncells + j
                A[:, q] -= w_vec * (d_ij * v)
            for k, v in zip(gxi_idx, gxi_val):
                q = k * ncells + j
                A[:, q] -= w_vec * (dx_ij * v)

            # Y-direction contributions (vary col index, same row)
            lyj_idx, lyj_val = L_rows[j]
            gyj_idx, gyj_val = G_rows[j]
            for k, v in zip(lyj_idx, lyj_val):
                q = i * ncells + k
                A[:, q] -= w_vec * (d_ij * v)
            for k, v in zip(gyj_idx, gyj_val):
                q = i * ncells + k
                A[:, q] -= w_vec * (dy_ij * v)

    return A, elecpos


def generate_no_fibrosis_dataset(n_simulations: int = 5,
                                 ncells: int = 100,
                                 D0: float = 0.1,
                                 save_dir: str = "Eikonal-PINNs data",
                                 seed_start: int = 0,
                                 stim_size: int = 10,
                                 numelec_x: int = 10,
                                 numelec_y: int = 10,
                                 save_videos: bool = True) -> list:
    """
    Generate simulations with no fibrosis, one pacing cycle (ncyc=1),
    and random stimulation locations. Saves Vsav and phie per simulation.

    Outputs (.npz) include: Vsav, t_sav, phie, elecpos, stim_center, ncells, D0, seed.
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # Uniform diffusion (no fibrosis) and shared operator A
    X = ncells + 2
    D_matrix = np.full((X, X), D0, dtype=np.float32)
    elecposX, elecposY = make_electrodes_from_domain(
        X=ncells + 2, Y=ncells + 2, numelec_x=numelec_x, numelec_y=numelec_y
    )
    A, elecpos = build_phie_matrix_dense(
        D_matrix=D_matrix,
        h=0.1,
        elecposX=np.array(elecposX),
        elecposY=np.array(elecposY),
    )

    # Vpos shares the frame of elecpos, that of the grid including its ghost
    # border: interior cell [i, j] is cell [i+1, j+1] of the full grid.
    rows, cols = np.meshgrid(
        np.arange(1, ncells + 1, dtype=np.int32),
        np.arange(1, ncells + 1, dtype=np.int32),
        indexing="ij",
    )
    Vpos = np.stack([rows.ravel(), cols.ravel()], axis=1)
    Vpos_mm = Vpos.astype(np.float32) * 0.1
    elecpos_mm = elecpos.astype(np.float32) * 0.1

    A_path = os.path.join(save_dir, "phie_operator_A.npz")
    np.savez_compressed(
        A_path,
        A=A,
        elecpos=elecpos_mm,
        Vpos=Vpos_mm,
        ncells=ncells,
        D0=D0,
        h=0.1,
    )
    print(f"Saved operator A to {A_path}")

    for i in range(n_simulations):
        seed = seed_start + i
        rng = np.random.default_rng(seed)

        # One random stimulus, no forbidden regions
        stim_mask, stim_center = generate_random_stim_mask(
            ncells=ncells,
            stim_size=stim_size,
            seed=int(rng.integers(0, 2**31 - 1))
        )

        p = Params(
            dt=0.01,
            tend=50.0,
            BCL=50.0,
            ncyc=1,
            stimdur=2.0,
            gathert=10,
            ncells=ncells,
            h=0.1,
            k=8.0,
            mu1=0.2,
            mu2=0.3,
            epsi=0.002,
            D_scalar=D0,
            a0=0.01,
            b0=0.15,
            D_matrix=D_matrix,
            stim_mask=stim_mask,
            stim_amp_scale=0.1,
        )

        print(f"Running no-fibrosis simulation {i + 1}/{n_simulations} (seed={seed})...")
        out = simulate(p)

        print("  Calculating phie...")
        phie, elecpos = calc_phie_jax(
            h=p.h,
            D_matrix=D_matrix,
            elecposX=np.array(elecposX),
            elecposY=np.array(elecposY),
            Vsav=out["Vsav"],
            vsav_layout="TXY",
        )

        Vsav_full = out["Vsav"]
        Vsav_flat = Vsav_full.reshape(Vsav_full.shape[0], -1).T
        t_sav_ms = out["t_sav"] * MS_PER_TU

        if i == 0:
            phie_from_A = A @ Vsav_flat
            max_abs_err = float(np.max(np.abs(phie_from_A - phie)))
            mean_abs_err = float(np.mean(np.abs(phie_from_A - phie)))
            print(f"  A consistency check: max |A@V - phie| = {max_abs_err:.3e}, mean = {mean_abs_err:.3e}")

        filename = os.path.join(save_dir, f"iso_2D_{seed}.npz")
        np.savez_compressed(
            filename,
            Vsav=Vsav_flat,
            Vpos=Vpos_mm,
            t_sav=t_sav_ms,
            phie=phie,
            elecpos=elecpos_mm,
            stim_center=np.array(stim_center, dtype=np.int32),
            ncells=ncells,
            D0=D0,
            seed=seed,
        )

        if save_videos:
            video_filename = os.path.join(save_dir, f"iso_2D_{seed}.mp4")
            save_VW_video(
                out["Vsav"],
                out["Wsav"],
                np.zeros((ncells, ncells), dtype=bool),
                filename=video_filename,
                fps=10,
                vmin=0.0,
                vmax=1.0,
                dt=p.dt,
                gathert=p.gathert,
            )

        results.append(
            {
                "Vsav": Vsav_flat,
                "Vpos": Vpos_mm,
                "t_sav": t_sav_ms,
                "phie": phie,
                "elecpos": elecpos_mm,
                "stim_center": stim_center,
                "params": p,
                "seed": seed,
            }
        )

        del out
        gc.collect()
        try:
            jax.clear_caches()
        except Exception:
            pass

    return results
