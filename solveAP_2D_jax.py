"""
2D Aliev-Panfilov cardiac electrophysiology solver, written in JAX.

Solves the Aliev-Panfilov model on a square sheet of tissue (ncells x ncells)
with explicit RK4 time stepping and no-flux boundaries. It can also compute the
extracellular potential (phie) recorded by a grid of electrodes and save a
video of the simulation.

Installation
------------
    conda env create -f environment.yml
    conda activate aliev-panfilov

or, in an existing Python environment (3.10 or newer):

    pip install -r requirements.txt

See the notes in requirements.txt for GPU support and Apple-silicon Macs.

Quick start: a planar wave on homogeneous tissue
------------------------------------------------
    from solveAP_2D_jax import Params, planar_stim_mask, run_planar_simulation

    params = Params(stim_mask=planar_stim_mask(ncells=100, edge="top"))
    out = run_planar_simulation(params)   # saves planar_output/planar_wave.npz and .mp4
    V = out["Vsav"]                       # (frames, ncells, ncells)

Running this file directly (python solveAP_2D_jax.py) does the same; see the
example at the bottom of the file.

Contents
--------
    1. Parameters            Params
    2. Solver                simulate
    3. Planar stimulus       planar_stim_mask
    4. Electrodes and phie   make_electrodes_from_domain, calc_phie_jax
    5. Video                 save_VW_video
    6. Run and save          run_planar_simulation, run_planar_batch
    7. Example

Conventions
-----------
- Arrays are indexed (row, col). The saved states Vsav and Wsav have shape
  (frames, ncells, ncells).
- Internally the grid carries a 1-cell "ghost" border (size ncells + 2), used
  to enforce the no-flux boundary. Outputs have the border removed; spatially
  varying inputs (the advanced fields of Params) include it.
- Time is in model time units (written AU or TU below); one unit is 12.9 ms
  (MS_PER_TU). Lengths are in the same unit as the grid spacing h.
- The diffusion term is D * Laplacian(V) + Dx * Vx + Dy * Vy, which reduces to
  D * Laplacian(V) when D is uniform.

How JAX is used
---------------
- Pure functions, JIT compilation and lax.scan for the time loop.
- Stencils (central differences, 5-point Laplacian) built with jnp.roll.

Tissue heterogeneity (fibrotic or hyper-excitable patches), other pacing
protocols and dataset generators are in solveAP_2D_heterogeneity.py.
"""
#%%
# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations

import gc
import os
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax
from PIL import Image, ImageDraw, ImageFont

Array = jnp.ndarray

MS_PER_TU = 12.9  # milliseconds per model time unit

# GPU: the code runs unchanged on CPU or GPU. To use a GPU, install JAX with
# CUDA support, e.g. pip install --upgrade "jax[cuda12]"; jax.devices() shows
# which device JAX is using. If memory runs out, reduce ncells or the number of
# electrodes (a 100x100 grid with 100 electrodes and 1500 saved frames needs a
# few GB).


#%%
# =============================================================================
# 1. PARAMETERS
# =============================================================================
@dataclass(frozen=True)
class Params:
    """
    Simulation settings. The defaults give one paced beat on homogeneous
    tissue; only stim_mask needs to be supplied (see planar_stim_mask).
    """
    # Grid
    ncells: int = 100                       # interior size (ncells x ncells)
    h: float = 0.1                          # spatial step

    # Time
    dt: float = 0.01                        # AU
    tend: float = 50.0                      # AU
    gathert: int = 10                       # save every this many steps (integer iterations)

    # Pacing
    BCL: float = 50.0                       # AU, basic cycle length (time between paced beats)
    ncyc: int = 1                           # number of paced beats
    stimdur: float = 2.0                    # AU, duration of each pacing stimulus
    stim_mask: Optional[np.ndarray] = None  # interior (ncells,ncells) boolean/0-1
    stim_amp_scale: float = 0.1             # stimulus current: Ia = stim_amp_scale * stim_mask

    # Model
    k: float = 8.0
    a0: float = 0.01                        # baseline a (scalar)
    b0: float = 0.15                        # baseline b (scalar)
    mu1: float = 0.2
    mu2: float = 0.3
    epsi: float = 0.002
    D_scalar: float = 0.1                   # baseline diffusion coefficient D0

    # -------------------------------------------------------------------------
    # Advanced: not needed for planar waves on homogeneous tissue.
    # See solveAP_2D_heterogeneity.py for how these are used.
    # -------------------------------------------------------------------------
    # Spatial heterogeneity. X = ncells + 2 (the grid including its ghost border).
    a_field: Optional[np.ndarray] = None    # (X,X) including ghost; if None -> a0 everywhere
    b_field: Optional[np.ndarray] = None    # (X,X) including ghost; if None -> b0 everywhere
    k_field: Optional[np.ndarray] = None    # (X,X) including ghost; if None -> k everywhere
    D_matrix: Optional[np.ndarray] = None   # (X,X) including ghost; if None -> D_scalar everywhere

    # A different pacing site for each beat
    stim_masks: Optional[np.ndarray] = None  # interior (ncyc,ncells,ncells); if set, one mask per cycle

    # Spiral wave generation (cross-field stimulation)
    cross_stim_mask: Optional[np.ndarray] = None  # interior (ncells,ncells) boolean/0-1 for cross-field stimulus
    cross_stim_time: float = 42.0                 # AU, time at which cross-field stimulus is applied
    cross_stim_duration: float = 1.0              # AU, duration of cross-field stimulus


#%%
# =============================================================================
# 2. SOLVER
# =============================================================================
def _pad_with_ghost(interior: np.ndarray, ghost_value: float = 0.0) -> np.ndarray:
    """Pad (ncells,ncells) -> (ncells+2,ncells+2) with ghost border."""
    return np.pad(interior, pad_width=1, mode="constant", constant_values=ghost_value)


def _ensure_fields(params: Params) -> Dict[str, np.ndarray]:
    """Build full-grid (with ghost) fields on host as numpy arrays."""
    X = params.ncells + 2

    if params.D_matrix is None:
        D = np.full((X, X), params.D_scalar, dtype=np.float32)
    else:
        D = np.asarray(params.D_matrix, dtype=np.float32)
        assert D.shape == (X, X)

    if params.a_field is None:
        a = np.full((X, X), params.a0, dtype=np.float32)
    else:
        a = np.asarray(params.a_field, dtype=np.float32)
        assert a.shape == (X, X)

    if params.b_field is None:
        b = np.full((X, X), params.b0, dtype=np.float32)
    else:
        b = np.asarray(params.b_field, dtype=np.float32)
        assert b.shape == (X, X)

    if params.k_field is None:
        k = np.full((X, X), params.k, dtype=np.float32)
    else:
        k = np.asarray(params.k_field, dtype=np.float32)
        assert k.shape == (X, X)

    if params.stim_masks is not None:
        stim_masks_interior = np.asarray(params.stim_masks, dtype=np.float32)
        assert stim_masks_interior.shape == (params.ncyc, params.ncells, params.ncells)
        stim_masks_full = np.pad(
            stim_masks_interior,
            pad_width=((0, 0), (1, 1), (1, 1)),
            mode="constant",
            constant_values=0.0,
        ).astype(np.float32)
        stim_full = stim_masks_full[0]
    else:
        if params.stim_mask is None:
            stim_interior = np.zeros((params.ncells, params.ncells), dtype=np.float32)
        else:
            stim_interior = np.asarray(params.stim_mask, dtype=np.float32)
            assert stim_interior.shape == (params.ncells, params.ncells)

        stim_full = _pad_with_ghost(stim_interior, ghost_value=0.0).astype(np.float32)
        stim_masks_full = np.repeat(stim_full[None, ...], params.ncyc, axis=0).astype(np.float32)

    # Cross-field stimulation for spiral wave
    if params.cross_stim_mask is None:
        cross_stim_interior = np.zeros((params.ncells, params.ncells), dtype=np.float32)
    else:
        cross_stim_interior = np.asarray(params.cross_stim_mask, dtype=np.float32)
        assert cross_stim_interior.shape == (params.ncells, params.ncells)

    cross_stim_full = _pad_with_ghost(cross_stim_interior, ghost_value=0.0).astype(np.float32)

    return {
        "D": D,
        "a": a,
        "b": b,
        "k": k,
        "stim_full": stim_full,
        "stim_masks_full": stim_masks_full,
        "cross_stim_full": cross_stim_full,
    }


def _apply_neumann_bc(V: Array) -> Array:
    """No-flux BC: copy adjacent interior into ghost border."""
    # top/bottom rows
    V = V.at[0, :].set(V[1, :])   # i.e. V[0,: ] = V[1,:]
    V = V.at[-1, :].set(V[-2, :])
    # left/right cols
    V = V.at[:, 0].set(V[:, 1])
    V = V.at[:, -1].set(V[:, -2])
    return V


def _grad_central(F: Array, h: float) -> Tuple[Array, Array]:
    """Central differences using roll; assumes ghost cells valid for BC."""
    Fx = (jnp.roll(F, -1, axis=0) - jnp.roll(F, 1, axis=0)) / (2.0 * h)
    Fy = (jnp.roll(F, -1, axis=1) - jnp.roll(F, 1, axis=1)) / (2.0 * h)
    return Fx, Fy


def _laplacian_5pt(F: Array, h: float) -> Array:
    """5-point Laplacian on full grid."""
    return (
        jnp.roll(F, -1, axis=0)
        + jnp.roll(F, 1, axis=0)
        + jnp.roll(F, -1, axis=1)
        + jnp.roll(F, 1, axis=1)
        - 4.0 * F
    ) / (h * h)


def _rhs_alpan(V: Array, W: Array, Istim: Array, a: Array, b: Array, D: Array, Dx: Array, Dy: Array,
               h: float, k: float, mu1: float, mu2: float, epsi: float) -> Tuple[Array, Array]:
    """Compute dV/dt and dW/dt."""
    # Ensure BC before stencils
    V = _apply_neumann_bc(V)

    Vx, Vy = _grad_central(V, h)
    lapV = _laplacian_5pt(V, h)

    diffusion = D * lapV + Dx * Vx + Dy * Vy

    dWdt = (epsi + mu1 * W / (mu2 + V)) * (-W - k * V * (V - b - 1.0))
    dVdt = (-k * V * (V - a) * (V - 1.0) - W * V) + diffusion + Istim

    return dVdt, dWdt


def _rk4_step(V: Array, W: Array, Istim: Array, a: Array, b: Array, D: Array, Dx: Array, Dy: Array,
              dt: float, h: float, k: float, mu1: float, mu2: float, epsi: float) -> Tuple[Array, Array]:
    """One explicit RK4 step."""
    k1V, k1W = _rhs_alpan(V, W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)
    k2V, k2W = _rhs_alpan(V + 0.5 * dt * k1V, W + 0.5 * dt * k1W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)
    k3V, k3W = _rhs_alpan(V + 0.5 * dt * k2V, W + 0.5 * dt * k2W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)
    k4V, k4W = _rhs_alpan(V + dt * k3V, W + dt * k3W, Istim, a, b, D, Dx, Dy, h, k, mu1, mu2, epsi)

    Vn = V + (dt / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V)
    Wn = W + (dt / 6.0) * (k1W + 2.0 * k2W + 2.0 * k3W + k4W)

    # Enforce BC on updated V
    Vn = _apply_neumann_bc(Vn)
    return Vn, Wn


def simulate(params: Params,
             V0: Optional[np.ndarray] = None,
             W0: Optional[np.ndarray] = None,
             dtype=jnp.float32) -> Dict[str, np.ndarray]:
    """
    Run the simulation fully in JAX (JIT+scan), returning saved states.

    Returns dict:
      Vsav: (nsaves, ncells, ncells) interior only (ghost removed)
      Wsav: (nsaves, ncells, ncells)
      t_sav: (nsaves,) times corresponding to saved frames
    """
    fields = _ensure_fields(params)
    X = params.ncells + 2

    # Host -> device constants
    a = jnp.asarray(fields["a"], dtype=dtype)
    b = jnp.asarray(fields["b"], dtype=dtype)
    k = jnp.asarray(fields["k"], dtype=dtype)
    D = jnp.asarray(fields["D"], dtype=dtype)
    stim_masks_full = jnp.asarray(fields["stim_masks_full"], dtype=dtype)
    cross_stim_full = jnp.asarray(fields["cross_stim_full"], dtype=dtype)

    # Precompute grad(D) once (D is static)
    Dx, Dy = _grad_central(D, params.h)

    # Initial conditions (with ghost)
    if V0 is None:
        V_init = np.zeros((X, X), dtype=np.float32)
    else:
        V_init = np.asarray(V0, dtype=np.float32)
        assert V_init.shape == (X, X)

    if W0 is None:
        W_init = np.full((X, X), 0.01, dtype=np.float32)
    else:
        W_init = np.asarray(W0, dtype=np.float32)
        assert W_init.shape == (X, X)

    V_init = jnp.asarray(V_init, dtype=dtype)
    W_init = jnp.asarray(W_init, dtype=dtype)

    # Timesteps
    nsteps = int(np.floor(params.tend / params.dt))
    # Save every gathert steps
    nsaves = nsteps // params.gathert

    # Preallocate save buffers on device
    Vsav0 = jnp.zeros((nsaves, params.ncells, params.ncells), dtype=dtype)
    Wsav0 = jnp.zeros((nsaves, params.ncells, params.ncells), dtype=dtype)

    dt = params.dt
    h = params.h

    mu1 = params.mu1
    mu2 = params.mu2
    epsi = params.epsi

    BCL = params.BCL
    ncyc = params.ncyc
    stimdur = params.stimdur
    cross_stim_time = params.cross_stim_time
    cross_stim_duration = params.cross_stim_duration

    Ia_masks = params.stim_amp_scale * stim_masks_full
    Ia_cross = params.stim_amp_scale * cross_stim_full

    def step_fn(carry, n):
        V, W, kk, save_idx, Vsav, Wsav = carry
        t = (n + 1) * dt

        # Determine stimulation window for current cycle kk (integer)
        kk_f = kk.astype(dtype)  # for t comparisons with BCL*kk
        stim_on = (kk < ncyc) & (t >= BCL * kk_f) & (t < (BCL * kk_f + stimdur))

        # Cross-field stimulation for spiral wave (applied at specific time)
        cross_stim_on = (t >= cross_stim_time) & (t < (cross_stim_time + cross_stim_duration))

        # Cycle-specific pacing mask: each cycle can use a different location.
        cycle_idx = jnp.clip(kk, 0, ncyc - 1)
        Ia_cycle = Ia_masks[cycle_idx]

        # Combine both stimulations
        Istim_primary = jnp.where(stim_on, Ia_cycle, jnp.zeros_like(Ia_cycle))
        Istim_cross = jnp.where(cross_stim_on, Ia_cross, jnp.zeros_like(Ia_cross))
        Istim = Istim_primary + Istim_cross

        # Update kk when stimulation window has ended
        kk_inc = (kk < ncyc) & (t >= (BCL * kk_f + stimdur))
        kk_next = kk + kk_inc.astype(kk.dtype)

        # RK4 step
        Vn, Wn = _rk4_step(V, W, Istim, a, b, D, Dx, Dy, dt, h, k, mu1, mu2, epsi)

        # Save every gathert steps
        do_save = ((n + 1) % params.gathert) == 0

        def save_branch(args):
            Vn, Wn, save_idx, Vsav, Wsav = args
            V_interior = Vn[1:-1, 1:-1]
            W_interior = Wn[1:-1, 1:-1]
            Vsav = Vsav.at[save_idx].set(V_interior)
            Wsav = Wsav.at[save_idx].set(W_interior)
            return (save_idx + 1, Vsav, Wsav)

        def nosave_branch(args):
            _, _, save_idx, Vsav, Wsav = args
            return (save_idx, Vsav, Wsav)

        save_idx2, Vsav2, Wsav2 = lax.cond(
            do_save,
            save_branch,
            nosave_branch,
            (Vn, Wn, save_idx, Vsav, Wsav)
        ) # lax.cond: conditionally apply true_fun (save_branch) or false_fun (nosave_branch).

        # Optional: early abort on NaN/Inf (JAX-friendly way is to track a flag;
        # true early-break isn't supported inside scan)
        return (Vn, Wn, kk_next, save_idx2, Vsav2, Wsav2), None

    # Carry: (V, W, kk, save_idx, Vsav, Wsav)
    carry0 = (V_init, W_init, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32), Vsav0, Wsav0)

    # JIT the whole scan
    @jax.jit
    def run():
        carryT, _ = lax.scan(step_fn, carry0, jnp.arange(nsteps))
        _, _, _, _, Vsav, Wsav = carryT
        return Vsav, Wsav

    Vsav_dev, Wsav_dev = run()

    # Bring back to host numpy
    Vsav = np.array(Vsav_dev)
    Wsav = np.array(Wsav_dev)

    # Saved times (correspond to ind = gathert, 2*gathert, ...)
    t_sav = (np.arange(1, nsaves + 1) * params.gathert) * params.dt

    return {"Vsav": Vsav, "Wsav": Wsav, "t_sav": t_sav}


#%%
# =============================================================================
# 3. PLANAR STIMULUS
# =============================================================================
def planar_stim_mask(ncells: int, width: int = 5, edge: str = "top") -> np.ndarray:
    """
    Stimulus mask for a planar wave: a strip `width` cells wide along one edge
    of the tissue. The wave then travels away from that edge.

    Parameters:
    -----------
    ncells : int
        Interior grid size (ncells x ncells)
    width : int
        Width of the stimulated strip, in cells
    edge : str
        'top' (first rows, the top of the video), 'bottom' (last rows),
        'left' (first columns) or 'right' (last columns)

    Returns:
    --------
    stim_mask : (ncells, ncells) ndarray
        1 where the tissue is stimulated, 0 elsewhere
    """
    stim_mask = np.zeros((ncells, ncells), dtype=np.float32)
    if edge == "top":
        stim_mask[:width, :] = 1.0
    elif edge == "bottom":
        stim_mask[-width:, :] = 1.0
    elif edge == "left":
        stim_mask[:, :width] = 1.0
    elif edge == "right":
        stim_mask[:, -width:] = 1.0
    else:
        raise ValueError(f"edge must be 'top', 'bottom', 'left' or 'right', got {edge!r}")
    return stim_mask


#%%
# =============================================================================
# 4. ELECTRODES AND PHIE
# =============================================================================
def make_electrodes_from_domain(X: int, Y: int, numelec_x: int, numelec_y: int):
    """
    Positions of a regular numelec_x x numelec_y grid of electrodes, centred on
    an X x Y domain. Pass the grid size including its ghost border
    (X = Y = ncells + 2).

    Returns two flat arrays (elecposX, elecposY) of length numelec_x * numelec_y,
    holding the row and column coordinate of each electrode in grid-index units.
    """
    centerX = X / 2.0
    centerY = Y / 2.0
    spacing_x = X / (numelec_x + 1)
    spacing_y = Y / (numelec_y + 1)

    gridx = np.linspace(centerX - (spacing_x * (numelec_x - 1)) / 2.0,
                        centerX + (spacing_x * (numelec_x - 1)) / 2.0,
                        numelec_x)
    gridy = np.linspace(centerY - (spacing_y * (numelec_y - 1)) / 2.0,
                        centerY + (spacing_y * (numelec_y - 1)) / 2.0,
                        numelec_y)

    # indexing="ij": first dimension follows gridx (rows), second follows gridy (cols)
    elecposX, elecposY = np.meshgrid(gridx, gridy, indexing="ij")
    return elecposX.reshape(-1), elecposY.reshape(-1)


def _trapz_jax(y: jnp.ndarray, x: jnp.ndarray, axis: int) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    y_m = jnp.moveaxis(y, axis, -1)
    return jnp.sum(0.5 * (y_m[..., 1:] + y_m[..., :-1]) * dx, axis=-1)


def calc_phie_jax(
    Vsav: np.ndarray,              # (T, ncells, ncells) interior only
    h: float,
    D_matrix: np.ndarray,          # (X, Y) with ghost, X=ncells+2
    elecposX: np.ndarray,          # (E,) electrode positions
    elecposY: np.ndarray,          # (E,)
    vsav_layout: str = "TXY",
    dtype=jnp.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracellular potential (phie) at every electrode, for every saved frame.

    For each electrode, phie is minus the integral over the tissue of the
    diffusion term of the model divided by the distance to the electrode.

    Returns:
    --------
    phie : (E, T) ndarray
        One row per electrode, one column per saved frame
    elecpos : (E, 2) ndarray
        Electrode (row, col) positions in grid-index units of the grid including
        its ghost border. Coordinates that fall exactly on a grid line are nudged
        by 1e-4 to avoid a zero distance.
    """
    if vsav_layout.upper() == "TXY":
        Vt = Vsav  # (T, X, Y)
    elif vsav_layout.upper() == "XYT":
        Vt = np.moveaxis(Vsav, 2, 0)  # Convert (X, Y, T) -> (T, X, Y)
    else:
        raise ValueError("vsav_layout must be 'TXY' or 'XYT'")

    ncells = Vt.shape[1]

    Dx, Dy = _grad_central(D_matrix, h)   # (ncells+2, ncells+2)
    Dx_core = Dx[1:-1, 1:-1]              # (ncells, ncells)
    Dy_core = Dy[1:-1, 1:-1]              # (ncells, ncells)

    # Electrode positions with small nudge to avoid exact integers
    ex = np.array(elecposX, dtype=np.float32, copy=True)
    ey = np.array(elecposY, dtype=np.float32, copy=True)
    ex[np.isclose(ex % 1.0, 0.0)] += 1e-4
    ey[np.isclose(ey % 1.0, 0.0)] += 1e-4
    elecpos = np.stack([ex, ey], axis=1)

    x_coords = np.arange(0, ncells+2, dtype=np.float32)  # (ncells+2,)
    y_coords = np.arange(0, ncells+2, dtype=np.float32)  # (ncells+2,)
    x_phys = jnp.asarray(x_coords * h, dtype=dtype)
    y_phys = jnp.asarray(y_coords * h, dtype=dtype)

    # Convert to JAX arrays
    Vt_jax = jnp.asarray(Vt, dtype=dtype)
    Dx_jax = jnp.asarray(Dx_core, dtype=dtype)
    Dy_jax = jnp.asarray(Dy_core, dtype=dtype)

    ex_jax = jnp.asarray(ex, dtype=dtype)
    ey_jax = jnp.asarray(ey, dtype=dtype)
    x_coords_jax = jnp.asarray(x_coords, dtype=dtype)
    y_coords_jax = jnp.asarray(y_coords, dtype=dtype)

    @jax.jit
    def compute_phie_all_timesteps(Vt_arr, D_matrix, Dx_core, Dy_core, ex, ey, x_coords, y_coords, x_phys, y_phys):
        """Compute phie for all timesteps - JIT compiled."""

        def compute_phie_one_timestep(V: jnp.ndarray) -> jnp.ndarray:
            """Compute phie for one timestep and all electrodes."""
            # V is (ncells, ncells) core only
            # Use jnp.gradient which handles boundaries with forward/backward diff
            gx_core = jnp.gradient(V, h, axis=0)  # (ncells, ncells)
            gy_core = jnp.gradient(V, h, axis=1)  # (ncells, ncells)

            # Compute Laplacian using jnp.gradient
            # ∇²V = ∂²V/∂x² + ∂²V/∂y²
            gxx = jnp.gradient(gx_core, h, axis=0)  # ∂²V/∂x²
            gyy = jnp.gradient(gy_core, h, axis=1)  # ∂²V/∂y²
            lap_core = gxx + gyy

            D_core = D_matrix[1:-1, 1:-1]
            du_core = D_core * lap_core + Dx_core * gx_core + Dy_core * gy_core

            def compute_phie_one_electrode(elec_x, elec_y):
                """Compute phie for one electrode."""
                # Create 2D meshgrid for distance calculation
                # Core indices are 1 to ncells (in 0-indexed: [1:-1])
                x_core = x_coords[1:-1]  # (ncells,)
                y_core = y_coords[1:-1]  # (ncells,)
                # Create 2D grid: X[i,j] has x-coord, Y[i,j] has y-coord
                X, Y = jnp.meshgrid(x_core, y_core, indexing='ij')
                distance = jnp.sqrt((X - elec_x)**2 + (Y - elec_y)**2) * h   # (ncells, ncells)

                # 2D trapezoidal integration of du_core / distance
                integrand = du_core / distance
                inner = _trapz_jax(integrand, x_phys[1:-1], axis=0)  # Integrate over x first
                result = -_trapz_jax(inner, y_phys[1:-1], axis=0)    # Then integrate over y
                return result

            # Vectorize over all electrodes
            phie_t = jax.vmap(compute_phie_one_electrode)(ex, ey)
            return phie_t

        # Vectorize over all timesteps
        phie = jax.vmap(compute_phie_one_timestep)(Vt_arr)  # (T, E)
        return phie

    # Compute phie: (T, E)
    phie_t = compute_phie_all_timesteps(
        Vt_jax, D_matrix, Dx_jax, Dy_jax, ex_jax, ey_jax,
        x_coords_jax, y_coords_jax, x_phys, y_phys
    )

    # Transpose to (E, T) to match expected output
    phie = jnp.transpose(phie_t, (1, 0))

    return np.array(phie), elecpos


#%%
# =============================================================================
# 5. VIDEO
# =============================================================================
def save_VW_video(Vsav, Wsav, overlay_mask=None, filename="aliev_panfilov.mp4",
                  fps=20, vmin=0.0, vmax=1.0, dt=0.01, gathert=10,
                  add_labels=True, add_time=True, separator_width=4,
                  upscale_factor=4):
    """
    Save a video of V (left panel) and W (right panel).

    Parameters:
    -----------
    Vsav, Wsav : (frames, ncells, ncells) ndarray
        Saved states from simulate()
    overlay_mask : boolean ndarray, optional
        Regions to darken in the video, e.g. fibrotic patches. May include the
        ghost border. If None, nothing is overlaid.
    filename : str
        Output path (.mp4). A .gif is written instead if ffmpeg is unavailable.
    fps : int
        Frames per second
    vmin, vmax : float
        Limits of the colour scale
    dt, gathert : float, int
        Time step and save interval of the simulation, used for the timestamp
    separator_width : int
        Width of white separator between V and W panels (pixels)
    add_labels : bool
        Add "V" and "W" titles to each panel
    add_time : bool
        Add a timestamp (bottom left)
    upscale_factor : int
        Upscale frames by this factor for better text quality (e.g., 4 = 400x400 from 100x100)
    """
    T = Vsav.shape[0]
    viridis = plt.get_cmap("viridis")

    # Normalize function
    def to_rgb(x, mask):
        x_norm = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb = viridis(x_norm)[:, :, :3]  # (H, W, 3) float in [0,1]
        # Darken the overlay regions
        rgb[mask] *= 0.5
        return (255 * rgb).astype(np.uint8)

    # Regions to darken (ghost border removed if present)
    if overlay_mask is None:
        overlay = np.zeros(Vsav.shape[1:], dtype=bool)
    elif overlay_mask.shape[0] == Vsav.shape[1] + 2:
        overlay = overlay_mask[1:-1, 1:-1].astype(bool)
    else:
        overlay = overlay_mask.astype(bool)

    # Try to load a font (fallback to default if not available)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    def build_frame(k: int) -> np.ndarray:
        V_rgb = to_rgb(Vsav[k], overlay)
        W_rgb = to_rgb(Wsav[k], overlay)

        # Upscale for better text quality
        if upscale_factor > 1:
            V_img = Image.fromarray(V_rgb).resize(
                (V_rgb.shape[1] * upscale_factor, V_rgb.shape[0] * upscale_factor),
                Image.LANCZOS,
            )
            W_img = Image.fromarray(W_rgb).resize(
                (W_rgb.shape[1] * upscale_factor, W_rgb.shape[0] * upscale_factor),
                Image.LANCZOS,
            )
            V_rgb = np.array(V_img)
            W_rgb = np.array(W_img)

        # Create separator (white vertical bar)
        if separator_width > 0:
            sep = np.ones((V_rgb.shape[0], separator_width * upscale_factor, 3), dtype=np.uint8) * 255
            frame = np.concatenate([V_rgb, sep, W_rgb], axis=1)
        else:
            frame = np.concatenate([V_rgb, W_rgb], axis=1)

        # Add labels using PIL (much faster than matplotlib)
        if add_labels or add_time:
            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            if add_labels:
                v_x = V_rgb.shape[1] // 2 - 10
                w_x = V_rgb.shape[1] + separator_width * upscale_factor + W_rgb.shape[1] // 2 - 10
                draw.text((v_x, 5), "V", fill=(255, 255, 255), font=font_large)
                draw.text((w_x, 5), "W", fill=(255, 255, 255), font=font_large)

            if add_time:
                t_tu = (k + 1) * dt * gathert   # frame k is saved after (k + 1) * gathert steps
                t_ms = t_tu * MS_PER_TU
                time_str = f"Time = {t_tu:.2f} TU ({t_ms:.2f} ms)"
                draw.text((10, frame.shape[0] - 25), time_str, fill=(255, 255, 255), font=font_small)

            frame = np.array(img)

        return frame

    # Prefer ffmpeg-backed mp4 writing; fallback to gif if ffmpeg/plugin support is unavailable.
    try:
        with imageio.get_writer(filename, format="FFMPEG", mode="I", fps=fps, codec="libx264") as writer:
            for k in range(T):
                writer.append_data(build_frame(k))
        print(f"Video saved: {filename}")
    except Exception as exc:
        gif_filename = os.path.splitext(filename)[0] + ".gif"
        print(f"MP4 writer unavailable ({exc}); falling back to GIF: {gif_filename}")
        with imageio.get_writer(gif_filename, mode="I", duration=1.0 / max(fps, 1)) as writer:
            for k in range(T):
                writer.append_data(build_frame(k))
        print(f"Video saved: {gif_filename}")


#%%
# =============================================================================
# 6. RUN AND SAVE
# =============================================================================
def run_planar_simulation(params: Params,
                          compute_phie: bool = False,
                          save_dir: Optional[str] = "planar_output",
                          name: str = "planar_wave",
                          save_video: bool = True,
                          numelec_x: int = 10,
                          numelec_y: int = 10) -> Dict[str, np.ndarray]:
    """
    Run one simulation and save the results.

    Parameters:
    -----------
    params : Params
        Simulation settings, including the stimulus mask (see planar_stim_mask)
    compute_phie : bool
        If True, also compute the extracellular potential (phie) on a
        numelec_x x numelec_y grid of electrodes. Off by default: it is not
        needed to study the action potentials.
    save_dir : str, optional
        Folder for the output files. If None, nothing is saved to disk.
    name : str
        Name of the output files, without extension. Use a different name for
        each run so that results are not overwritten.
    save_video : bool
        Also save a video of V and W
    numelec_x, numelec_y : int
        Size of the electrode grid (only used if compute_phie is True)

    Returns:
    --------
    dict containing:
        Vsav, Wsav : (frames, ncells, ncells)
        t_sav : (frames,) time of each saved frame
        Vpos : (ncells, ncells, 2), Vpos[i, j] is the (row, col) position of Vsav[:, i, j]
        phie : (E, frames) and elecpos : (E, 2), only if compute_phie is True

    The same arrays are saved to <save_dir>/<name>.npz, and the video to
    <save_dir>/<name>.mp4.

    Note:
    -----
    Vpos and elecpos are in the same length unit as h and share one frame of
    reference, that of the grid including its ghost border. The first interior
    cell is therefore at (h, h).
    """
    print(f"Running simulation '{name}'...")
    out = simulate(params)
    results = {"Vsav": out["Vsav"], "Wsav": out["Wsav"], "t_sav": out["t_sav"]}

    # Position of every interior cell: interior cell [i, j] is cell [i+1, j+1] of the full grid
    coords = (np.arange(params.ncells, dtype=np.float32) + 1.0) * params.h
    rows, cols = np.meshgrid(coords, coords, indexing="ij")
    results["Vpos"] = np.stack([rows, cols], axis=-1)

    if compute_phie:
        print("  Calculating phie...")
        X = params.ncells + 2
        elecposX, elecposY = make_electrodes_from_domain(
            X=X, Y=X, numelec_x=numelec_x, numelec_y=numelec_y
        )
        phie, elecpos = calc_phie_jax(
            Vsav=out["Vsav"],
            h=params.h,
            D_matrix=_ensure_fields(params)["D"],
            elecposX=elecposX,
            elecposY=elecposY,
        )
        results["phie"] = phie
        results["elecpos"] = elecpos * params.h

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{name}.npz")
        np.savez_compressed(filename, **results)
        print(f"  Saved to {filename}")

        if save_video:
            save_VW_video(
                out["Vsav"],
                out["Wsav"],
                filename=os.path.join(save_dir, f"{name}.mp4"),
                dt=params.dt,
                gathert=params.gathert,
            )

    return results


def run_planar_batch(edges: Sequence[str] = ("top", "bottom", "left", "right"),
                     base_params: Optional[Params] = None,
                     stim_width: int = 5,
                     compute_phie: bool = False,
                     save_dir: str = "planar_output",
                     save_video: bool = True) -> None:
    """
    Example batch: one planar-wave simulation per pacing location.

    Each run is saved as <save_dir>/planar_<edge>.npz (and .mp4). Nothing is
    returned, so memory use stays flat however long the batch is; load a
    result back with np.load.

    This is meant as a template. To vary something else, loop over a different
    list and build the Params of each run with dataclasses.replace, as below.

    Parameters:
    -----------
    edges : sequence of str
        Pacing locations, any of 'top', 'bottom', 'left', 'right'
    base_params : Params, optional
        Settings shared by all runs (default: Params()). The stimulus mask is
        set separately for each run.
    stim_width : int
        Width of the stimulated strip, in cells
    compute_phie, save_dir, save_video :
        As in run_planar_simulation
    """
    if base_params is None:
        base_params = Params()

    for i, edge in enumerate(edges, start=1):
        print(f"\n[{i}/{len(edges)}] Planar wave paced from the {edge} edge")
        stim_mask = planar_stim_mask(base_params.ncells, width=stim_width, edge=edge)
        params = replace(base_params, stim_mask=stim_mask)

        run_planar_simulation(
            params,
            compute_phie=compute_phie,
            save_dir=save_dir,
            name=f"planar_{edge}",
            save_video=save_video,
        )

        # Free memory before the next run
        gc.collect()
        jax.clear_caches()


#%%
# =============================================================================
# 7. EXAMPLE
# =============================================================================
if __name__ == "__main__":
    COMPUTE_PHIE = False   # True: also compute and save phie and the electrode positions
    RUN_BATCH = False      # True: run one simulation per pacing edge instead of a single one

    print("JAX devices:", jax.devices())

    if RUN_BATCH:
        run_planar_batch(edges=("top", "bottom", "left", "right"), compute_phie=COMPUTE_PHIE)
    else:
        params = Params(stim_mask=planar_stim_mask(ncells=100, width=5, edge="top"))
        out = run_planar_simulation(params, compute_phie=COMPUTE_PHIE)
        print("Vsav:", out["Vsav"].shape, " Wsav:", out["Wsav"].shape, " t_sav:", out["t_sav"].shape)
