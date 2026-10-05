# 2D Aliev-Panfilov solver (JAX)

Simulates electrical waves in a square sheet of cardiac tissue with the
Aliev-Panfilov model. Each run saves the voltage `V` and the recovery variable
`W` over time and a video of both. It can also compute the extracellular
potential (`phie`) recorded by a grid of electrodes.

## Install

Needs Python 3.10 or newer.

    conda env create -f environment.yml
    conda activate aliev-panfilov

or, in an existing environment:

    pip install -r requirements.txt

The notes at the top of `requirements.txt` cover GPU support and Apple-silicon
Macs.

## Run

    python solveAP_2D_jax.py

This paces one beat from the top edge of a 100 x 100 sheet and writes
`planar_output/planar_wave.npz` and `planar_output/planar_wave.mp4`. It takes a
few seconds on a laptop CPU.

The same from Python:

```python
from solveAP_2D_jax import Params, planar_stim_mask, run_planar_simulation

params = Params(stim_mask=planar_stim_mask(ncells=100, edge="top"))
out = run_planar_simulation(params, name="planar_wave")
V = out["Vsav"]   # (frames, ncells, ncells)
```

The files are split into `#%%` cells. In the VS Code interactive window, run the
cells from the top: the imports cell has to run before any other.

## Pacing somewhere else

`planar_stim_mask` paces a strip along the `"top"`, `"bottom"`, `"left"` or
`"right"` edge. For any other site, build the mask yourself: an
`(ncells, ncells)` array that is 1 where the tissue is stimulated.

```python
import numpy as np
from solveAP_2D_jax import Params, run_planar_simulation

mask = np.zeros((100, 100), dtype=np.float32)
mask[45:55, 20:30] = 1.0   # rows 45-54, columns 20-29; row 0 is the top of the video
out = run_planar_simulation(Params(stim_mask=mask), name="point_pacing")
```

Give each run its own `name`, or it overwrites the previous one.
`run_planar_batch` runs one simulation per edge and is a template for looping
over anything else.

## Output

`run_planar_simulation` returns these arrays and saves them to
`<save_dir>/<name>.npz`:

| Array | Shape | Meaning |
|---|---|---|
| `Vsav`, `Wsav` | (frames, ncells, ncells) | `V` and `W` at each saved frame |
| `t_sav` | (frames,) | time of each frame |
| `Vpos` | (ncells, ncells, 2) | (row, col) position of each cell |
| `phie` | (electrodes, frames) | only with `compute_phie=True` |
| `elecpos` | (electrodes, 2) | (row, col) position of each electrode, only with `compute_phie=True` |

Time is in model time units; one unit is 12.9 ms. Positions are in the same
length unit as the grid spacing `h`. The first cell is at `(h, h)`, because the
solver keeps a one-cell border around the sheet to enforce the no-flux boundary.

## Settings

Everything is set through `Params`. The defaults are a 100 x 100 sheet
(`ncells`) with spacing `h = 0.1`, time step `dt = 0.01`, 50 time units
(`tend`), a frame saved every 10 steps (`gathert`), and one beat (`ncyc`) from
a stimulus of amplitude 0.1 (`stim_amp_scale`) lasting 2 time units
(`stimdur`). With several beats, `BCL` is the time between them.

The model is

    dV/dt = -k V (V - a)(V - 1) - V W + div(D grad V) + I_stim
    dW/dt = (epsi + mu1 W / (mu2 + V)) (-W - k V (V - b - 1))

with defaults `k = 8`, `a0 = 0.01`, `b0 = 0.15`, `mu1 = 0.2`, `mu2 = 0.3`,
`epsi = 0.002` and `D_scalar = 0.1`.

## Files

- `solveAP_2D_jax.py`: the solver, `phie`, video and the planar-wave runners.
  This is all you need for homogeneous tissue. Its docstring at the top lists
  the contents.
- `solveAP_2D_heterogeneity.py`: fibrotic and hyper-excitable patches, random
  pacing sites, a spiral-wave stimulus and dataset generators. It imports the
  solver, so keep both files in the same folder.

## Known limitations of solveAP_2D_heterogeneity.py

- A paced beat does not start a wave if its site is still refractory. With the
  default three beats 50 time units apart, this happened in one of three test
  runs, so check `phie` or the video before assuming a run has three waves.
- `make_D_with_rect_patches` retries forever if the patches cannot fit, for
  example several patches on a small grid.
- `fiblocs` is saved as an object array: load the file with
  `np.load(..., allow_pickle=True)`. For a run with one patch, convert it with
  `.astype(int)` before using it as an index.
- `stim_centers` are indices on the sheet itself, while `fiblocs` and `D_matrix`
  include the one-cell border. Add 1 to a stimulus centre to compare them.
