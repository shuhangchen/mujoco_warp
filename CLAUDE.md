# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuJoCo Warp (MJWarp) is a GPU-optimized version of the MuJoCo physics simulator built on NVIDIA Warp. It provides the same physics simulation as MuJoCo but runs on GPU via Warp kernels.

## Common Commands

```bash
# Install for development (requires CUDA 12.4+)
uv pip install -e .[dev,cuda]

# Run all tests
pytest

# Run a single test file
pytest mujoco_warp/_src/forward_test.py

# Run a specific test
pytest mujoco_warp/_src/forward_test.py::ForwardTest::test_step -v

# Run tests on CPU
pytest --cpu

# Run tests with CUDA error checking
pytest --verify_cuda

# Lint and format
ruff check .
ruff format .

# Interactive viewer
mjwarp-viewer benchmark/humanoid/humanoid.xml

# Benchmarking
mjwarp-testspeed benchmark/humanoid/humanoid.xml
mjwarp-testspeed benchmark/humanoid/humanoid.xml --event_trace=True

# Kernel analyzer (run before PRs)
python contrib/kernel_analyzer/kernel_analyzer/cli.py mujoco_warp/_src/*.py --types mujoco_warp/_src/types.py
```

## Code Style

- 2-space indentation, 128 char line length
- Google docstring convention
- Single-line imports (enforced by ruff)
- Run `ruff format .` before committing

## Architecture

### Data Flow: CPU → GPU → CPU

```
mujoco.MjModel/MjData  →  put_model()/put_data()  →  Model/Data (GPU)
                                                          ↓
                                                      step(m, d)
                                                          ↓
                       get_data_into()  ←  Model/Data (GPU)
```

### Physics Pipeline (`step()` in forward.py)

```
step(m, d)
  └── forward(m, d)
        ├── fwd_position(m, d)     # Kinematics, COM, tendons, inertia
        │     ├── kinematics()     # Forward kinematics
        │     ├── com_pos()        # Center of mass
        │     ├── tendon()         # Tendon routing
        │     ├── crb()            # Composite rigid body inertia
        │     └── factor_m()       # Mass matrix factorization
        ├── sensor_pos()           # Position sensors
        ├── fwd_velocity(m, d)     # Velocity-dependent quantities
        ├── sensor_vel()           # Velocity sensors
        ├── fwd_actuation(m, d)    # Actuator forces
        ├── fwd_acceleration(m, d) # Sum forces → qacc_smooth
        ├── collision(m, d)        # Contact detection
        ├── make_constraint()      # Build constraint Jacobian
        └── solve(m, d)            # Constraint solver
  └── euler/rk4/implicit(m, d)     # Time integration
```

### Core Modules in `mujoco_warp/_src/`

| Module | Purpose |
|--------|---------|
| `types.py` | `Model`, `Data`, `State`, `Contact`, `Constraint` dataclasses and enums |
| `io.py` | `put_model()`, `put_data()`, `get_data_into()`, `make_data()` - CPU↔GPU transfer |
| `forward.py` | `step()`, `forward()`, `euler()`, `rk4()` - main simulation loop |
| `smooth.py` | `kinematics()`, `crb()`, `rne()`, `tendon()` - kinematics and dynamics |
| `solver.py` | `solve()` - constraint solver (CG, Newton) |
| `collision_driver.py` | `collision()`, `nxn_broadphase()`, `sap_broadphase()` |
| `collision_primitive.py` | Sphere, capsule, cylinder, box collision |
| `collision_gjk.py` | GJK algorithm for convex shapes |
| `collision_sdf.py` | SDF-based collision |
| `constraint.py` | `make_constraint()` - builds constraint Jacobian |
| `sensor.py` | `sensor_pos()`, `sensor_vel()`, `sensor_acc()` |
| `support.py` | `mul_m()`, `solve_m()`, `contact_force()` |
| `ray.py` | `ray()`, `rays()` - ray casting |
| `render.py` | `render()` - GPU raytracing renderer |

### Warp Kernel Patterns

Kernels use `@wp.kernel` decorator and are launched with `wp.launch()` or `wp.launch_tiled()`:

```python
@wp.kernel
def _my_kernel(
  # Model arrays (read-only):
  model_array: wp.array(dtype=float),
  # Data in:
  data_in: wp.array2d(dtype=float),
  # Data out:
  data_out: wp.array2d(dtype=float),
):
  worldid, itemid = wp.tid()  # Get thread indices
  # ... kernel body
```

First dimension is typically `worldid` (batch dimension), second is the item being processed.

### Testing

Tests compare MJWarp outputs against canonical MuJoCo C implementation with ~5e-5 tolerance. Test files are named `*_test.py` in `_src/`.

### Kernel Analyzer

Before submitting PRs, run the kernel analyzer to check for issues. It validates Warp kernel correctness and is also available as a VSCode plugin in `contrib/kernel_analyzer/`.
