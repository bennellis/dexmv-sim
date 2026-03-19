"""Top-level package for hand_imitation.

The original package eagerly imported the MuJoCo environment stack here.
That makes lightweight utilities under ``hand_imitation.misc`` unusable in
environments without ``mujoco_py``. Keep the kinematics import eager, but make
the environment import optional so Isaac/URDF-based tooling can still run.
"""

import hand_imitation.kinematics

try:
    import hand_imitation.env  # noqa: F401
except ModuleNotFoundError as exc:
    # Allow non-MuJoCo utilities to import cleanly when mujoco_py is absent.
    if exc.name != "mujoco_py":
        raise
