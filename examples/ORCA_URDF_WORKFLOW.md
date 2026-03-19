## ORCA URDF / Isaac Workflow

This workspace now treats the ORCA retarget path as:

1. Offline retargeting from HOT3D UME-TRACK to ORCA joint angles using ORCA URDF kinematics.
2. Visual verification of UME fingertip indices and ORCA correspondence points.
3. Playback / inspection in Isaac Sim.

Use these scripts in order:

1. Verify setup:

```bash
python dexmv-sim/examples/visualize_umetrack_orca_setup.py \
  --recording-dir /path/to/hot3d_recording \
  --hand-id 0 \
  --hand-side right \
  --orca-side right \
  --show-joint-labels
```

2. Retarget with the URDF-based solver:

```bash
python dexmv-sim/examples/retarget_umetrack_to_orca.py \
  --recording-dir /path/to/hot3d_recording \
  --hand-id 0 \
  --hand-side right \
  --orca-side right \
  --target-set palm_mids \
  --output-npz /tmp/umetrack_to_orca.npz
```

3. Replay in Isaac Sim using the existing viewer:

```bash
cd IsaacLab
TERM=xterm ./isaaclab.sh -p ../dexterous_manipulation/scripts/visualize_orca_hand_trajectory.py \
  --npz-path /tmp/umetrack_to_orca.npz \
  --side right \
  --loop
```

Notes:

- The UME fingertip map in `dexterous_manipulation/scripts/umetrack_tip_indices.json` still needs manual validation.
- ORCA tip offsets in `hand_imitation/misc/orca_adapter.py` are still a first-pass approximation and may need tuning.
- The DexMV MuJoCo optimizer is not used for the ORCA pipeline anymore.
