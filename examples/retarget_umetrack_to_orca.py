from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy import signal
try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency in some environments
    yaml = None


SCRIPT_PATH = Path(__file__).resolve()
DEXMV_DIR = SCRIPT_PATH.parents[1]
WORKSPACE_DIR = SCRIPT_PATH.parents[2]
for path in [DEXMV_DIR, WORKSPACE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexterous_manipulation.utils.hot3d_data import load_umetrack_hand_trajectory
from dexterous_manipulation.utils.hot3d_models import quat_wxyz_to_rotmat
from hand_imitation.misc.orca_adapter import ORCAHandAdapter
from hand_imitation.misc.umetrack_adapter import UMETrackHandAdapter


DEFAULT_UMETRACK_TIP_MAP_JSON = (
    WORKSPACE_DIR / "dexterous_manipulation" / "scripts" / "umetrack_tip_indices.json"
)
DEFAULT_ORCA_URDF = WORKSPACE_DIR / "orcahand_description" / "models" / "urdf" / "orcahand_right_extended.urdf"
DEFAULT_UME_TO_ORCA_X_OFFSET_DEG = 90.0
DEFAULT_UME_TO_ORCA_Y_OFFSET_DEG = 90.0
DEFAULT_UME_TO_ORCA_Z_OFFSET_DEG = -90.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Retarget HOT3D UME-TRACK trajectory to ORCA using ORCA URDF kinematics")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON or YAML config file. CLI flags override config values.",
    )
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--umetrack-jsonl", type=str, default="umetrack_hand_pose_trajectory.jsonl")
    parser.add_argument("--umetrack-profile-json", type=str, default="umetrack_hand_user_profile.json")
    parser.add_argument("--umetrack-tip-map-json", type=Path, default=DEFAULT_UMETRACK_TIP_MAP_JSON)
    parser.add_argument("--umetrack-unit-scale", type=float, default=0.001)
    parser.add_argument("--umetrack-left-mirror-x", action="store_true", default=False)
    parser.add_argument(
        "--umetrack-right-mirror-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror UME-TRACK local X for right hands.",
    )
    parser.add_argument("--hand-id", type=str, default="0")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--orca-side", choices=("left", "right"), default="right")
    parser.add_argument("--orca-urdf", type=Path, default=DEFAULT_ORCA_URDF)
    parser.add_argument("--target-set", choices=("palm_mids", "palm_mids_tips"), default="palm_mids")
    parser.add_argument(
        "--fit-similarity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit a fixed hand-frame rotation between UME-TRACK and ORCA targets. Disabled by default to use the manual alignment.",
    )
    parser.add_argument(
        "--fit-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also fit a uniform scale. Leave disabled for metric interaction retargeting.",
    )
    parser.add_argument(
        "--wrist-pose-mode",
        choices=("mapped_orca_root", "raw_umetrack"),
        default="mapped_orca_root",
        help="How to export wrist/root world pose for playback.",
    )
    parser.add_argument("--filter-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-wn", type=float, default=5.0)
    parser.add_argument("--filter-fs", type=float, default=100.0)
    parser.add_argument("--reg-weight", type=float, default=0.03)
    parser.add_argument(
        "--optimize-palm-delta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimize a 6-DoF palm-frame correction on top of the current hand-frame initialization.",
    )
    parser.add_argument(
        "--palm-delta-pos-reg-weight",
        type=float,
        default=25.0,
        help="Regularization weight on palm translation delta (meters) toward zero.",
    )
    parser.add_argument(
        "--palm-delta-rot-reg-weight",
        type=float,
        default=25.0,
        help="Regularization weight on palm rotation-vector delta (radians) toward zero.",
    )
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument(
        "--print-joint-limits",
        action="store_true",
        default=False,
        help="Print the exact ORCA joint limits used by the optimizer.",
    )
    parser.add_argument(
        "--joint-limit-override",
        action="append",
        default=[],
        help="Override a joint limit as 'joint_name:lower:upper'. Repeat for multiple joints.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start retargeting at this frame index after timestamp filtering and before stride/num-frame slicing.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=-1,
        help="Number of frames to optimize after --start-frame and --stride. -1 means all remaining frames.",
    )
    parser.add_argument("--start-timestamp-ns", type=int, default=None)
    parser.add_argument("--end-timestamp-ns", type=int, default=None)
    parser.add_argument(
        "--consolidate-close-timestamps-ns",
        type=int,
        default=0,
        help=(
            "Collapse consecutive UME-TRACK timestamps separated by at most this many nanoseconds "
            "before frame-based slicing. Use 1000000 to consolidate the HOT3D near-duplicate <1 ms groups."
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--output-pkl", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--verbose-every", type=int, default=50)
    return parser


def _action_expects_path(action: argparse.Action) -> bool:
    if getattr(action, "type", None) is Path:
        return True
    return isinstance(getattr(action, "default", None), Path)


def _config_value_to_argv(action: argparse.Action, value, config_dir: Path) -> list[str]:
    option_strings = list(getattr(action, "option_strings", []))
    if not option_strings or value is None:
        return []

    if isinstance(action, argparse.BooleanOptionalAction):
        positive = next((opt for opt in option_strings if opt.startswith("--") and not opt.startswith("--no-")), option_strings[0])
        negative = next((opt for opt in option_strings if opt.startswith("--no-")), None)
        if bool(value):
            return [positive]
        return [negative] if negative is not None else []

    if action.nargs == 0 and isinstance(getattr(action, "default", None), bool):
        desired = bool(value)
        if desired == bool(action.default):
            return []
        preferred = next((opt for opt in option_strings if opt.startswith("--")), option_strings[0])
        return [preferred]

    preferred = next((opt for opt in option_strings if opt.startswith("--") and not opt.startswith("--no-")), option_strings[0])

    def _normalize_scalar(v):
        if _action_expects_path(action):
            path_value = Path(v)
            if not path_value.is_absolute():
                path_value = (config_dir / path_value).resolve()
            return str(path_value)
        return str(v)

    if isinstance(value, (list, tuple)):
        return [preferred, *[_normalize_scalar(v) for v in value]]
    return [preferred, _normalize_scalar(value)]


def _load_config_dict(config_path: Path) -> dict:
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            raw = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ImportError(
                    "YAML config support requires PyYAML. Install it in the active environment or use a JSON config."
                )
            raw = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file type for {config_path}. Use .json, .yaml, or .yml.")
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping/object at the top level: {config_path}")
    return raw


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(raw_argv)
    if config_args.config is not None:
        config_path = config_args.config.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        raw = _load_config_dict(config_path)
        action_by_dest = {
            action.dest: action
            for action in parser._actions
            if getattr(action, "dest", None) not in {None, argparse.SUPPRESS, "help"}
        }
        config_argv: list[str] = []
        ignored_keys: list[str] = []
        for key, value in raw.items():
            if key == "config":
                continue
            action = action_by_dest.get(str(key))
            if action is None:
                ignored_keys.append(str(key))
                continue
            config_argv.extend(_config_value_to_argv(action, value, config_path.parent))
        if ignored_keys:
            print(f"[WARN] Ignoring unknown config key(s): {', '.join(sorted(ignored_keys))}", flush=True)
        raw_argv = config_argv + raw_argv

    return parser.parse_args(raw_argv)


def load_tip_idx_by_finger(path: Path, hand_side: str) -> dict[str, int]:
    with path.resolve().open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("tip_idx_by_finger", {})
    if hand_side in raw:
        return {k: int(v) for k, v in raw[hand_side].items()}
    return {k: int(v) for k, v in raw.items()}


def filter_timestamps_by_range(timestamps: list[int], start_ns: int | None, end_ns: int | None) -> list[int]:
    out = list(timestamps)
    if start_ns is not None:
        out = [ts for ts in out if ts >= int(start_ns)]
    if end_ns is not None:
        out = [ts for ts in out if ts <= int(end_ns)]
    return out


def consolidate_close_timestamps(
    timestamps: list[int], threshold_ns: int
) -> tuple[list[int], dict[str, int]]:
    out = list(timestamps)
    threshold_ns = int(threshold_ns)
    if threshold_ns <= 0 or len(out) <= 1:
        return out, {
            "input_count": len(out),
            "output_count": len(out),
            "removed_count": 0,
            "group_count": len(out),
            "max_group_size": 1 if out else 0,
        }

    consolidated = [out[0]]
    group_count = 1
    group_size = 1
    max_group_size = 1
    prev_ts = out[0]
    for ts in out[1:]:
        if int(ts) - int(prev_ts) <= threshold_ns:
            group_size += 1
            max_group_size = max(max_group_size, group_size)
        else:
            consolidated.append(ts)
            group_count += 1
            group_size = 1
        prev_ts = ts
    return consolidated, {
        "input_count": len(out),
        "output_count": len(consolidated),
        "removed_count": len(out) - len(consolidated),
        "group_count": group_count,
        "max_group_size": max_group_size,
    }


def apply_timestamp_filters(
    timestamps: list[int],
    start_frame: int,
    stride: int,
    num_frames: int,
    max_frames: int,
) -> list[int]:
    out = list(timestamps)
    if not out:
        return out
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if start_frame >= len(out):
        raise ValueError(f"--start-frame={start_frame} out of range for filtered trajectory length={len(out)}")
    out = out[start_frame:]
    if stride > 1:
        out = out[::stride]
    if num_frames == 0 or num_frames < -1:
        raise ValueError("--num-frames must be -1 (all) or positive")
    if num_frames > 0:
        out = out[:num_frames]
    if max_frames > 0:
        out = out[:max_frames]
    return out


def fit_similarity(src_points: np.ndarray, dst_points: np.ndarray, fit_scale: bool) -> tuple[float, np.ndarray]:
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src0 = src_points - src_mean
    dst0 = dst_points - dst_mean

    h_mat = src0.T @ dst0
    u_mat, singular_values, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0.0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T

    if fit_scale:
        denom = float(np.sum(src0 * src0))
        scale = 1.0 if denom < 1e-12 else float(np.sum(singular_values) / denom)
    else:
        scale = 1.0
    return scale, rot


def rot_x_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c_val = float(np.cos(angle_rad))
    s_val = float(np.sin(angle_rad))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_val, -s_val],
            [0.0, s_val, c_val],
        ],
        dtype=np.float64,
    )


def rot_y_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c_val = float(np.cos(angle_rad))
    s_val = float(np.sin(angle_rad))
    return np.asarray(
        [
            [c_val, 0.0, s_val],
            [0.0, 1.0, 0.0],
            [-s_val, 0.0, c_val],
        ],
        dtype=np.float64,
    )


def rot_z_deg(angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(float(angle_deg))
    c_val = float(np.cos(angle_rad))
    s_val = float(np.sin(angle_rad))
    return np.asarray(
        [
            [c_val, -s_val, 0.0],
            [s_val, c_val, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotvec_to_rotmat(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / theta
    x_val, y_val, z_val = axis.tolist()
    k_mat = np.asarray(
        [
            [0.0, -z_val, y_val],
            [z_val, 0.0, -x_val],
            [-y_val, x_val, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(theta) * k_mat + (1.0 - np.cos(theta)) * (k_mat @ k_mat)


def parse_joint_limit_overrides(raw_items: list[str]) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    for item in raw_items:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --joint-limit-override={item!r}. Expected format 'joint_name:lower:upper'."
            )
        name = parts[0].strip()
        if not name:
            raise ValueError(f"Invalid --joint-limit-override={item!r}: joint name is empty.")
        lo = float(parts[1])
        hi = float(parts[2])
        if lo > hi:
            lo, hi = hi, lo
        overrides[name] = (lo, hi)
    return overrides


def get_validated_joint_limits(
    adapter_dst: ORCAHandAdapter,
    joint_names: list[str],
    overrides: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    raw_limits = adapter_dst.joint_limits_by_name()
    validated: dict[str, tuple[float, float]] = {}
    missing: list[str] = []
    invalid: list[tuple[str, tuple[float, float] | None]] = []
    overrides = {} if overrides is None else dict(overrides)

    for name in joint_names:
        lim = overrides.get(name, raw_limits.get(name))
        if lim is None:
            missing.append(name)
            continue
        lo, hi = float(lim[0]), float(lim[1])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
            invalid.append((name, (lo, hi)))
            continue
        validated[name] = (lo, hi)

    if missing:
        raise KeyError(f"Missing URDF joint limit(s) for optimized joint(s): {missing}")
    if invalid:
        raise ValueError(f"Invalid URDF joint limit(s) for optimized joint(s): {invalid}")
    return validated


def filter_position_sequence(position_seq: np.ndarray, wn: float = 5.0, fs: float = 25.0) -> np.ndarray:
    sos = signal.butter(2, wn, "lowpass", fs=fs, output="sos", analog=False)
    seq = np.asarray(position_seq, dtype=np.float64)
    seq_shape = seq.shape
    if len(seq_shape) < 2:
        raise ValueError(f"Position sequence must have shape [T, D] or [T, N, D], got {seq_shape}")

    result_seq = np.empty_like(seq)
    padlen = 3 * (2 * sos.shape[0] + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))

    def _filter_1d(values: np.ndarray) -> np.ndarray:
        # Zero-phase filtering avoids the startup transient from sosfilt that was collapsing frame 0.
        if values.shape[0] <= padlen:
            return signal.sosfilt(sos, values)
        return signal.sosfiltfilt(sos, values)

    if len(seq_shape) == 3:
        for i in range(seq_shape[1]):
            for k in range(seq_shape[2]):
                result_seq[:, i, k] = _filter_1d(seq[:, i, k])
    elif len(seq_shape) == 2:
        for i in range(seq_shape[1]):
            result_seq[:, i] = _filter_1d(seq[:, i])
    else:
        raise ValueError(f"Unsupported position sequence shape: {seq_shape}")

    return result_seq


def retarget_sequence(
    target_pos_sequence: np.ndarray,
    feature_names: list[str],
    adapter_dst: ORCAHandAdapter,
    correspondences,
    joint_limits_by_name: dict[str, tuple[float, float]],
    rest_pose_by_name: dict[str, float],
    reg_weight: float,
    optimize_palm_delta: bool,
    palm_delta_pos_reg_weight: float,
    palm_delta_rot_reg_weight: float,
    max_nfev: int,
    verbose_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_order = adapter_dst.joint_order()
    wrist_joint_name = joint_order[0]
    opt_joint_names = joint_order[1:]

    q_lower = np.asarray([joint_limits_by_name[name][0] for name in opt_joint_names], dtype=np.float64)
    q_upper = np.asarray([joint_limits_by_name[name][1] for name in opt_joint_names], dtype=np.float64)
    q_prev = np.asarray(
        [float(rest_pose_by_name.get(name, 0.0)) for name in opt_joint_names],
        dtype=np.float64,
    )
    q_prev = np.clip(q_prev, q_lower, q_upper)
    sqrt_reg = float(np.sqrt(max(reg_weight, 0.0)))
    delta_t_prev = np.zeros(3, dtype=np.float64)
    delta_r_prev = np.zeros(3, dtype=np.float64)
    sqrt_palm_pos_reg = float(np.sqrt(max(palm_delta_pos_reg_weight, 0.0)))
    sqrt_palm_rot_reg = float(np.sqrt(max(palm_delta_rot_reg_weight, 0.0)))

    qpos_sequence: list[np.ndarray] = []
    palm_delta_t_sequence: list[np.ndarray] = []
    palm_delta_rotvec_sequence: list[np.ndarray] = []
    for index, target_pos in enumerate(target_pos_sequence):
        target_dict = {
            name: np.asarray(target_pos[k], dtype=np.float64)
            for k, name in enumerate(feature_names)
        }

        def residual_fn(x: np.ndarray) -> np.ndarray:
            q_x = x[: len(opt_joint_names)]
            delta_t = np.zeros(3, dtype=np.float64)
            delta_r = np.zeros(3, dtype=np.float64)
            if optimize_palm_delta:
                delta_t = np.asarray(x[len(opt_joint_names): len(opt_joint_names) + 3], dtype=np.float64)
                delta_r = np.asarray(x[len(opt_joint_names) + 3: len(opt_joint_names) + 6], dtype=np.float64)
            q_by_name = {wrist_joint_name: 0.0}
            for joint_name, qv in zip(opt_joint_names, q_x.tolist()):
                q_by_name[joint_name] = float(qv)

            model_points = adapter_dst.points_in_palm(q_by_name, correspondences)
            if optimize_palm_delta:
                palm_delta_rot = rotvec_to_rotmat(delta_r)
                model_points = {
                    name: palm_delta_rot @ np.asarray(point, dtype=np.float64) + delta_t
                    for name, point in model_points.items()
                }
            residuals: list[float] = []
            for cp in correspondences:
                delta = cp.weight * (model_points[cp.name] - target_dict[cp.name])
                residuals.extend(delta.tolist())
            if sqrt_reg > 0.0:
                residuals.extend((sqrt_reg * (q_x - q_prev)).tolist())
            if optimize_palm_delta:
                if sqrt_palm_pos_reg > 0.0:
                    residuals.extend((sqrt_palm_pos_reg * delta_t).tolist())
                if sqrt_palm_rot_reg > 0.0:
                    residuals.extend((sqrt_palm_rot_reg * delta_r).tolist())
            return np.asarray(residuals, dtype=np.float64)

        if optimize_palm_delta:
            x0 = np.concatenate([q_prev, delta_t_prev, delta_r_prev], axis=0)
            delta_t_bound = np.full(3, 0.10, dtype=np.float64)
            delta_r_bound = np.full(3, np.pi, dtype=np.float64)
            x_lower = np.concatenate([q_lower, -delta_t_bound, -delta_r_bound], axis=0)
            x_upper = np.concatenate([q_upper, delta_t_bound, delta_r_bound], axis=0)
        else:
            x0 = q_prev
            x_lower = q_lower
            x_upper = q_upper

        result = least_squares(
            residual_fn,
            x0=x0,
            bounds=(x_lower, x_upper),
            method="trf",
            loss="soft_l1",
            f_scale=1.0,
            ftol=1e-5,
            xtol=1e-5,
            gtol=1e-5,
            max_nfev=max(8, int(max_nfev)),
        )

        if result.success:
            q_opt = np.asarray(result.x[: len(opt_joint_names)], dtype=np.float64)
            if optimize_palm_delta:
                delta_t_opt = np.asarray(result.x[len(opt_joint_names): len(opt_joint_names) + 3], dtype=np.float64)
                delta_r_opt = np.asarray(result.x[len(opt_joint_names) + 3: len(opt_joint_names) + 6], dtype=np.float64)
            else:
                delta_t_opt = np.zeros(3, dtype=np.float64)
                delta_r_opt = np.zeros(3, dtype=np.float64)
        else:
            q_opt = q_prev.copy()
            delta_t_opt = delta_t_prev.copy()
            delta_r_opt = delta_r_prev.copy()

        q_prev = q_opt.copy()
        delta_t_prev = delta_t_opt.copy()
        delta_r_prev = delta_r_opt.copy()

        q_full = np.zeros((len(joint_order),), dtype=np.float32)
        q_full[0] = 0.0
        q_full[1:] = q_prev.astype(np.float32, copy=False)
        qpos_sequence.append(q_full)
        palm_delta_t_sequence.append(delta_t_prev.astype(np.float32, copy=False))
        palm_delta_rotvec_sequence.append(delta_r_prev.astype(np.float32, copy=False))

        if verbose_every > 0 and index % verbose_every == 0:
            print(
                f"[INFO] Frame {index + 1}/{len(target_pos_sequence)} "
                f"opt_status={result.status} cost={result.cost:.6f}",
                flush=True,
            )

    return (
        np.asarray(qpos_sequence, dtype=np.float32),
        np.asarray(palm_delta_t_sequence, dtype=np.float32),
        np.asarray(palm_delta_rotvec_sequence, dtype=np.float32),
    )


def summarize_joint_saturation(
    qpos_sequence: np.ndarray,
    joint_order: list[str],
    joint_limits_by_name: dict[str, tuple[float, float]],
    tol: float = 1e-3,
) -> None:
    if qpos_sequence.ndim != 2 or qpos_sequence.shape[0] == 0:
        return

    print(f"[INFO] Joint saturation summary (tol={tol:.1e} rad):", flush=True)
    any_hits = False
    for joint_idx, joint_name in enumerate(joint_order[1:], start=1):
        lo, hi = joint_limits_by_name[joint_name]
        values = np.asarray(qpos_sequence[:, joint_idx], dtype=np.float64)
        near_lo = int(np.count_nonzero(np.abs(values - lo) <= tol))
        near_hi = int(np.count_nonzero(np.abs(values - hi) <= tol))
        if near_lo == 0 and near_hi == 0:
            continue
        any_hits = True
        print(
            f"  - {joint_name}: min={values.min():.5f} max={values.max():.5f} "
            f"limit=[{lo:.5f}, {hi:.5f}] near_lo={near_lo}/{len(values)} near_hi={near_hi}/{len(values)}",
            flush=True,
        )
    if not any_hits:
        print("  - no joints were near their limits", flush=True)


def _orthonormalize_rotmat(rot: np.ndarray) -> np.ndarray:
    u_mat, _, vh_mat = np.linalg.svd(np.asarray(rot, dtype=np.float64))
    rot_ortho = u_mat @ vh_mat
    if np.linalg.det(rot_ortho) < 0.0:
        u_mat[:, -1] *= -1.0
        rot_ortho = u_mat @ vh_mat
    return rot_ortho


def _rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    rot = _orthonormalize_rotmat(np.asarray(rotmat, dtype=np.float64))
    trace = float(np.trace(rot))

    if trace > 0.0:
        s_val = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s_val
        qx = (rot[2, 1] - rot[1, 2]) / s_val
        qy = (rot[0, 2] - rot[2, 0]) / s_val
        qz = (rot[1, 0] - rot[0, 1]) / s_val
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s_val = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s_val
        qx = 0.25 * s_val
        qy = (rot[0, 1] + rot[1, 0]) / s_val
        qz = (rot[0, 2] + rot[2, 0]) / s_val
    elif rot[1, 1] > rot[2, 2]:
        s_val = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s_val
        qx = (rot[0, 1] + rot[1, 0]) / s_val
        qy = 0.25 * s_val
        qz = (rot[1, 2] + rot[2, 1]) / s_val
    else:
        s_val = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s_val
        qx = (rot[0, 2] + rot[2, 0]) / s_val
        qy = (rot[1, 2] + rot[2, 1]) / s_val
        qz = 0.25 * s_val

    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return quat


def _build_q_by_name(joint_order: list[str], qpos_row: np.ndarray) -> dict[str, float]:
    q = np.asarray(qpos_row, dtype=np.float64).reshape(-1)
    count = min(len(joint_order), int(q.shape[0]))
    q_by_name = {joint_order[i]: float(q[i]) for i in range(count)}
    for i in range(count, len(joint_order)):
        q_by_name[joint_order[i]] = 0.0
    return q_by_name


def compute_exported_wrist_pose(
    hand_states: list,
    qpos_sequence: np.ndarray,
    palm_delta_t_sequence: np.ndarray,
    palm_delta_rotvec_sequence: np.ndarray,
    adapter_src: UMETrackHandAdapter,
    adapter_dst: ORCAHandAdapter,
    rot_ume_to_orca: np.ndarray,
    wrist_pose_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if wrist_pose_mode == "raw_umetrack":
        wrist_t_world = np.stack([np.asarray(state.wrist_t, dtype=np.float32).reshape(3) for state in hand_states], axis=0)
        wrist_q_wxyz = np.stack([np.asarray(state.wrist_q_wxyz, dtype=np.float32).reshape(4) for state in hand_states], axis=0)
        return wrist_t_world, wrist_q_wxyz

    if wrist_pose_mode != "mapped_orca_root":
        raise ValueError(f"Unsupported wrist_pose_mode: {wrist_pose_mode}")

    joint_order = adapter_dst.joint_order()
    palm_link_name = adapter_dst.palm_link_name()
    wrist_t_world_all: list[np.ndarray] = []
    wrist_q_wxyz_all: list[np.ndarray] = []

    for hand_state, qpos_row, palm_delta_t, palm_delta_rotvec in zip(
        hand_states,
        qpos_sequence,
        palm_delta_t_sequence,
        palm_delta_rotvec_sequence,
        strict=True,
    ):
        ume_palm_local = adapter_src.semantic_palm_center_model(hand_state.joint_angles)

        ume_wrist_t = np.asarray(hand_state.wrist_t, dtype=np.float64).reshape(3)
        ume_wrist_rot = quat_wxyz_to_rotmat(np.asarray(hand_state.wrist_q_wxyz, dtype=np.float64).reshape(4))

        orca_palm_world_rot = _orthonormalize_rotmat(ume_wrist_rot @ np.asarray(rot_ume_to_orca, dtype=np.float64).T)
        orca_palm_world_t = ume_wrist_t + ume_wrist_rot @ ume_palm_local

        q_by_name = _build_q_by_name(joint_order, np.asarray(qpos_row, dtype=np.float64))
        link_poses = adapter_dst.kin.forward_link_poses(q_by_name)
        palm_T_root = np.asarray(link_poses[palm_link_name], dtype=np.float64)
        root_T_palm = adapter_dst.kin._invert_transform(palm_T_root)

        world_T_semantic_palm = np.eye(4, dtype=np.float64)
        world_T_semantic_palm[:3, :3] = orca_palm_world_rot
        world_T_semantic_palm[:3, 3] = orca_palm_world_t

        semantic_T_actual_palm = np.eye(4, dtype=np.float64)
        semantic_T_actual_palm[:3, :3] = rotvec_to_rotmat(np.asarray(palm_delta_rotvec, dtype=np.float64))
        semantic_T_actual_palm[:3, 3] = np.asarray(palm_delta_t, dtype=np.float64)

        world_T_palm = world_T_semantic_palm @ semantic_T_actual_palm
        world_T_root = world_T_palm @ root_T_palm

        wrist_t_world_all.append(world_T_root[:3, 3].astype(np.float32, copy=False))
        wrist_q_wxyz_all.append(_rotmat_to_quat_wxyz(world_T_root[:3, :3]))

    return (
        np.stack(wrist_t_world_all, axis=0).astype(np.float32, copy=False),
        np.stack(wrist_q_wxyz_all, axis=0).astype(np.float32, copy=False),
    )


def main() -> None:
    args = parse_args()
    if args.output_pkl is None and args.output_npz is None:
        raise ValueError("Provide at least one of --output-pkl or --output-npz")

    recording_dir = args.recording_dir.resolve()
    umetrack_jsonl = (recording_dir / args.umetrack_jsonl).resolve()
    umetrack_profile_json = (recording_dir / args.umetrack_profile_json).resolve()
    tip_json = args.umetrack_tip_map_json.resolve()
    orca_urdf = args.orca_urdf.resolve()

    adapter_src = UMETrackHandAdapter(
        profile_json=umetrack_profile_json,
        hand_side=args.hand_side,
        unit_scale=float(args.umetrack_unit_scale),
        mirror_left_x=bool(args.umetrack_left_mirror_x),
        mirror_right_x=bool(args.umetrack_right_mirror_x),
    )
    tip_idx_by_finger = load_tip_idx_by_finger(tip_json, args.hand_side)
    adapter_src.build_finger_map(tip_idx_by_finger)

    adapter_dst = ORCAHandAdapter(urdf_path=orca_urdf, side=args.orca_side)
    correspondences = adapter_dst.correspondences(args.target_set)
    body_names = [cp.link_name for cp in correspondences]
    feature_names = [cp.name for cp in correspondences]
    opt_joint_names = adapter_dst.joint_order()[1:]
    rest_pose_by_name = adapter_dst.rest_pose_by_name()
    joint_limit_overrides = parse_joint_limit_overrides(list(args.joint_limit_override))
    joint_limits_by_name = get_validated_joint_limits(adapter_dst, opt_joint_names, overrides=joint_limit_overrides)

    hand_traj_by_ts, hand_ids = load_umetrack_hand_trajectory(umetrack_jsonl)
    if args.hand_id not in hand_ids:
        raise ValueError(f"--hand-id={args.hand_id} not found. Available hand ids: {hand_ids}")

    raw_timestamps = sorted(ts for ts, hands in hand_traj_by_ts.items() if args.hand_id in hands)
    timestamps = filter_timestamps_by_range(
        raw_timestamps,
        start_ns=args.start_timestamp_ns,
        end_ns=args.end_timestamp_ns,
    )
    timestamps, consolidation_stats = consolidate_close_timestamps(
        timestamps,
        threshold_ns=int(args.consolidate_close_timestamps_ns),
    )
    timestamps = apply_timestamp_filters(
        timestamps,
        start_frame=int(args.start_frame),
        stride=args.stride,
        num_frames=int(args.num_frames),
        max_frames=args.max_frames,
    )
    if not timestamps:
        raise RuntimeError("No timestamps remain after filtering.")

    hand_states = [hand_traj_by_ts[ts][args.hand_id] for ts in timestamps]
    target_pos_sequence = []
    for hand_state in hand_states:
        targets = adapter_src.target_points_local(hand_state.joint_angles)
        target_pos_sequence.append(np.stack([targets[name] for name in feature_names], axis=0))
    target_pos_sequence = np.asarray(target_pos_sequence, dtype=np.float64)

    if args.fit_similarity:
        src_ref = target_pos_sequence[0]
        zero_q = {name: 0.0 for name in adapter_dst.joint_order()}
        dst_points = adapter_dst.points_in_palm(zero_q, correspondences)
        dst_ref = np.stack([dst_points[name] for name in feature_names], axis=0)
        if "palm_center" in feature_names:
            src_fit = src_ref[1:]
            dst_fit = dst_ref[1:]
        else:
            src_fit = src_ref
            dst_fit = dst_ref
        scale, rot = fit_similarity(src_fit, dst_fit, fit_scale=bool(args.fit_scale))
    else:
        scale = 1.0
        rot = (
            rot_z_deg(DEFAULT_UME_TO_ORCA_Z_OFFSET_DEG)
            @ rot_y_deg(DEFAULT_UME_TO_ORCA_Y_OFFSET_DEG)
            @ rot_x_deg(DEFAULT_UME_TO_ORCA_X_OFFSET_DEG)
        )
    target_pos_sequence = scale * np.einsum("tnd,dk->tnk", target_pos_sequence, rot.T, optimize=True)

    if args.filter_targets:
        target_pos_sequence = filter_position_sequence(
            np.asarray(target_pos_sequence, dtype=np.float64),
            wn=float(args.filter_wn),
            fs=float(args.filter_fs),
        )

    print(
        f"[INFO] URDF retargeting frames={len(timestamps)} hand_id={args.hand_id} "
        f"hand_side={args.hand_side} orca_side={args.orca_side}",
        flush=True,
    )
    if int(args.consolidate_close_timestamps_ns) > 0:
        print(
            f"[INFO] Timestamp consolidation: threshold_ns={int(args.consolidate_close_timestamps_ns)} "
            f"windowed_frames={consolidation_stats['input_count']} consolidated_frames={consolidation_stats['output_count']} "
            f"removed={consolidation_stats['removed_count']} max_group_size={consolidation_stats['max_group_size']}",
            flush=True,
        )
    print(f"[INFO] UME-TRACK tip map: {tip_idx_by_finger}", flush=True)
    print(f"[INFO] ORCA URDF: {orca_urdf}", flush=True)
    print(f"[INFO] Target set: {args.target_set}", flush=True)
    print(f"[INFO] Feature names: {feature_names}", flush=True)
    print(f"[INFO] Body names: {body_names}", flush=True)
    print(f"[INFO] ORCA export root link: {adapter_dst.urdf_root_link_name()}", flush=True)
    print(
        f"[INFO] Alignment: fit_similarity={bool(args.fit_similarity)} fit_scale={bool(args.fit_scale)} "
        f"resolved_scale={scale:.4f}",
        flush=True,
    )
    print(
        f"[INFO] Palm delta optimization: enabled={bool(args.optimize_palm_delta)} "
        f"pos_reg={float(args.palm_delta_pos_reg_weight):.3f} rot_reg={float(args.palm_delta_rot_reg_weight):.3f} "
        f"loss=soft_l1",
        flush=True,
    )
    if args.print_joint_limits:
        print("[INFO] Optimizer joint limits (radians):", flush=True)
        for name in opt_joint_names:
            lo, hi = joint_limits_by_name[name]
            print(f"  - {name}: [{lo:.5f}, {hi:.5f}]", flush=True)
    if joint_limit_overrides:
        print("[INFO] Applied joint-limit overrides:", flush=True)
        for name in sorted(joint_limit_overrides):
            lo, hi = joint_limit_overrides[name]
            print(f"  - {name}: [{lo:.5f}, {hi:.5f}]", flush=True)
    print("[INFO] Frame-0 ORCA initialization from MJCF rest pose:", flush=True)
    for name in opt_joint_names:
        q_init = float(rest_pose_by_name.get(name, 0.0))
        if abs(q_init) < 1e-8:
            continue
        print(f"  - {name}: {q_init:.5f}", flush=True)
    print(
        f"[INFO] Frame slicing: start_frame={int(args.start_frame)} stride={int(args.stride)} "
        f"num_frames={int(args.num_frames)} max_frames={int(args.max_frames)}",
        flush=True,
    )

    qpos_sequence, palm_delta_t_sequence, palm_delta_rotvec_sequence = retarget_sequence(
        target_pos_sequence=np.asarray(target_pos_sequence, dtype=np.float64),
        feature_names=feature_names,
        adapter_dst=adapter_dst,
        correspondences=correspondences,
        joint_limits_by_name=joint_limits_by_name,
        rest_pose_by_name=rest_pose_by_name,
        reg_weight=float(args.reg_weight),
        optimize_palm_delta=bool(args.optimize_palm_delta),
        palm_delta_pos_reg_weight=float(args.palm_delta_pos_reg_weight),
        palm_delta_rot_reg_weight=float(args.palm_delta_rot_reg_weight),
        max_nfev=int(args.max_nfev),
        verbose_every=int(args.verbose_every),
    )
    summarize_joint_saturation(
        qpos_sequence=qpos_sequence,
        joint_order=adapter_dst.joint_order(),
        joint_limits_by_name=joint_limits_by_name,
    )
    wrist_t_world, wrist_q_wxyz = compute_exported_wrist_pose(
        hand_states=hand_states,
        qpos_sequence=qpos_sequence,
        palm_delta_t_sequence=palm_delta_t_sequence,
        palm_delta_rotvec_sequence=palm_delta_rotvec_sequence,
        adapter_src=adapter_src,
        adapter_dst=adapter_dst,
        rot_ume_to_orca=rot,
        wrist_pose_mode=args.wrist_pose_mode,
    )

    if args.output_pkl is not None:
        args.output_pkl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_pkl.open("wb") as f:
            pickle.dump(qpos_sequence, f)

    if args.output_npz is not None:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            timestamp=np.asarray(timestamps, dtype=np.int64),
            qpos_hand=qpos_sequence,
            wrist_t_world=wrist_t_world,
            wrist_1_wxyz=wrist_q_wxyz,
            wrist_q_wxyz=wrist_q_wxyz,
            palm_delta_t=palm_delta_t_sequence,
            palm_delta_rotvec=palm_delta_rotvec_sequence,
            feature_names=np.asarray(feature_names, dtype=object),
            body_names=np.asarray(body_names, dtype=object),
            similarity_scale=np.asarray([scale], dtype=np.float64),
            similarity_rot=rot.astype(np.float64),
            wrist_pose_mode=np.asarray([args.wrist_pose_mode], dtype=object),
        )


if __name__ == "__main__":
    main()
