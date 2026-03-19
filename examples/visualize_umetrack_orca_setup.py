from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
DEXMV_DIR = SCRIPT_PATH.parents[1]
WORKSPACE_DIR = SCRIPT_PATH.parents[2]
for path in [DEXMV_DIR, WORKSPACE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexterous_manipulation.utils.hot3d_data import load_umetrack_hand_trajectory
from hand_imitation.misc.orca_adapter import ORCAHandAdapter
from hand_imitation.misc.umetrack_adapter import FINGER_NAMES, UMETrackHandAdapter


DEFAULT_UMETRACK_TIP_MAP_JSON = (
    WORKSPACE_DIR / "dexterous_manipulation" / "scripts" / "umetrack_tip_indices.json"
)
DEFAULT_ORCA_URDF = WORKSPACE_DIR / "orcahand_description" / "models" / "urdf" / "orcahand_right_extended.urdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Visualize UME-TRACK joint indices and ORCA correspondence points")
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
    parser.add_argument("--target-set", choices=("palm_mids", "palm_mids_tips"), default="palm_mids_tips")
    parser.add_argument("--timestamp-ns", type=int, default=None)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--retarget-npz", type=Path, default=None)
    parser.add_argument("--show-joint-labels", action="store_true", default=False)
    parser.add_argument("--show-link-labels", action="store_true", default=False)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true", default=False)
    return parser.parse_args()


def load_tip_idx_by_finger(path: Path, hand_side: str) -> dict[str, int]:
    with path.resolve().open("r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("tip_idx_by_finger", {})
    if hand_side in raw:
        return {k: int(v) for k, v in raw[hand_side].items()}
    return {k: int(v) for k, v in raw.items()}


def pick_timestamp(available_timestamps: list[int], requested_timestamp_ns: int | None, frame_index: int) -> int:
    if not available_timestamps:
        raise RuntimeError("No timestamps available.")
    if requested_timestamp_ns is not None:
        arr = np.asarray(available_timestamps, dtype=np.int64)
        return int(arr[int(np.argmin(np.abs(arr - int(requested_timestamp_ns))))])
    if frame_index < 0 or frame_index >= len(available_timestamps):
        raise ValueError(f"--frame-index={frame_index} out of range [0, {len(available_timestamps) - 1}]")
    return int(available_timestamps[frame_index])


def fit_similarity(src_points: np.ndarray, dst_points: np.ndarray) -> tuple[float, np.ndarray]:
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

    denom = float(np.sum(src0 * src0))
    scale = 1.0 if denom < 1e-12 else float(np.sum(singular_values) / denom)
    return scale, rot


def load_retarget_row(npz_path: Path, ts_pick: int) -> np.ndarray | None:
    if npz_path is None:
        return None
    with np.load(npz_path.resolve(), allow_pickle=True) as data:
        timestamps = np.asarray(data["timestamp"], dtype=np.int64)
        qpos_hand = np.asarray(data["qpos_hand"], dtype=np.float64)
    if timestamps.ndim != 1 or qpos_hand.ndim != 2 or qpos_hand.shape[0] != timestamps.shape[0]:
        raise ValueError(f"Invalid retarget NPZ shapes: timestamp={timestamps.shape}, qpos_hand={qpos_hand.shape}")
    row_idx = int(np.argmin(np.abs(timestamps - int(ts_pick))))
    return qpos_hand[row_idx]


def build_q_by_name(joint_order: list[str], qpos_row: np.ndarray | None) -> dict[str, float]:
    if qpos_row is None:
        return {name: 0.0 for name in joint_order}
    q_by_name: dict[str, float] = {}
    count = min(len(joint_order), int(qpos_row.shape[0]))
    for idx in range(count):
        q_by_name[joint_order[idx]] = float(qpos_row[idx])
    for idx in range(count, len(joint_order)):
        q_by_name[joint_order[idx]] = 0.0
    return q_by_name


def orca_link_points_in_palm(adapter: ORCAHandAdapter, q_by_name: dict[str, float]) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    links = sorted({joint.parent_link for joint in adapter.kin.joints}.union({joint.child_link for joint in adapter.kin.joints}))
    link_points: dict[str, np.ndarray] = {}
    for link_name in links:
        try:
            link_points[link_name] = adapter.point_in_palm(q_by_name, link_name, np.zeros(3, dtype=np.float64))
        except KeyError:
            continue

    edges: list[tuple[str, str]] = []
    for joint in adapter.kin.joints:
        if joint.parent_link in link_points and joint.child_link in link_points:
            edges.append((joint.parent_link, joint.child_link))
    return link_points, edges


def set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = [abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0]), 1e-6]
    radius = 0.5 * max(ranges)
    x_mid = float(np.mean(x_limits))
    y_mid = float(np.mean(y_limits))
    z_mid = float(np.mean(z_limits))
    ax.set_xlim3d([x_mid - radius, x_mid + radius])
    ax.set_ylim3d([y_mid - radius, y_mid + radius])
    ax.set_zlim3d([z_mid - radius, z_mid + radius])


def finger_color(name: str) -> tuple[float, float, float]:
    colors = {
        "thumb": (0.95, 0.45, 0.20),
        "index": (0.20, 0.75, 0.95),
        "middle": (0.20, 0.90, 0.45),
        "ring": (0.95, 0.80, 0.20),
        "pinky": (0.88, 0.35, 0.88),
        "palm": (0.90, 0.90, 0.90),
    }
    return colors.get(name, (0.7, 0.7, 0.7))


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    recording_dir = args.recording_dir.resolve()
    umetrack_jsonl = (recording_dir / args.umetrack_jsonl).resolve()
    umetrack_profile_json = (recording_dir / args.umetrack_profile_json).resolve()

    adapter_src = UMETrackHandAdapter(
        profile_json=umetrack_profile_json,
        hand_side=args.hand_side,
        unit_scale=float(args.umetrack_unit_scale),
        mirror_left_x=bool(args.umetrack_left_mirror_x),
        mirror_right_x=bool(args.umetrack_right_mirror_x),
    )
    tip_map = load_tip_idx_by_finger(args.umetrack_tip_map_json, args.hand_side)
    finger_map = adapter_src.build_finger_map(tip_map)

    traj_by_ts, hand_ids = load_umetrack_hand_trajectory(umetrack_jsonl)
    if args.hand_id not in hand_ids:
        raise ValueError(f"--hand-id={args.hand_id} not found. Available: {hand_ids}")
    timestamps = sorted(ts for ts, hands in traj_by_ts.items() if args.hand_id in hands)
    ts_pick = pick_timestamp(timestamps, args.timestamp_ns, args.frame_index)
    hand_state = traj_by_ts[ts_pick][args.hand_id]

    joint_positions = adapter_src.joint_positions_local(hand_state.joint_angles, center_at_palm=True)
    target_points = adapter_src.target_points_local(hand_state.joint_angles)

    adapter_dst = ORCAHandAdapter(args.orca_urdf.resolve(), side=args.orca_side)
    correspondences = adapter_dst.correspondences(args.target_set)
    feature_names = [cp.name for cp in correspondences]
    target_stack = np.stack([target_points[name] for name in feature_names], axis=0)

    qpos_row = load_retarget_row(args.retarget_npz, ts_pick) if args.retarget_npz is not None else None
    q_by_name = build_q_by_name(adapter_dst.joint_order(), qpos_row)
    corr_points = adapter_dst.points_in_palm(q_by_name, correspondences)
    corr_stack = np.stack([corr_points[name] for name in feature_names], axis=0)

    if "palm_center" in feature_names:
        src_fit = target_stack[1:]
        dst_fit = corr_stack[1:]
    else:
        src_fit = target_stack
        dst_fit = corr_stack
    scale, rot = fit_similarity(src_fit, dst_fit)
    aligned_targets = scale * (target_stack @ rot.T)

    link_points, edges = orca_link_points_in_palm(adapter_dst, q_by_name)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    ax1.set_title(f"UME Local Joints\nframe ts={ts_pick}")
    ax2.set_title("ORCA Correspondence Points")
    ax3.set_title("Aligned UME Targets vs ORCA")

    parents = np.asarray(adapter_src.model.joint_parent, dtype=np.int32).reshape(-1)
    for joint_idx, parent_idx in enumerate(parents.tolist()):
        if parent_idx == 255:
            continue
        xs = [joint_positions[parent_idx, 0], joint_positions[joint_idx, 0]]
        ys = [joint_positions[parent_idx, 1], joint_positions[joint_idx, 1]]
        zs = [joint_positions[parent_idx, 2], joint_positions[joint_idx, 2]]
        ax1.plot(xs, ys, zs, color=(0.5, 0.5, 0.5), linewidth=1.0)

    ax1.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], c="k", s=16)
    if args.show_joint_labels:
        for joint_idx, point in enumerate(joint_positions):
            ax1.text(point[0], point[1], point[2], str(joint_idx), fontsize=7)

    for finger in FINGER_NAMES:
        mid_idx = finger_map.mid_idx_by_finger[finger]
        tip_idx = finger_map.tip_idx_by_finger[finger]
        color = finger_color(finger)
        mid_name = f"{finger}_mid"
        tip_name = f"{finger}_tip"
        mid_point = np.asarray(target_points[mid_name], dtype=np.float64)
        tip_point = np.asarray(target_points[tip_name], dtype=np.float64)

        ax1.scatter(
            [mid_point[0], tip_point[0]],
            [mid_point[1], tip_point[1]],
            [mid_point[2], tip_point[2]],
            c=[color, color],
            s=[40, 52],
            marker="o",
        )
        ax1.text(mid_point[0], mid_point[1], mid_point[2], f"{finger}:mid", fontsize=8, color=color)
        ax1.text(tip_point[0], tip_point[1], tip_point[2], f"{finger}:tip", fontsize=8, color=color)
        ax1.plot(
            [joint_positions[mid_idx, 0], mid_point[0]],
            [joint_positions[mid_idx, 1], mid_point[1]],
            [joint_positions[mid_idx, 2], mid_point[2]],
            color=color,
            linewidth=0.8,
            alpha=0.6,
        )
        ax1.plot(
            [joint_positions[tip_idx, 0], tip_point[0]],
            [joint_positions[tip_idx, 1], tip_point[1]],
            [joint_positions[tip_idx, 2], tip_point[2]],
            color=color,
            linewidth=0.8,
            alpha=0.6,
        )

    for parent_link, child_link in edges:
        p0 = link_points[parent_link]
        p1 = link_points[child_link]
        ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=(0.7, 0.7, 0.7), linewidth=1.0)

    for cp in correspondences:
        point = corr_points[cp.name]
        finger_name = "palm" if cp.name == "palm_center" else cp.name.split("_")[0]
        color = finger_color(finger_name)
        ax2.scatter([point[0]], [point[1]], [point[2]], c=[color], s=50)
        ax2.text(point[0], point[1], point[2], cp.name, fontsize=8)

    if args.show_link_labels:
        for link_name, point in link_points.items():
            ax2.text(point[0], point[1], point[2], link_name, fontsize=6)

    for parent_link, child_link in edges:
        p0 = link_points[parent_link]
        p1 = link_points[child_link]
        ax3.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=(0.75, 0.75, 0.75), linewidth=1.0)

    for cp, src_point, dst_point in zip(correspondences, aligned_targets, corr_stack):
        finger_name = "palm" if cp.name == "palm_center" else cp.name.split("_")[0]
        color = finger_color(finger_name)
        ax3.scatter([src_point[0]], [src_point[1]], [src_point[2]], c=[color], marker="o", s=42)
        ax3.scatter([dst_point[0]], [dst_point[1]], [dst_point[2]], c=[color], marker="^", s=52)
        ax3.plot([src_point[0], dst_point[0]], [src_point[1], dst_point[1]], [src_point[2], dst_point[2]], color=color, linewidth=1.0)
        ax3.text(dst_point[0], dst_point[1], dst_point[2], cp.name, fontsize=8)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        set_axes_equal(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="UME target (aligned)", markerfacecolor="black", markersize=7),
        Line2D([0], [0], marker="^", color="w", label="ORCA correspondence", markerfacecolor="black", markersize=8),
        Line2D([0], [0], color="black", lw=1.0, label="Residual"),
    ]
    ax3.legend(handles=legend_handles, loc="upper right")

    fig.suptitle(
        f"UME tip map={tip_map} | similarity scale={scale:.4f} | "
        f"mode={'retargeted qpos' if qpos_row is not None else 'zero ORCA qpos'}",
        fontsize=11,
    )
    fig.tight_layout()

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=180)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
