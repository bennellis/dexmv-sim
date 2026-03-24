from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


WORKSPACE_DIR = Path(__file__).resolve().parents[3]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from dexterous_manipulation.utils.hot3d_data import load_umetrack_profile
from dexterous_manipulation.utils.hot3d_models import UmeTrackHandModel


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
UME_PALM_CENTER_OFFSET_LOCAL = np.array([0.02, 0.0, 0.01], dtype=np.float64)


@dataclass(frozen=True)
class UMETrackFingerMap:
    root_idx: int
    tip_idx_by_finger: dict[str, int]
    mid_idx_by_finger: dict[str, int]
    palm_base_indices: list[int]


class UMETrackHandAdapter:
    def __init__(
        self,
        profile_json: Path,
        hand_side: str,
        unit_scale: float = 0.001,
        mirror_left_x: bool = False,
        mirror_right_x: bool = True,
    ) -> None:
        if hand_side not in ("left", "right"):
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

        self.hand_side = hand_side
        self.mirror_left_x = bool(mirror_left_x)
        self.mirror_right_x = bool(mirror_right_x)
        self.profile_json = Path(profile_json).resolve()

        profile = load_umetrack_profile(self.profile_json)
        self.model = UmeTrackHandModel(profile, unit_scale=float(unit_scale))
        self.finger_map: UMETrackFingerMap | None = None

    @staticmethod
    def _validate_tip_idx_map(mapping: dict[str, int], num_joints: int) -> dict[str, int]:
        keys = set(mapping.keys())
        expected = set(FINGER_NAMES)
        if keys != expected:
            raise ValueError(f"tip_idx_by_finger keys must be {sorted(expected)}, got {sorted(keys)}")

        out: dict[str, int] = {}
        for finger in FINGER_NAMES:
            idx = int(mapping[finger])
            if idx < 0 or idx >= num_joints:
                raise ValueError(f"Joint index for '{finger}' out of range [0, {num_joints - 1}], got {idx}")
            out[finger] = idx
        return out

    @staticmethod
    def _ancestry(parents: np.ndarray, joint_idx: int) -> list[int]:
        chain = [int(joint_idx)]
        current = int(joint_idx)
        while True:
            parent = int(parents[current])
            if parent == 255:
                break
            chain.append(parent)
            current = parent
        return chain

    def build_finger_map(self, tip_idx_by_finger: dict[str, int]) -> UMETrackFingerMap:
        parents = np.asarray(self.model.joint_parent, dtype=np.int32).reshape(-1)
        num_joints = int(parents.shape[0])

        root_candidates = np.where(parents == 255)[0]
        root_idx = int(root_candidates[0]) if root_candidates.size > 0 else 0

        tip_map = self._validate_tip_idx_map(tip_idx_by_finger, num_joints=num_joints)

        mid_idx_by_finger: dict[str, int] = {}
        palm_base_indices: list[int] = []
        for finger in FINGER_NAMES:
            chain = self._ancestry(parents, tip_map[finger])
            # UME-TRACK finger chains are longer than the MANO-derived setup DexMV used.
            # Use the immediate parent of the tip joint as the semantic "mid" target.
            mid_idx_by_finger[finger] = int(chain[min(1, len(chain) - 1)])
            base_idx = int(chain[-2]) if len(chain) >= 2 else int(chain[-1])
            palm_base_indices.append(base_idx)

        self.finger_map = UMETrackFingerMap(
            root_idx=root_idx,
            tip_idx_by_finger=tip_map,
            mid_idx_by_finger=mid_idx_by_finger,
            palm_base_indices=sorted(set(palm_base_indices)),
        )
        return self.finger_map

    def _require_finger_map(self) -> UMETrackFingerMap:
        if self.finger_map is None:
            raise RuntimeError("Finger map is not initialized. Call build_finger_map() first.")
        return self.finger_map

    def _apply_mirror_to_points(self, points_xyz: np.ndarray) -> np.ndarray:
        out = np.asarray(points_xyz, dtype=np.float64).copy()
        if self.hand_side == "left" and self.mirror_left_x:
            out[:, 0] *= -1.0
        if self.hand_side == "right" and self.mirror_right_x:
            out[:, 0] *= -1.0
        return out

    def _apply_mirror_to_frames(self, frames: np.ndarray) -> np.ndarray:
        out = np.asarray(frames, dtype=np.float64).copy()
        do_mirror = (self.hand_side == "left" and self.mirror_left_x) or (
            self.hand_side == "right" and self.mirror_right_x
        )
        if not do_mirror:
            return out

        mirror = np.eye(4, dtype=np.float64)
        mirror[0, 0] = -1.0
        return mirror[None, ...] @ out @ mirror[None, ...]

    def _raw_palm_center_from_local(self, local_joint_positions: np.ndarray) -> np.ndarray:
        finger_map = self._require_finger_map()
        return np.mean(np.asarray(local_joint_positions, dtype=np.float64)[finger_map.palm_base_indices], axis=0)

    def _semantic_palm_center_from_local(self, local_joint_positions: np.ndarray) -> np.ndarray:
        return self._raw_palm_center_from_local(local_joint_positions) + UME_PALM_CENTER_OFFSET_LOCAL

    def semantic_palm_center_local(self, joint_angles: np.ndarray) -> np.ndarray:
        local = self.joint_positions_local(joint_angles, center_at_palm=False)
        return self._semantic_palm_center_from_local(local)

    def semantic_palm_center_model(self, joint_angles: np.ndarray) -> np.ndarray:
        joint_xf = self.model._compute_joint_global(joint_angles)
        joint_pos = np.asarray(joint_xf[:, :3, 3], dtype=np.float64)
        joint_pos = self._apply_mirror_to_points(joint_pos)
        return self._semantic_palm_center_from_local(joint_pos)

    def joint_positions_local(self, joint_angles: np.ndarray, center_at_palm: bool = False) -> np.ndarray:
        finger_map = self._require_finger_map()
        joint_xf = self.model._compute_joint_global(joint_angles)
        joint_pos = np.asarray(joint_xf[:, :3, 3], dtype=np.float64)

        root_pos = joint_pos[finger_map.root_idx]
        local = joint_pos - root_pos.reshape(1, 3)
        local = self._apply_mirror_to_points(local)

        if center_at_palm:
            palm_center = self._semantic_palm_center_from_local(local)
            local = local - palm_center.reshape(1, 3)
        return local

    def joint_frames_local(self, joint_angles: np.ndarray, center_at_palm: bool = False) -> np.ndarray:
        finger_map = self._require_finger_map()
        joint_xf = np.asarray(self.model._compute_joint_global(joint_angles), dtype=np.float64)

        root_T = joint_xf[finger_map.root_idx]
        root_inv = np.eye(4, dtype=np.float64)
        root_inv[:3, :3] = root_T[:3, :3].T
        root_inv[:3, 3] = -root_T[:3, :3].T @ root_T[:3, 3]
        local = root_inv[None, ...] @ joint_xf
        local = self._apply_mirror_to_frames(local)

        if center_at_palm:
            palm_center = self._semantic_palm_center_from_local(local[:, :3, 3])
            local[:, :3, 3] -= palm_center.reshape(1, 3)
        return local

    def mesh_vertices_local(self, joint_angles: np.ndarray, center_at_palm: bool = False) -> np.ndarray:
        finger_map = self._require_finger_map()
        joint_xf = self.model._compute_joint_global(joint_angles)
        joint_pos = np.asarray(joint_xf[:, :3, 3], dtype=np.float64)
        verts = np.asarray(self.model.deform(joint_angles), dtype=np.float64)

        root_pos = joint_pos[finger_map.root_idx]
        joint_local = joint_pos - root_pos.reshape(1, 3)
        verts_local = verts - root_pos.reshape(1, 3)

        joint_local = self._apply_mirror_to_points(joint_local)
        verts_local = self._apply_mirror_to_points(verts_local)

        if center_at_palm:
            palm_center = self._semantic_palm_center_from_local(joint_local)
            verts_local = verts_local - palm_center.reshape(1, 3)
        return verts_local

    @staticmethod
    def _closest_point_on_triangle(point_xyz: np.ndarray, tri_xyz: np.ndarray) -> np.ndarray:
        a_pt = tri_xyz[0]
        b_pt = tri_xyz[1]
        c_pt = tri_xyz[2]
        ab_vec = b_pt - a_pt
        ac_vec = c_pt - a_pt
        ap_vec = point_xyz - a_pt

        d1 = float(np.dot(ab_vec, ap_vec))
        d2 = float(np.dot(ac_vec, ap_vec))
        if d1 <= 0.0 and d2 <= 0.0:
            return a_pt

        bp_vec = point_xyz - b_pt
        d3 = float(np.dot(ab_vec, bp_vec))
        d4 = float(np.dot(ac_vec, bp_vec))
        if d3 >= 0.0 and d4 <= d3:
            return b_pt

        vc_val = d1 * d4 - d3 * d2
        if vc_val <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v_val = d1 / (d1 - d3)
            return a_pt + v_val * ab_vec

        cp_vec = point_xyz - c_pt
        d5 = float(np.dot(ab_vec, cp_vec))
        d6 = float(np.dot(ac_vec, cp_vec))
        if d6 >= 0.0 and d5 <= d6:
            return c_pt

        vb_val = d5 * d2 - d1 * d6
        if vb_val <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w_val = d2 / (d2 - d6)
            return a_pt + w_val * ac_vec

        va_val = d3 * d6 - d5 * d4
        if va_val <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            bc_vec = c_pt - b_pt
            w_val = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b_pt + w_val * bc_vec

        denom = 1.0 / (va_val + vb_val + vc_val)
        v_val = vb_val * denom
        w_val = vc_val * denom
        return a_pt + ab_vec * v_val + ac_vec * w_val

    @staticmethod
    def _snap_point_to_mesh_surface(
        mesh_vertices_local: np.ndarray,
        mesh_faces: np.ndarray,
        desired_point_local: np.ndarray,
        candidate_vertex_mask: np.ndarray,
    ) -> np.ndarray:
        vertex_ids = np.flatnonzero(candidate_vertex_mask)
        if vertex_ids.size > 0:
            face_mask = np.any(np.isin(mesh_faces, vertex_ids), axis=1)
            candidate_faces = np.asarray(mesh_faces[face_mask], dtype=np.int64)
        else:
            candidate_faces = np.asarray(mesh_faces, dtype=np.int64)

        if candidate_faces.shape[0] == 0:
            distances = np.linalg.norm(
                np.asarray(mesh_vertices_local, dtype=np.float64) - np.asarray(desired_point_local, dtype=np.float64).reshape(1, 3),
                axis=1,
            )
            return np.asarray(mesh_vertices_local[int(np.argmin(distances))], dtype=np.float64)

        best_point = None
        best_dist = np.inf
        for tri_idx in candidate_faces:
            tri = np.asarray(mesh_vertices_local[tri_idx], dtype=np.float64)
            candidate = UMETrackHandAdapter._closest_point_on_triangle(np.asarray(desired_point_local, dtype=np.float64), tri)
            dist = float(np.linalg.norm(candidate - np.asarray(desired_point_local, dtype=np.float64)))
            if dist < best_dist:
                best_dist = dist
                best_point = candidate

        if best_point is None:
            return np.asarray(desired_point_local, dtype=np.float64)
        return np.asarray(best_point, dtype=np.float64)

    @staticmethod
    def _estimate_tip_point_from_mesh(
        mesh_vertices_local: np.ndarray,
        mesh_faces: np.ndarray,
        mid_point_local: np.ndarray,
        tip_joint_local: np.ndarray,
        palm_center_local: np.ndarray,
        finger_name: str,
    ) -> np.ndarray:
        distal = np.asarray(tip_joint_local, dtype=np.float64) - np.asarray(mid_point_local, dtype=np.float64)
        seg_len = float(np.linalg.norm(distal))
        if seg_len < 1e-8 or mesh_vertices_local.shape[0] == 0:
            return np.asarray(tip_joint_local, dtype=np.float64)

        distal_dir = distal / seg_len
        palm_vec = np.asarray(palm_center_local, dtype=np.float64) - np.asarray(tip_joint_local, dtype=np.float64)
        pad_dir = palm_vec - float(np.dot(palm_vec, distal_dir)) * distal_dir
        pad_norm = float(np.linalg.norm(pad_dir))
        if pad_norm > 1e-8:
            pad_dir = pad_dir / pad_norm
        else:
            pad_dir = np.zeros(3, dtype=np.float64)
        rel = np.asarray(mesh_vertices_local, dtype=np.float64) - np.asarray(tip_joint_local, dtype=np.float64).reshape(1, 3)
        proj = rel @ distal_dir
        ortho = np.linalg.norm(rel - proj[:, None] * distal_dir.reshape(1, 3), axis=1)
        dist_to_tip = np.linalg.norm(rel, axis=1)
        palmar_proj = rel @ pad_dir if pad_norm > 1e-8 else np.zeros((rel.shape[0],), dtype=np.float64)

        radial_thresh = max(0.40 * seg_len, 0.007) if finger_name == "thumb" else max(0.55 * seg_len, 0.008)
        backward_thresh = -0.05 * seg_len if finger_name == "thumb" else -0.20 * seg_len
        forward_thresh = 0.75 * seg_len if finger_name == "thumb" else 1.35 * seg_len
        candidate_mask = (proj >= backward_thresh) & (proj <= forward_thresh) & (ortho <= radial_thresh)
        if finger_name == "thumb":
            thumb_local_radius = max(0.80 * seg_len, 0.018)
            candidate_mask &= dist_to_tip <= thumb_local_radius

        tip_surface_point: np.ndarray | None = None
        if np.any(candidate_mask):
            candidate_idx = np.where(candidate_mask)[0]
            score = proj[candidate_idx] - 0.35 * ortho[candidate_idx] - 0.25 * dist_to_tip[candidate_idx]
            best_idx = int(candidate_idx[int(np.argmax(score))])
            if float(proj[best_idx]) > 0.10 * seg_len:
                tip_surface_point = np.asarray(mesh_vertices_local[best_idx], dtype=np.float64)

        if tip_surface_point is None:
            default_extension = 0.22 * seg_len if finger_name == "thumb" else 0.55 * seg_len
            tip_surface_point = np.asarray(tip_joint_local, dtype=np.float64) + default_extension * distal_dir

        # The geometric fingertip sits too far distal for contact-style landmarks.
        # Pull the point back toward the distal pad center while keeping the local surface offset.
        tip_rel = tip_surface_point - np.asarray(tip_joint_local, dtype=np.float64)
        forward = float(np.dot(tip_rel, distal_dir))
        lateral = tip_rel - forward * distal_dir
        pad_backoff = 0.18 * seg_len if finger_name == "thumb" else 0.28 * seg_len
        min_forward = 0.10 * seg_len if finger_name == "thumb" else 0.14 * seg_len
        pad_forward = max(min_forward, forward - pad_backoff)
        if pad_norm > 1e-8:
            pad_extent = 0.0
            if np.any(candidate_mask):
                pad_extent = float(np.max(np.clip(palmar_proj[candidate_mask], a_min=0.0, a_max=None)))
            if pad_extent <= 1e-8:
                pad_extent = 0.18 * seg_len if finger_name == "thumb" else 0.24 * seg_len
            pad_component = max(0.55 * pad_extent, 0.10 * seg_len)
            lateral_side = lateral - float(np.dot(lateral, pad_dir)) * pad_dir
            side_scale = 0.20
            desired_point = (
                np.asarray(tip_joint_local, dtype=np.float64)
                + pad_forward * distal_dir
                + pad_component * pad_dir
                + side_scale * lateral_side
            )
        else:
            desired_point = np.asarray(tip_joint_local, dtype=np.float64) + pad_forward * distal_dir + lateral

        snap_rel = np.asarray(mesh_vertices_local, dtype=np.float64) - np.asarray(tip_joint_local, dtype=np.float64).reshape(1, 3)
        snap_proj = snap_rel @ distal_dir
        snap_ortho = np.linalg.norm(
            snap_rel - snap_proj[:, None] * distal_dir.reshape(1, 3),
            axis=1,
        )
        snap_mask = (
            (snap_proj >= -0.10 * seg_len)
            & (snap_proj <= (0.80 * seg_len if finger_name == "thumb" else 1.10 * seg_len))
            & (snap_ortho <= (max(0.40 * seg_len, 0.007) if finger_name == "thumb" else max(0.55 * seg_len, 0.008)))
        )
        if pad_norm > 1e-8:
            snap_palmar = snap_rel @ pad_dir
            snap_mask &= snap_palmar >= -0.10 * seg_len
        if finger_name == "thumb":
            snap_mask &= np.linalg.norm(snap_rel, axis=1) <= max(0.85 * seg_len, 0.02)

        return UMETrackHandAdapter._snap_point_to_mesh_surface(
            mesh_vertices_local=mesh_vertices_local,
            mesh_faces=np.asarray(mesh_faces, dtype=np.int64),
            desired_point_local=desired_point,
            candidate_vertex_mask=snap_mask,
        )

    def _mesh_tip_points_local(self, joint_angles: np.ndarray, joint_local: np.ndarray) -> dict[str, np.ndarray]:
        finger_map = self._require_finger_map()
        mesh_local = self.mesh_vertices_local(joint_angles, center_at_palm=False)
        palm_center = self._semantic_palm_center_from_local(joint_local)
        out: dict[str, np.ndarray] = {}
        for finger in FINGER_NAMES:
            mid_point = joint_local[finger_map.mid_idx_by_finger[finger]]
            tip_joint = joint_local[finger_map.tip_idx_by_finger[finger]]
            out[finger] = self._estimate_tip_point_from_mesh(
                mesh_local,
                self.model.faces,
                mid_point,
                tip_joint,
                palm_center,
                finger,
            )
        return out

    def target_points_local(self, joint_angles: np.ndarray) -> dict[str, np.ndarray]:
        finger_map = self._require_finger_map()
        local = self.joint_positions_local(joint_angles, center_at_palm=False)
        palm_center = self._semantic_palm_center_from_local(local)
        mesh_tip_points = self._mesh_tip_points_local(joint_angles, joint_local=local)

        targets = {
            "palm_center": palm_center,
            "thumb_mid": local[finger_map.mid_idx_by_finger["thumb"]],
            "thumb_tip": mesh_tip_points["thumb"],
            "index_mid": local[finger_map.mid_idx_by_finger["index"]],
            "index_tip": mesh_tip_points["index"],
            "middle_mid": local[finger_map.mid_idx_by_finger["middle"]],
            "middle_tip": mesh_tip_points["middle"],
            "ring_mid": local[finger_map.mid_idx_by_finger["ring"]],
            "ring_tip": mesh_tip_points["ring"],
            "pinky_mid": local[finger_map.mid_idx_by_finger["pinky"]],
            "pinky_tip": mesh_tip_points["pinky"],
        }

        centered = {name: (value - palm_center).astype(np.float64, copy=False) for name, value in targets.items()}
        centered["palm_center"] = np.zeros(3, dtype=np.float64)
        return centered
