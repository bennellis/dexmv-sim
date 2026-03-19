from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class Correspondence:
    name: str
    link_name: str
    offset_xyz: np.ndarray
    weight: float


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis_xyz: np.ndarray
    limit_lower: float | None
    limit_upper: float | None


class OrcaKinematics:
    def __init__(self, urdf_path: Path) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"ORCA URDF not found: {self.urdf_path}")

        self._links: set[str] = set()
        self._joints: list[UrdfJoint] = []
        self._children_by_link: dict[str, list[UrdfJoint]] = {}
        self._root_link: str | None = None
        self._parse_urdf()

    @property
    def joints(self) -> list[UrdfJoint]:
        return self._joints

    def root_link_name(self) -> str:
        if self._root_link is None:
            raise RuntimeError("Kinematic tree is not initialized.")
        return self._root_link

    def child_joints(self, link_name: str) -> list[UrdfJoint]:
        return list(self._children_by_link.get(link_name, []))

    @staticmethod
    def _parse_xyz(text: str | None) -> np.ndarray:
        if text is None:
            return np.zeros(3, dtype=np.float64)
        values = [float(x) for x in text.split()]
        if len(values) != 3:
            raise ValueError(f"Expected 3 values, got {text!r}")
        return np.asarray(values, dtype=np.float64)

    def _parse_urdf(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        self._links = {link.attrib["name"] for link in root.findall("link")}

        child_links: set[str] = set()
        for joint_elem in root.findall("joint"):
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                continue

            origin_elem = joint_elem.find("origin")
            axis_elem = joint_elem.find("axis")
            limit_elem = joint_elem.find("limit")

            parent_link = parent_elem.attrib["link"]
            child_link = child_elem.attrib["link"]
            child_links.add(child_link)

            axis_xyz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if axis_elem is not None:
                axis_xyz = self._parse_xyz(axis_elem.attrib.get("xyz"))

            limit_lower = None
            limit_upper = None
            if limit_elem is not None:
                if "lower" in limit_elem.attrib:
                    limit_lower = float(limit_elem.attrib["lower"])
                if "upper" in limit_elem.attrib:
                    limit_upper = float(limit_elem.attrib["upper"])

            joint = UrdfJoint(
                name=joint_elem.attrib["name"],
                joint_type=joint_elem.attrib["type"],
                parent_link=parent_link,
                child_link=child_link,
                origin_xyz=self._parse_xyz(None if origin_elem is None else origin_elem.attrib.get("xyz")),
                origin_rpy=self._parse_xyz(None if origin_elem is None else origin_elem.attrib.get("rpy")),
                axis_xyz=axis_xyz,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            )
            self._joints.append(joint)
            self._children_by_link.setdefault(parent_link, []).append(joint)

        roots = [link for link in self._links if link not in child_links]
        if not roots:
            raise RuntimeError("Could not determine ORCA URDF root link.")
        self._root_link = sorted(roots)[0]

    @staticmethod
    def _rot_x(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    @staticmethod
    def _rot_y(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)

    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    @classmethod
    def _rpy_to_rotmat(cls, rpy_xyz: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = float(rpy_xyz[0]), float(rpy_xyz[1]), float(rpy_xyz[2])
        return cls._rot_z(yaw) @ cls._rot_y(pitch) @ cls._rot_x(roll)

    @classmethod
    def _axis_angle_to_rotmat(cls, axis_xyz: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(axis))
        if n < 1e-12 or abs(float(angle)) < 1e-12:
            return np.eye(3, dtype=np.float64)
        axis = axis / n
        x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        one_minus_c = 1.0 - c
        return np.asarray(
            [
                [x * x * one_minus_c + c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
                [y * x * one_minus_c + z * s, y * y * one_minus_c + c, y * z * one_minus_c - x * s],
                [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, z * z * one_minus_c + c],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _compose_transform(cls, xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = cls._rpy_to_rotmat(rpy)
        transform[:3, 3] = xyz
        return transform

    @staticmethod
    def _invert_transform(transform: np.ndarray) -> np.ndarray:
        inv = np.eye(4, dtype=np.float64)
        inv[:3, :3] = transform[:3, :3].T
        inv[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
        return inv

    def forward_link_poses(self, q_by_name: dict[str, float]) -> dict[str, np.ndarray]:
        if self._root_link is None:
            raise RuntimeError("Kinematic tree is not initialized.")

        poses: dict[str, np.ndarray] = {}

        def _dfs(link_name: str, world_T_link: np.ndarray) -> None:
            poses[link_name] = world_T_link
            for joint in self._children_by_link.get(link_name, []):
                joint_T = self._compose_transform(joint.origin_xyz, joint.origin_rpy)
                motion_T = np.eye(4, dtype=np.float64)
                if joint.joint_type in ("revolute", "continuous"):
                    theta = float(q_by_name.get(joint.name, 0.0))
                    motion_T[:3, :3] = self._axis_angle_to_rotmat(joint.axis_xyz, theta)
                elif joint.joint_type == "prismatic":
                    axis = np.asarray(joint.axis_xyz, dtype=np.float64)
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm > 1e-12:
                        axis = axis / axis_norm
                    motion_T[:3, 3] = axis * float(q_by_name.get(joint.name, 0.0))

                _dfs(joint.child_link, world_T_link @ joint_T @ motion_T)

        _dfs(self._root_link, np.eye(4, dtype=np.float64))
        return poses

    def point_in_palm_frame(
        self,
        q_by_name: dict[str, float],
        palm_link_name: str,
        link_name: str,
        point_xyz_in_link: np.ndarray,
    ) -> np.ndarray:
        poses = self.forward_link_poses(q_by_name)
        if palm_link_name not in poses:
            raise KeyError(f"Palm link {palm_link_name!r} not found in FK results.")
        if link_name not in poses:
            raise KeyError(f"Link {link_name!r} not found in FK results.")

        point_h = np.ones(4, dtype=np.float64)
        point_h[:3] = point_xyz_in_link
        palm_inv = self._invert_transform(poses[palm_link_name])
        point_palm = palm_inv @ poses[link_name] @ point_h
        return point_palm[:3]


class ORCAHandAdapter:
    def __init__(self, urdf_path: Path, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side
        self.urdf_path = Path(urdf_path).resolve()
        self.kin = OrcaKinematics(self.urdf_path)
        self._rest_pose_cache: dict[str, float] | None = None

    def joint_order(self) -> list[str]:
        prefix = self.side
        return [
            f"{prefix}_wrist",
            f"{prefix}_thumb_mcp",
            f"{prefix}_thumb_abd",
            f"{prefix}_thumb_pip",
            f"{prefix}_thumb_dip",
            f"{prefix}_index_abd",
            f"{prefix}_index_mcp",
            f"{prefix}_index_pip",
            f"{prefix}_middle_abd",
            f"{prefix}_middle_mcp",
            f"{prefix}_middle_pip",
            f"{prefix}_ring_abd",
            f"{prefix}_ring_mcp",
            f"{prefix}_ring_pip",
            f"{prefix}_pinky_abd",
            f"{prefix}_pinky_mcp",
            f"{prefix}_pinky_pip",
        ]

    def correspondences(self, target_set: str = "palm_mids") -> list[Correspondence]:
        prefix = self.side
        tip_sign = 1.0 if self.side == "left" else -1.0
        tip_offset = np.array([tip_sign * 0.009, 0.0, 0.035], dtype=np.float64)
        thumb_tip_offset = np.array([tip_sign * 0.003, 0.0, 0.025], dtype=np.float64)

        mids = [
            Correspondence("palm_center", f"{prefix}_palm", np.zeros(3, dtype=np.float64), 0.5),
            Correspondence("thumb_mid", f"{prefix}_thumb_ip", np.zeros(3, dtype=np.float64), 1.0),
            Correspondence("index_mid", f"{prefix}_index_ip", np.zeros(3, dtype=np.float64), 1.0),
            Correspondence("middle_mid", f"{prefix}_middle_ip", np.zeros(3, dtype=np.float64), 1.0),
            Correspondence("ring_mid", f"{prefix}_ring_ip", np.zeros(3, dtype=np.float64), 1.0),
            Correspondence("pinky_mid", f"{prefix}_pinky_ip", np.zeros(3, dtype=np.float64), 1.0),
        ]
        if target_set == "palm_mids":
            return mids
        if target_set == "palm_mids_tips":
            return mids + [
                Correspondence("thumb_tip", f"{prefix}_thumb_dp", thumb_tip_offset, 3.0),
                Correspondence("index_tip", f"{prefix}_index_ip", tip_offset, 3.0),
                Correspondence("middle_tip", f"{prefix}_middle_ip", tip_offset, 3.0),
                Correspondence("ring_tip", f"{prefix}_ring_ip", tip_offset, 3.0),
                Correspondence("pinky_tip", f"{prefix}_pinky_ip", tip_offset, 3.0),
            ]
        raise ValueError(f"Unsupported target_set: {target_set}")

    def palm_link_name(self) -> str:
        return f"{self.side}_palm"

    def urdf_root_link_name(self) -> str:
        return self.kin.root_link_name()

    def articulation_root_link_name(self) -> str:
        root_link = self.urdf_root_link_name()
        # Isaac's URDF importer drops the dummy top-level `world` link and writes
        # the articulation root pose onto the first articulated child link instead.
        if root_link == "world":
            child_joints = self.kin.child_joints(root_link)
            if len(child_joints) == 1 and child_joints[0].joint_type == "fixed":
                return child_joints[0].child_link
        return root_link

    def point_in_palm(self, q_by_name: dict[str, float], link_name: str, offset_xyz: np.ndarray) -> np.ndarray:
        return self.kin.point_in_palm_frame(
            q_by_name=q_by_name,
            palm_link_name=self.palm_link_name(),
            link_name=link_name,
            point_xyz_in_link=np.asarray(offset_xyz, dtype=np.float64),
        )

    def points_in_palm(
        self,
        q_by_name: dict[str, float],
        correspondences: list[Correspondence],
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for cp in correspondences:
            out[cp.name] = self.point_in_palm(q_by_name, cp.link_name, cp.offset_xyz)
        return out

    def joint_limits_by_name(self) -> dict[str, tuple[float, float]]:
        limits: dict[str, tuple[float, float]] = {}
        for joint in self.kin.joints:
            lo = -np.pi if joint.limit_lower is None else float(joint.limit_lower)
            hi = np.pi if joint.limit_upper is None else float(joint.limit_upper)
            if lo > hi:
                lo, hi = hi, lo
            limits[joint.name] = (lo, hi)
        return limits

    def rest_pose_by_name(self) -> dict[str, float]:
        if self._rest_pose_cache is None:
            self._rest_pose_cache = self._load_rest_pose_from_mjcf()
        return dict(self._rest_pose_cache)

    def _load_rest_pose_from_mjcf(self) -> dict[str, float]:
        mjcf_path = self._infer_matching_mjcf_path()
        root = ET.parse(mjcf_path).getroot()
        rest_pose = {joint_name: 0.0 for joint_name in self.joint_order()}

        for joint_elem in root.findall(".//joint"):
            name = joint_elem.attrib.get("name")
            if name not in rest_pose:
                continue
            ref_value = joint_elem.attrib.get("ref")
            rest_pose[name] = 0.0 if ref_value is None else float(ref_value)

        return rest_pose

    def _infer_matching_mjcf_path(self) -> Path:
        urdf_dir = self.urdf_path.parent
        mjcf_dir = urdf_dir.parent / "mjcf"
        mjcf_path = mjcf_dir / self.urdf_path.name.replace(".urdf", ".mjcf")
        if not mjcf_path.is_file():
            raise FileNotFoundError(
                f"Could not find matching ORCA MJCF for rest-pose lookup. "
                f"Expected: {mjcf_path}"
            )
        return mjcf_path
