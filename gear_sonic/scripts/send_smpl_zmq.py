#!/usr/bin/env python3
"""Publish a mock zero-SMPL Sonic mode-2 stream over ZMQ."""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import zmq

from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa
from gear_sonic.trl.utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

DEFAULT_TOPIC = "pose"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5556
DEFAULT_FPS = 50.0
DEFAULT_WINDOW = 10
G1_DOF = 29

G1_L_WRIST_ROLL_IDX = 23
G1_R_WRIST_ROLL_IDX = 24
G1_L_WRIST_PITCH_IDX = 25
G1_R_WRIST_PITCH_IDX = 26
G1_L_WRIST_YAW_IDX = 27
G1_R_WRIST_YAW_IDX = 28

SMPL_L_ELBOW_IDX = 17
SMPL_R_ELBOW_IDX = 18
SMPL_L_WRIST_IDX = 19
SMPL_R_WRIST_IDX = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300, help="Mock frame count")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Publish rate")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW, help="Frames per packet")
    parser.add_argument("--bind-host", type=str, default=DEFAULT_HOST, help="PUB bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="PUB bind port")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Topic name")
    parser.add_argument(
        "--protocol-version",
        type=int,
        default=2,
        choices=[2, 3],
        help="Header protocol version. Both map to Sonic encode mode 2.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop forever")
    parser.add_argument("--dry-run", action="store_true", help="Print shapes and exit")
    parser.add_argument("--connect", action="store_true", help="Connect PUB socket instead of binding")
    return parser.parse_args()


def build_mock_pose(frames: int) -> np.ndarray:
    pose = np.zeros((24, 3), dtype=np.float32)
    pose_72 = pose.reshape(72)
    return np.repeat(pose_72[None, :], frames, axis=0)


def compute_body_quat_and_smpl_joints(smpl_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_count = smpl_pose.shape[0]
    body_pose_t = torch.from_numpy(smpl_pose[:, 3:66].astype(np.float32))
    root_orient_t = torch.from_numpy(smpl_pose[:, :3].astype(np.float32))

    root_quat = angle_axis_to_quaternion(root_orient_t)
    root_quat = smpl_root_ytoz_up(root_quat)
    root_for_fk = quaternion_to_angle_axis(root_quat)

    smpl_joints_world = compute_human_joints(body_pose=body_pose_t, global_orient=root_for_fk)
    root_quat = remove_smpl_base_rot(root_quat, w_last=False)
    root_quat_inv = quat_inv(root_quat).unsqueeze(1).repeat(1, smpl_joints_world.shape[1], 1)
    smpl_joints_local = quat_apply(root_quat_inv, smpl_joints_world)

    return (
        smpl_joints_local.detach().cpu().numpy().astype(np.float32),
        root_quat.detach().cpu().numpy().astype(np.float32),
    )


def derive_wrist_joint_positions(smpl_pose: np.ndarray) -> np.ndarray:
    joint_pos = np.zeros((smpl_pose.shape[0], G1_DOF), dtype=np.float32)
    body_pose = smpl_pose[:, 3:66].reshape(-1, 21, 3)

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

    _, g1_l_elbow_q_swing = decompose_rotation_aa(
        smpl_l_elbow_aa, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    _, g1_r_elbow_q_swing = decompose_rotation_aa(
        smpl_r_elbow_aa, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    joint_pos[:, G1_L_WRIST_ROLL_IDX] = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    joint_pos[:, G1_L_WRIST_PITCH_IDX] = l_wrist_euler[:, 1]
    joint_pos[:, G1_L_WRIST_YAW_IDX] = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    joint_pos[:, G1_R_WRIST_ROLL_IDX] = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    joint_pos[:, G1_R_WRIST_PITCH_IDX] = -r_wrist_euler[:, 1]
    joint_pos[:, G1_R_WRIST_YAW_IDX] = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]
    return joint_pos


def build_mock_stream(frames: int) -> dict[str, np.ndarray]:
    pose_72 = build_mock_pose(frames)
    smpl_pose = pose_72[:, 3:66].reshape(frames, 21, 3).astype(np.float32)
    smpl_joints, body_quat_w = compute_body_quat_and_smpl_joints(pose_72)
    joint_pos = derive_wrist_joint_positions(pose_72)
    joint_vel = np.zeros((frames, G1_DOF), dtype=np.float32)
    frame_index = np.arange(frames, dtype=np.int64)
    return {
        "smpl_pose": smpl_pose,
        "smpl_joints": smpl_joints,
        "body_quat_w": body_quat_w,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "frame_index": frame_index,
    }


def publish_mock_stream(stream: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    endpoint = f"tcp://{args.bind_host}:{args.port}"
    interval = 1.0 / args.fps
    field_names = ("smpl_pose", "smpl_joints", "body_quat_w", "joint_pos", "joint_vel", "frame_index")

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    if args.connect:
        socket.connect(endpoint)
    else:
        socket.bind(endpoint)

    print(
        f"[send_smpl_zmq] mock mode-2 stream on {endpoint} "
        f"topic='{args.topic}' protocol={args.protocol_version} fps={args.fps:.3f} window={args.window_size}"
    )
    time.sleep(0.25)

    try:
        while True:
            buffers = {name: deque(maxlen=args.window_size) for name in field_names}
            for frame_idx in range(stream["frame_index"].shape[0]):
                for name in field_names:
                    buffers[name].append(stream[name][frame_idx])

                if len(buffers["frame_index"]) < args.window_size:
                    time.sleep(interval)
                    continue

                payload = {
                    "smpl_pose": np.stack(buffers["smpl_pose"], axis=0).astype(np.float32),
                    "smpl_joints": np.stack(buffers["smpl_joints"], axis=0).astype(np.float32),
                    "body_quat_w": np.stack(buffers["body_quat_w"], axis=0).astype(np.float32),
                    "joint_pos": np.stack(buffers["joint_pos"], axis=0).astype(np.float32),
                    "joint_vel": np.stack(buffers["joint_vel"], axis=0).astype(np.float32),
                    "frame_index": np.asarray(buffers["frame_index"], dtype=np.int64),
                }
                socket.send(
                    pack_pose_message(payload, topic=args.topic, version=args.protocol_version)
                )
                time.sleep(interval)

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[send_smpl_zmq] Interrupted")
    finally:
        socket.close(linger=0)


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.window_size <= 0:
        raise ValueError("--window-size must be positive")

    stream = build_mock_stream(args.frames)
    print(
        "[send_smpl_zmq] shapes: "
        f"smpl_pose={stream['smpl_pose'].shape}, "
        f"smpl_joints={stream['smpl_joints'].shape}, "
        f"body_quat_w={stream['body_quat_w'].shape}, "
        f"joint_pos={stream['joint_pos'].shape}"
    )

    if args.dry_run:
        return

    publish_mock_stream(stream, args)


if __name__ == "__main__":
    main()
