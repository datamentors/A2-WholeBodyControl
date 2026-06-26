from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

FINGER_STIFFNESS = 20.0
FINGER_DAMPING = 1.0
FINGER_ARMATURE = 0.001

# Root body + actuated child links in the URDF order. This is the body order
# SONIC expects after slicing the full 74-body MJCF down to the actuated subset.
A2_ISAACLAB_JOINTS = [
    "base_link",
    "left_hip_roll",
    "left_hip_yaw",
    "left_hip_pitch",
    "left_tarsus",
    "left_toe_pitch",
    "left_toe_roll",
    "right_hip_roll",
    "right_hip_yaw",
    "right_hip_pitch",
    "right_tarsus",
    "right_toe_pitch",
    "right_toe_roll",
    "left_arm_link01",
    "left_arm_link02",
    "left_arm_link03",
    "left_arm_link04",
    "left_arm_link05",
    "left_arm_link06",
    "left_arm_link07",
    "L_thumb_swing",
    "L_thumb_1",
    "L_thumb_2",
    "L_thumb_3",
    "L_index_1",
    "L_index_2",
    "L_middle_1",
    "L_middle_2",
    "L_ring_1",
    "L_ring_2",
    "L_pinky_1",
    "L_pinky_2",
    "right_arm_link01",
    "right_arm_link02",
    "right_arm_link03",
    "right_arm_link04",
    "right_arm_link05",
    "right_arm_link06",
    "right_arm_link07",
    "R_thumb_swing",
    "R_thumb_1",
    "R_thumb_2",
    "R_thumb_3",
    "R_index_1",
    "R_index_2",
    "R_middle_1",
    "R_middle_2",
    "R_ring_1",
    "R_ring_2",
    "R_pinky_1",
    "R_pinky_2",
    "head_link01",
    "head_link02",
]

# The MJCF contains extra passive linkage bodies. These indices select the
# actuated/root subset above from the full 74-body MuJoCo ordering.
A2_ISAACLAB_TO_MUJOCO_BODY = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    11,
    12,
    13,
    14,
    15,
    16,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    71,
    72,
]

A2_MUJOCO_TO_ISAACLAB_BODY = A2_ISAACLAB_TO_MUJOCO_BODY

# Isaac Lab traverses the URDF in BFS order; MuJoCo uses depth-first XML order.
# Computed offline via yourdfpy (BFS) + mujoco.MjModel joint enumeration.
A2_ISAACLAB_TO_MUJOCO_DOF = [
    0, 6, 50, 1, 7, 12, 31, 51, 2, 8, 13, 32, 3, 9, 14, 33,
    4, 10, 15, 34, 5, 11, 16, 35, 17, 36, 18, 37, 19, 23, 25, 27,
    29, 38, 42, 44, 46, 48, 20, 24, 26, 28, 30, 39, 43, 45, 47, 49,
    21, 40, 22, 41,
]
A2_MUJOCO_TO_ISAACLAB_DOF = [
    0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21, 5, 10, 14, 18,
    22, 24, 26, 28, 38, 48, 50, 29, 39, 30, 40, 31, 41, 32, 42, 6,
    11, 15, 19, 23, 25, 27, 33, 43, 49, 51, 34, 44, 35, 45, 36, 46,
    37, 47, 2, 7,
]

A2_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": A2_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": A2_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": A2_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": A2_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": A2_MUJOCO_TO_ISAACLAB_BODY,
}

A2_LEG_JOINTS = [
    "idx01_left_hip_roll",
    "idx02_left_hip_yaw",
    "idx03_left_hip_pitch",
    "idx04_left_tarsus",
    "idx07_right_hip_roll",
    "idx08_right_hip_yaw",
    "idx09_right_hip_pitch",
    "idx10_right_tarsus",
]

A2_TOE_JOINTS = [
    "idx05_left_toe_pitch",
    "idx06_left_toe_roll",
    "idx11_right_toe_pitch",
    "idx12_right_toe_roll",
]

A2_ARM_JOINTS = [
    "idx13_left_arm_joint1",
    "idx14_left_arm_joint2",
    "idx15_left_arm_joint3",
    "idx16_left_arm_joint4",
    "idx17_left_arm_joint5",
    "idx18_left_arm_joint6",
    "idx19_left_arm_joint7",
    "idx20_right_arm_joint1",
    "idx21_right_arm_joint2",
    "idx22_right_arm_joint3",
    "idx23_right_arm_joint4",
    "idx24_right_arm_joint5",
    "idx25_right_arm_joint6",
    "idx26_right_arm_joint7",
]

A2_HAND_JOINTS = [
    "L_thumb_swing_joint",
    "L_thumb_1_joint",
    "L_thumb_2_joint",
    "L_thumb_3_joint",
    "L_index_1_joint",
    "L_index_2_joint",
    "L_middle_1_joint",
    "L_middle_2_joint",
    "L_ring_1_joint",
    "L_ring_2_joint",
    "L_pinky_1_joint",
    "L_pinky_2_joint",
    "R_thumb_swing_joint",
    "R_thumb_1_joint",
    "R_thumb_2_joint",
    "R_thumb_3_joint",
    "R_index_1_joint",
    "R_index_2_joint",
    "R_middle_1_joint",
    "R_middle_2_joint",
    "R_ring_1_joint",
    "R_ring_2_joint",
    "R_pinky_1_joint",
    "R_pinky_2_joint",
]

A2_HEAD_JOINTS = ["idx27_head_joint1", "idx28_head_joint2"]

_LEG_EFFORT = {
    "idx01_left_hip_roll": 120.0,
    "idx02_left_hip_yaw": 120.0,
    "idx03_left_hip_pitch": 140.0,
    "idx04_left_tarsus": 140.0,
    "idx07_right_hip_roll": 120.0,
    "idx08_right_hip_yaw": 120.0,
    "idx09_right_hip_pitch": 140.0,
    "idx10_right_tarsus": 140.0,
}

_LEG_STIFFNESS = {
    "idx01_left_hip_roll": STIFFNESS_7520_22,
    "idx02_left_hip_yaw": STIFFNESS_7520_14,
    "idx03_left_hip_pitch": STIFFNESS_7520_22,
    "idx04_left_tarsus": STIFFNESS_7520_22,
    "idx07_right_hip_roll": STIFFNESS_7520_22,
    "idx08_right_hip_yaw": STIFFNESS_7520_14,
    "idx09_right_hip_pitch": STIFFNESS_7520_22,
    "idx10_right_tarsus": STIFFNESS_7520_22,
}

_LEG_DAMPING = {
    "idx01_left_hip_roll": DAMPING_7520_22,
    "idx02_left_hip_yaw": DAMPING_7520_14,
    "idx03_left_hip_pitch": DAMPING_7520_22,
    "idx04_left_tarsus": DAMPING_7520_22,
    "idx07_right_hip_roll": DAMPING_7520_22,
    "idx08_right_hip_yaw": DAMPING_7520_14,
    "idx09_right_hip_pitch": DAMPING_7520_22,
    "idx10_right_tarsus": DAMPING_7520_22,
}

_LEG_ARMATURE = {
    "idx01_left_hip_roll": ARMATURE_7520_22,
    "idx02_left_hip_yaw": ARMATURE_7520_14,
    "idx03_left_hip_pitch": ARMATURE_7520_22,
    "idx04_left_tarsus": ARMATURE_7520_22,
    "idx07_right_hip_roll": ARMATURE_7520_22,
    "idx08_right_hip_yaw": ARMATURE_7520_14,
    "idx09_right_hip_pitch": ARMATURE_7520_22,
    "idx10_right_tarsus": ARMATURE_7520_22,
}

_TOE_EFFORT = dict.fromkeys(A2_TOE_JOINTS, 50.0)
_TOE_STIFFNESS = dict.fromkeys(A2_TOE_JOINTS, 2.0 * STIFFNESS_5020)
_TOE_DAMPING = dict.fromkeys(A2_TOE_JOINTS, 2.0 * DAMPING_5020)
_TOE_ARMATURE = dict.fromkeys(A2_TOE_JOINTS, 2.0 * ARMATURE_5020)

_ARM_EFFORT = {
    "idx13_left_arm_joint1": 75.0,
    "idx14_left_arm_joint2": 75.0,
    "idx15_left_arm_joint3": 75.0,
    "idx16_left_arm_joint4": 75.0,
    "idx17_left_arm_joint5": 75.0,
    "idx18_left_arm_joint6": 25.0,
    "idx19_left_arm_joint7": 25.0,
    "idx20_right_arm_joint1": 75.0,
    "idx21_right_arm_joint2": 75.0,
    "idx22_right_arm_joint3": 75.0,
    "idx23_right_arm_joint4": 75.0,
    "idx24_right_arm_joint5": 75.0,
    "idx25_right_arm_joint6": 25.0,
    "idx26_right_arm_joint7": 25.0,
}

_ARM_STIFFNESS = {
    "idx13_left_arm_joint1": STIFFNESS_5020,
    "idx14_left_arm_joint2": STIFFNESS_5020,
    "idx15_left_arm_joint3": STIFFNESS_5020,
    "idx16_left_arm_joint4": STIFFNESS_5020,
    "idx17_left_arm_joint5": STIFFNESS_5020,
    "idx18_left_arm_joint6": STIFFNESS_4010,
    "idx19_left_arm_joint7": STIFFNESS_4010,
    "idx20_right_arm_joint1": STIFFNESS_5020,
    "idx21_right_arm_joint2": STIFFNESS_5020,
    "idx22_right_arm_joint3": STIFFNESS_5020,
    "idx23_right_arm_joint4": STIFFNESS_5020,
    "idx24_right_arm_joint5": STIFFNESS_5020,
    "idx25_right_arm_joint6": STIFFNESS_4010,
    "idx26_right_arm_joint7": STIFFNESS_4010,
}

_ARM_DAMPING = {
    "idx13_left_arm_joint1": DAMPING_5020,
    "idx14_left_arm_joint2": DAMPING_5020,
    "idx15_left_arm_joint3": DAMPING_5020,
    "idx16_left_arm_joint4": DAMPING_5020,
    "idx17_left_arm_joint5": DAMPING_5020,
    "idx18_left_arm_joint6": DAMPING_4010,
    "idx19_left_arm_joint7": DAMPING_4010,
    "idx20_right_arm_joint1": DAMPING_5020,
    "idx21_right_arm_joint2": DAMPING_5020,
    "idx22_right_arm_joint3": DAMPING_5020,
    "idx23_right_arm_joint4": DAMPING_5020,
    "idx24_right_arm_joint5": DAMPING_5020,
    "idx25_right_arm_joint6": DAMPING_4010,
    "idx26_right_arm_joint7": DAMPING_4010,
}

_ARM_ARMATURE = {
    "idx13_left_arm_joint1": ARMATURE_5020,
    "idx14_left_arm_joint2": ARMATURE_5020,
    "idx15_left_arm_joint3": ARMATURE_5020,
    "idx16_left_arm_joint4": ARMATURE_5020,
    "idx17_left_arm_joint5": ARMATURE_5020,
    "idx18_left_arm_joint6": ARMATURE_4010,
    "idx19_left_arm_joint7": ARMATURE_4010,
    "idx20_right_arm_joint1": ARMATURE_5020,
    "idx21_right_arm_joint2": ARMATURE_5020,
    "idx22_right_arm_joint3": ARMATURE_5020,
    "idx23_right_arm_joint4": ARMATURE_5020,
    "idx24_right_arm_joint5": ARMATURE_5020,
    "idx25_right_arm_joint6": ARMATURE_4010,
    "idx26_right_arm_joint7": ARMATURE_4010,
}

_HAND_EFFORT = dict.fromkeys(A2_HAND_JOINTS, 5.0)
_HAND_STIFFNESS = dict.fromkeys(A2_HAND_JOINTS, FINGER_STIFFNESS)
_HAND_DAMPING = dict.fromkeys(A2_HAND_JOINTS, FINGER_DAMPING)
_HAND_ARMATURE = dict.fromkeys(A2_HAND_JOINTS, FINGER_ARMATURE)

_HEAD_EFFORT = dict.fromkeys(A2_HEAD_JOINTS, 20.0)
_HEAD_STIFFNESS = dict.fromkeys(A2_HEAD_JOINTS, STIFFNESS_4010)
_HEAD_DAMPING = dict.fromkeys(A2_HEAD_JOINTS, DAMPING_4010)
_HEAD_ARMATURE = dict.fromkeys(A2_HEAD_JOINTS, ARMATURE_4010)

A2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/robot_description/urdf/a2/a2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "idx03_left_hip_pitch": -0.30,
            "idx04_left_tarsus": 0.60,
            "idx09_right_hip_pitch": -0.30,
            "idx10_right_tarsus": 0.60,
            "idx13_left_arm_joint1": -0.02,
            "idx14_left_arm_joint2": 1.20,
            "idx15_left_arm_joint3": -0.14,
            "idx16_left_arm_joint4": -0.83,
            "idx17_left_arm_joint5": 1.50,
            "idx20_right_arm_joint1": -0.02,
            "idx21_right_arm_joint2": -1.20,
            "idx22_right_arm_joint3": -0.14,
            "idx23_right_arm_joint4": 0.83,
            "idx24_right_arm_joint5": 1.50,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=A2_LEG_JOINTS,
            effort_limit_sim=_LEG_EFFORT,
            velocity_limit_sim=dict.fromkeys(A2_LEG_JOINTS, 12.0),
            stiffness=_LEG_STIFFNESS,
            damping=_LEG_DAMPING,
            armature=_LEG_ARMATURE,
        ),
        "toes": ImplicitActuatorCfg(
            joint_names_expr=A2_TOE_JOINTS,
            effort_limit_sim=_TOE_EFFORT,
            velocity_limit_sim=dict.fromkeys(A2_TOE_JOINTS, 12.0),
            stiffness=_TOE_STIFFNESS,
            damping=_TOE_DAMPING,
            armature=_TOE_ARMATURE,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=A2_ARM_JOINTS,
            effort_limit_sim=_ARM_EFFORT,
            velocity_limit_sim=dict.fromkeys(A2_ARM_JOINTS, 20.0),
            stiffness=_ARM_STIFFNESS,
            damping=_ARM_DAMPING,
            armature=_ARM_ARMATURE,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=A2_HAND_JOINTS,
            effort_limit_sim=_HAND_EFFORT,
            velocity_limit_sim=dict.fromkeys(A2_HAND_JOINTS, 20.0),
            stiffness=_HAND_STIFFNESS,
            damping=_HAND_DAMPING,
            armature=_HAND_ARMATURE,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=A2_HEAD_JOINTS,
            effort_limit_sim=_HEAD_EFFORT,
            velocity_limit_sim=dict.fromkeys(A2_HEAD_JOINTS, 12.0),
            stiffness=_HEAD_STIFFNESS,
            damping=_HEAD_DAMPING,
            armature=_HEAD_ARMATURE,
        ),
    },
)

A2_ACTION_SCALE = {}
for actuator in A2_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = dict.fromkeys(names, effort)
    if not isinstance(stiffness, dict):
        stiffness = dict.fromkeys(names, stiffness)
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            A2_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
