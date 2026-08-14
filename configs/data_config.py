# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from typing import Optional
from BeingH.utils.schema import RotationType
from BeingH.dataset.transform.base import ComposedModalityTransform, ModalityTransform
from BeingH.dataset.transform.concat import ConcatTransform
from BeingH.dataset.transform.state_action import StateActionToTensor, StateActionTransform
from BeingH.utils.constants import TARGET_STATE_ROTATION_TYPE, TARGET_ACTION_ROTATION_TYPE, TARGET_STATE_ROTATION_DIM, TARGET_ACTION_ROTATION_DIM, AGIBOT_ABS_OR_RELA


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class ModalityDef(BaseModel):
    source_column: str = Field(..., description="Original column name in the Parquet file")
    start: int = Field(..., description="Start dimension index in the column")
    end: int = Field(..., description="End dimension index in the column (exclusive)")
    absolute: bool = True

    rotation_type: Optional[RotationType] = Field(None, description="Rotation representation type, if applicable")
    continuous: bool = Field(True, description="Whether the data is continuous (floating point)")


class BaseDataConfig(ABC):
    # Translation from policy/data wrist xyz to MuJoCo world xyz. Existing
    # datasets use world coordinates unless a concrete config overrides this.
    WRIST_WORLD_ORIGIN = (0.0, 0.0, 0.0)

    def __init__(self, embodiment_tag, use_fixed_view, max_view_num, 
                obs_indices=[0], action_indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        self.embodiment_tag = embodiment_tag
        self.use_fixed_view = use_fixed_view
        self.max_view_num = max_view_num
        self.obs_indices = obs_indices
        self.action_indices = action_indices

    @abstractmethod
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """
        Define how to extract and name new modalities from raw Parquet columns.
        Returns: {'modality.key': ModalityDef(...), ...}
        """
        pass

    def get_sampling_indices(self) -> Dict[str, List[int]]:
        """Define sampling indices"""
        sampling_map = {}
        for key in self.VIDEO_KEYS + self.STATE_KEYS:
            sampling_map[key] = self.obs_indices
        for key in self.ACTION_KEYS:
            sampling_map[key] = self.action_indices
        return sampling_map

    @abstractmethod
    def get_transforms(self) -> ModalityTransform:
        """
        Define a complete, ordered data transformation pipeline.
        Returns a ComposedModalityTransform object.
        """
        pass

    def add_video_modality(self, modalities):
        if self.use_fixed_view:
            video_keys = [next(iter(self.VIDEO_SOURCE_COLUMNS))]
        elif self.max_view_num == -1:
            video_keys = list(self.VIDEO_SOURCE_COLUMNS.keys())
            # rand_view_num = random.randint(1, len(self.VIDEO_SOURCE_COLUMNS))
            # video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), rand_view_num)
        else:
            max_view_num = min(self.max_view_num, len(self.VIDEO_SOURCE_COLUMNS))
            video_keys = random.sample(self.VIDEO_SOURCE_COLUMNS.keys(), max_view_num)
   
        for video_key in video_keys:
            modalities[video_key] = ModalityDef(source_column=self.VIDEO_SOURCE_COLUMNS[video_key], start=0, end=0)

        return modalities


class LiberoOriginDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.top_view']
    VIDEO_SOURCE_COLUMNS = {'video.top_view': 'observation.images.image'}
    STATE_KEYS = ['state.state']
    ACTION_KEYS = ['action.action']

    LANGUAGE_KEYS = ['language.instruction']

    state_normalization_modes = {'state.state': 'min_max'} 
    action_normalization_modes = {'action.action': 'min_max'}

    state_action_type = {'state.state': "7-d absolute state (xyz,roll,pitch,yaw,pad) + 1-d gripper pos", 
                         'action.action': "6-d relative action (xyz,roll,pitch,yaw) + 1-d gripper pos"
                        }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),
            'state.state': ModalityDef(source_column='observation.state', start=0, end=8),
            'action.action': ModalityDef(source_column='action', start=0, end=7, absolute=False),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                normalization_modes=self.state_normalization_modes
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                normalization_modes=self.action_normalization_modes
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class LiberoNoNormDataConfig(LiberoOriginDataConfig):
    VIDEO_KEYS = ['video.top_view', 'video.wrist_view']
    VIDEO_SOURCE_COLUMNS = {
        'video.top_view': 'observation.images.image',
        'video.wrist_view': 'observation.images.wrist_image',
    }
    STATE_KEYS = ['state.eef_position', 'state.eef_rotation', 'state.libero_gripper_position']
    ACTION_KEYS = ['action.eef_position', 'action.eef_rotation', 'action.gripper_position']

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position':     (0, 3),
        'state.eef_rotation':  (3, 6),
        'state.libero_gripper_position': (44, 46),

        'action.eef_position':    (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper_position':(18, 19),
    }

    state_normalization_modes = {
    }
    
    action_normalization_modes = {
    }

    def get_feature_meta(self):
        return {'state.eef_position': ("3-d absolute eef position (xyz)", 3), 
                'state.eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
                'state.libero_gripper_position': ("2-d gripper position", 2),
                'action.eef_position': ("3-d relative eef position (xyz)", 3), 
                'action.eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
                'action.gripper_position': ("1-d gripper position"),
            }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.eef_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.eef_rotation': ModalityDef(source_column='observation.state', start=3, end=6, rotation_type="axis_angle"),
            'state.libero_gripper_position': ModalityDef(source_column='observation.state', start=6, end=8),

            'action.eef_position': ModalityDef(source_column='action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(source_column='action', start=3, end=6, absolute=False, rotation_type="axis_angle"),
            'action.gripper_position': ModalityDef(source_column='action', start=6, end=7),
        }
        modalities = self.add_video_modality(modalities)

        return modalities


class RobocasaHumanDataConfig(BaseDataConfig):
    VIDEO_KEYS = ['video.left_view', 'video.right_view', 'video.wrist_view']
    VIDEO_SOURCE_COLUMNS = {
        'video.left_view': 'observation.images.left_view',
        'video.right_view': 'observation.images.right_view',
        'video.wrist_view': 'observation.images.wrist_view',
    }
    STATE_KEYS = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_qpos",
        "state.base_position",
        "state.base_rotation",
    ]
    ACTION_KEYS = [
        "action.eef_position",
        "action.eef_rotation",
        "action.gripper_position",
        "action.base_motion",
        "action.control_mode",
    ]

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        'state.eef_position':  (0, 3),
        'state.eef_rotation':  (3, 6),
        'state.gripper_qpos': (44, 46),
        'state.base_position': (70, 73),
        'state.base_rotation': (73, 76),

        'action.eef_position': (0, 3),
        'action.eef_rotation': (3, 6),
        'action.gripper_position': (18, 19),
        'action.base_motion': (70, 74),
        'action.control_mode': (74, 75),
    }

    LANGUAGE_KEYS = ['language.instruction']

    state_normalization_modes = {} 
    # action_normalization_modes = {}

    action_normalization_modes = {
        # "action.end_effector_position": "min_max",
        # "action.end_effector_rotation": "min_max",
        "action.gripper_position": "binary",
        # "action.base_motion": "min_max",
        "action.control_mode": "binary",
    }

    def get_feature_meta(self):
        return {'state.eef_position': ("3-d absolute eef position (xyz)", 3), 
                'state.eef_rotation': (f"{TARGET_STATE_ROTATION_DIM}-d absolute eef rotation ({TARGET_STATE_ROTATION_TYPE})", TARGET_STATE_ROTATION_DIM),
                'state.gripper_qpos': ("2-d gripper position", 2),
                'action.eef_position': ("3-d relative eef position (xyz)", 3), 
                'action.eef_rotation': (f"{TARGET_ACTION_ROTATION_DIM}-d relative eef rotation ({TARGET_ACTION_ROTATION_TYPE})", TARGET_ACTION_ROTATION_DIM),
                'action.gripper_position': ("1-d gripper position"),
            }
    
    def define_modalities(self) -> Dict[str, ModalityDef]:
        """Extract modalities from Parquet columns"""
        modalities = {
            'language.instruction': ModalityDef(source_column='task_index', start=0, end=0),

            'state.eef_position': ModalityDef(source_column='world_abs_state', start=0, end=3),
            'state.eef_rotation': ModalityDef(source_column='world_abs_state', start=3, end=6, rotation_type="axis_angle"),
            'state.gripper_qpos': ModalityDef(source_column='world_abs_state', start=6, end=8),
            'state.base_position': ModalityDef(source_column='observation.state', start=0, end=3),
            'state.base_rotation': ModalityDef(source_column='observation.state', start=3, end=7, rotation_type="quaternion"),

            'action.eef_position': ModalityDef(source_column='world_delta_action', start=0, end=3, absolute=False),
            'action.eef_rotation': ModalityDef(source_column='world_delta_action', start=3, end=6, absolute=False, rotation_type="axis_angle"),
            'action.gripper_position': ModalityDef(source_column='world_delta_action', start=6, end=7),
            'action.base_motion': ModalityDef(source_column='action', start=7, end=11, absolute=False),
            'action.control_mode': ModalityDef(source_column='action', start=11, end=12),
        }
        modalities = self.add_video_modality(modalities)
        return modalities

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={
                    # "state.eef_rotation": TARGET_STATE_ROTATION_TYPE,
                    "state.base_rotation": TARGET_STATE_ROTATION_TYPE
                },
                # normalization_modes=self.action_normalization_modes,
            ),

            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                # target_rotations={"action.eef_rotation": TARGET_ACTION_ROTATION_TYPE},
                normalization_modes=self.action_normalization_modes,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class ShadowGraspWristGestureDataConfig(BaseDataConfig):
    """Shadow Hand wrist pose plus 24-D gesture latent."""

    VIDEO_KEYS = ["video.ego_opposite"]
    VIDEO_SOURCE_COLUMNS = {
        "video.ego_opposite": "observation.images.ego_opposite",
    }

    STATE_KEYS = [
        "state.eef_position",
        "state.eef_rotation",
        "state.z_gesture",
    ]
    ACTION_KEYS = [
        "action.eef_position",
        "action.eef_rotation",
        "action.z_gesture",
    ]
    LANGUAGE_KEYS = ["language.instruction"]

    # The unified-space wrist slots follow Being-H's EEF convention.
    # Dims 20:44 are the 24-D right dexterous-hand region.
    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        "state.eef_position": (0, 3),
        "state.eef_rotation": (3, 6),
        "state.z_gesture": (20, 44),
        "action.eef_position": (0, 3),
        "action.eef_rotation": (3, 6),
        "action.z_gesture": (20, 44),
    }

    # z_gesture is already a learned latent; keep it in its native scale.
    state_normalization_modes = {}
    action_normalization_modes = {}

    def get_feature_meta(self):
        return {
            "state.eef_position": (
                "3-d absolute wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "state.eef_rotation": (
                f"{TARGET_STATE_ROTATION_DIM}-d absolute wrist rotation "
                f"({TARGET_STATE_ROTATION_TYPE}, converted from Euler RPY)",
                TARGET_STATE_ROTATION_DIM,
            ),
            "state.z_gesture": ("24-d absolute Shadow Hand gesture latent", 24),
            "action.eef_position": (
                "3-d absolute target wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "action.eef_rotation": (
                f"{TARGET_ACTION_ROTATION_DIM}-d absolute target wrist rotation "
                f"({TARGET_ACTION_ROTATION_TYPE}, converted from Euler RPY)",
                TARGET_ACTION_ROTATION_DIM,
            ),
            "action.z_gesture": (
                "24-d absolute target Shadow Hand gesture latent",
                24,
            ),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = {
            "language.instruction": ModalityDef(
                source_column="task_index", start=0, end=0
            ),
            "state.eef_position": ModalityDef(
                source_column="observation.state",
                start=0,
                end=3,
                absolute=True,
            ),
            "state.eef_rotation": ModalityDef(
                source_column="observation.state",
                start=3,
                end=6,
                absolute=True,
                rotation_type="euler_angles_rpy",
            ),
            # Raw dims 6:28 are the 22 physical joints and are intentionally skipped.
            "state.z_gesture": ModalityDef(
                source_column="observation.state",
                start=28,
                end=52,
                absolute=True,
            ),
            "action.eef_position": ModalityDef(
                source_column="action",
                start=0,
                end=3,
                absolute=True,
            ),
            "action.eef_rotation": ModalityDef(
                source_column="action",
                start=3,
                end=6,
                absolute=True,
                rotation_type="euler_angles_rpy",
            ),
            "action.z_gesture": ModalityDef(
                source_column="action",
                start=28,
                end=52,
                absolute=True,
            ),
        }
        return self.add_video_modality(modalities)

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={
                    "state.eef_rotation": TARGET_STATE_ROTATION_TYPE,
                },
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                target_rotations={
                    "action.eef_rotation": TARGET_ACTION_ROTATION_TYPE,
                },
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class ShadowGraspWristGestureQ99DataConfig(
    ShadowGraspWristGestureDataConfig
):
    """Shadow wrist/gesture config with built-in Being-H normalization."""

    # Position and gesture statistics come from per-dimension q01/q99 values
    # in meta/stats.json. Euler rotations are converted to axis-angle first;
    # Being-H therefore requires its representation-aware [-pi, pi] min-max
    # normalization instead of applying raw Euler quantiles.
    state_normalization_modes = {
        "state.eef_position": "q99",
        "state.eef_rotation": "min_max",
        "state.z_gesture": "q99",
    }
    action_normalization_modes = {
        "action.eef_position": "q99",
        "action.eef_rotation": "min_max",
        "action.z_gesture": "q99",
    }


class ShadowGraspWristGestureMinMaxDataConfig(
    ShadowGraspWristGestureDataConfig
):
    """Shadow wrist/gesture config with per-dimension min-max normalization."""

    # Position and gesture use the observed per-dimension min/max statistics
    # from meta/stats.json. Euler rotations are converted to axis-angle first,
    # then normalized with Being-H's representation-aware [-pi, pi] bounds.
    state_normalization_modes = {
        "state.eef_position": "min_max",
        "state.eef_rotation": "min_max",
        "state.z_gesture": "min_max",
    }
    action_normalization_modes = {
        "action.eef_position": "min_max",
        "action.eef_rotation": "min_max",
        "action.z_gesture": "min_max",
    }


class ShadowGraspWristMinMaxGestureRawDataConfig(
    ShadowGraspWristGestureDataConfig
):
    """Normalize wrist modalities only and keep the VAE gesture latent raw."""

    # Position uses the observed per-dimension min/max statistics. Euler
    # rotations are converted to axis-angle before Being-H applies its
    # representation-aware min-max transform. Deliberately omitting
    # state/action.z_gesture means those 24 latent dimensions pass through
    # unchanged in both training and inference.
    state_normalization_modes = {
        "state.eef_position": "min_max",
        "state.eef_rotation": "min_max",
    }
    action_normalization_modes = {
        "action.eef_position": "min_max",
        "action.eef_rotation": "min_max",
    }


class ShadowGraspWristEulerMinMaxGestureRawDataConfig(
    ShadowGraspWristGestureDataConfig
):
    """Train continuous intrinsic-XYZ wrist Euler angles without axis-angle conversion.

    The source dataset stores continuous/unwrapped Euler angles in radians. Keeping
    that representation avoids the principal-axis-angle branch discontinuity at
    rotation angle pi. Wrist position and Euler components use their observed
    per-dimension min/max statistics; the VAE gesture latent remains unnormalized.
    """

    state_normalization_modes = {
        "state.eef_position": "min_max",
        "state.eef_rotation": "min_max",
    }
    action_normalization_modes = {
        "action.eef_position": "min_max",
        "action.eef_rotation": "min_max",
    }

    def get_feature_meta(self):
        return {
            "state.eef_position": (
                "3-d absolute wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "state.eef_rotation": (
                "3-d absolute intrinsic-XYZ wrist Euler rotation "
                "(radians, continuous/unwrapped)",
                3,
            ),
            "state.z_gesture": ("24-d absolute Shadow Hand gesture latent", 24),
            "action.eef_position": (
                "3-d absolute target wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "action.eef_rotation": (
                "3-d absolute target intrinsic-XYZ wrist Euler rotation "
                "(radians, continuous/unwrapped)",
                3,
            ),
            "action.z_gesture": (
                "24-d absolute target Shadow Hand gesture latent",
                24,
            ),
        }

    def get_transforms(self) -> ModalityTransform:
        # Deliberately omit target_rotations: source Euler angles stay Euler
        # throughout model training and are returned as Euler at inference.
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={},
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                target_rotations={},
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class ShadowGraspTwoCameraWristEulerMinMaxGestureRawDataConfig(
    ShadowGraspWristEulerMinMaxGestureRawDataConfig
):
    """Euler/min-max wrist config using external and wrist-mounted cameras."""

    VIDEO_KEYS = ["video.ego_opposite", "video.wrist"]
    VIDEO_SOURCE_COLUMNS = {
        "video.ego_opposite": "observation.images.ego_opposite",
        "video.wrist": "observation.images.wrist",
    }
    WRIST_WORLD_ORIGIN = (0.0, 0.0, 0.4)


class ShadowGraspWristRot6DMinMaxGestureRawDataConfig(
    ShadowGraspWristGestureDataConfig
):
    """Use Rot6D inside Being-H while keeping raw Euler I/O and z latent scale.

    The LeRobot source remains intrinsic-XYZ Euler in dimensions 3:6. The
    invertible state/action transform converts Euler -> rotation matrix ->
    PyTorch3D Rot6D before packing, and converts model predictions back to the
    source Euler representation during inference.
    """

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        "state.eef_position": (0, 3),
        "state.eef_rotation": (3, 9),
        "state.z_gesture": (20, 44),
        "action.eef_position": (0, 3),
        "action.eef_rotation": (3, 9),
        "action.z_gesture": (20, 44),
    }

    # Converted absolute Rot6D uses StateActionTransform's representation-aware
    # fixed [-1, 1] bounds. Position uses dataset min/max; z_gesture stays raw.
    state_normalization_modes = {
        "state.eef_position": "min_max",
        "state.eef_rotation": "min_max",
    }
    action_normalization_modes = {
        "action.eef_position": "min_max",
        "action.eef_rotation": "min_max",
    }

    def get_feature_meta(self):
        return {
            "state.eef_position": (
                "3-d absolute wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "state.eef_rotation": (
                "6-d absolute wrist Rot6D converted from intrinsic-XYZ Euler",
                6,
            ),
            "state.z_gesture": ("24-d absolute Shadow Hand gesture latent", 24),
            "action.eef_position": (
                "3-d absolute target wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "action.eef_rotation": (
                "6-d absolute target wrist Rot6D converted from intrinsic-XYZ Euler",
                6,
            ),
            "action.z_gesture": (
                "24-d absolute target Shadow Hand gesture latent",
                24,
            ),
        }

    def get_transforms(self) -> ModalityTransform:
        transforms = [
            StateActionToTensor(apply_to=self.STATE_KEYS),
            StateActionTransform(
                apply_to=self.STATE_KEYS,
                target_rotations={"state.eef_rotation": "rotation_6d"},
                normalization_modes=self.state_normalization_modes,
            ),
            StateActionToTensor(apply_to=self.ACTION_KEYS),
            StateActionTransform(
                apply_to=self.ACTION_KEYS,
                target_rotations={"action.eef_rotation": "rotation_6d"},
                normalization_modes=self.action_normalization_modes,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


class ShadowGraspTwoCameraWristRot6DMinMaxGestureRawDataConfig(
    ShadowGraspWristRot6DMinMaxGestureRawDataConfig
):
    """Rot6D/min-max wrist config using external and wrist-mounted cameras."""

    VIDEO_KEYS = ["video.ego_opposite", "video.wrist"]
    VIDEO_SOURCE_COLUMNS = {
        "video.ego_opposite": "observation.images.ego_opposite",
        "video.wrist": "observation.images.wrist",
    }
    WRIST_WORLD_ORIGIN = (0.0, 0.0, 0.4)


class ShadowGraspTwoCameraWristRot6DMinMaxJointsDataConfig(
    ShadowGraspWristRot6DMinMaxGestureRawDataConfig
):
    """Rot6D wrist plus normalized 22-D physical Shadow joint positions.

    This is a morphology-specific baseline for measuring the value of the
    cross-hand z_gesture representation. It reads the physical Shadow joint
    coordinates from source dimensions 6:28 and deliberately ignores the
    z_gesture latent in dimensions 28:52. Wrist position, converted Rot6D, and
    every joint coordinate are min-max normalized.
    """

    VIDEO_KEYS = ["video.ego_opposite", "video.wrist"]
    VIDEO_SOURCE_COLUMNS = {
        "video.ego_opposite": "observation.images.ego_opposite",
        "video.wrist": "observation.images.wrist",
    }
    WRIST_WORLD_ORIGIN = (0.0, 0.0, 0.4)

    STATE_KEYS = [
        "state.eef_position",
        "state.eef_rotation",
        "state.shadow_joint_position",
    ]
    ACTION_KEYS = [
        "action.eef_position",
        "action.eef_rotation",
        "action.shadow_joint_position",
    ]

    # Reuse the right dexterous-hand region. The final two slots (42:44) stay
    # masked because this baseline has 22 physical joints instead of a 24-D
    # gesture latent.
    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        "state.eef_position": (0, 3),
        "state.eef_rotation": (3, 9),
        "state.shadow_joint_position": (20, 42),
        "action.eef_position": (0, 3),
        "action.eef_rotation": (3, 9),
        "action.shadow_joint_position": (20, 42),
    }

    state_normalization_modes = {
        "state.eef_position": "min_max",
        "state.eef_rotation": "min_max",
        "state.shadow_joint_position": "min_max",
    }
    action_normalization_modes = {
        "action.eef_position": "min_max",
        "action.eef_rotation": "min_max",
        "action.shadow_joint_position": "min_max",
    }

    def get_feature_meta(self):
        return {
            "state.eef_position": (
                "3-d absolute wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "state.eef_rotation": (
                "6-d absolute wrist Rot6D converted from intrinsic-XYZ Euler",
                6,
            ),
            "state.shadow_joint_position": (
                "22-d absolute physical Shadow Hand joint position",
                22,
            ),
            "action.eef_position": (
                "3-d absolute target wrist position in wrist_world_origin frame (xyz)",
                3,
            ),
            "action.eef_rotation": (
                "6-d absolute target wrist Rot6D converted from intrinsic-XYZ Euler",
                6,
            ),
            "action.shadow_joint_position": (
                "22-d absolute target physical Shadow Hand joint position",
                22,
            ),
        }

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = {
            "language.instruction": ModalityDef(
                source_column="task_index", start=0, end=0
            ),
            "state.eef_position": ModalityDef(
                source_column="observation.state",
                start=0,
                end=3,
                absolute=True,
            ),
            "state.eef_rotation": ModalityDef(
                source_column="observation.state",
                start=3,
                end=6,
                absolute=True,
                rotation_type="euler_angles_rpy",
            ),
            "state.shadow_joint_position": ModalityDef(
                source_column="observation.state",
                start=6,
                end=28,
                absolute=True,
            ),
            "action.eef_position": ModalityDef(
                source_column="action",
                start=0,
                end=3,
                absolute=True,
            ),
            "action.eef_rotation": ModalityDef(
                source_column="action",
                start=3,
                end=6,
                absolute=True,
                rotation_type="euler_angles_rpy",
            ),
            "action.shadow_joint_position": ModalityDef(
                source_column="action",
                start=6,
                end=28,
                absolute=True,
            ),
        }
        return self.add_video_modality(modalities)


class SharpaGraspTwoCameraWristRot6DMinMaxJointsDataConfig(
    ShadowGraspTwoCameraWristRot6DMinMaxJointsDataConfig
):
    """Two-camera Sharpa physical-joint baseline.

    The generic shadow_joint_position key is retained so the Being-H
    action pipeline can use the same physical-joint baseline path. The 22
    source joints are Sharpa joints, not Shadow joints.
    """


class GaiaGraspTwoCameraWristRot6DMinMaxJointsDataConfig(
    ShadowGraspTwoCameraWristRot6DMinMaxJointsDataConfig
):
    """Two-camera Gaia physical-joint baseline with 15 source joints."""

    UNIFIED_MAPPING: Dict[str, Tuple[int, int]] = {
        "state.eef_position": (0, 3),
        "state.eef_rotation": (3, 9),
        "state.shadow_joint_position": (20, 35),
        "action.eef_position": (0, 3),
        "action.eef_rotation": (3, 9),
        "action.shadow_joint_position": (20, 35),
    }

    def get_feature_meta(self):
        meta = super().get_feature_meta()
        meta["state.shadow_joint_position"] = (
            "15-d absolute physical Gaia Hand joint position",
            15,
        )
        meta["action.shadow_joint_position"] = (
            "15-d absolute target physical Gaia Hand joint position",
            15,
        )
        return meta

    def define_modalities(self) -> Dict[str, ModalityDef]:
        modalities = super().define_modalities()
        modalities["state.shadow_joint_position"] = ModalityDef(
            source_column="observation.state",
            start=6,
            end=21,
            absolute=True,
        )
        modalities["action.shadow_joint_position"] = ModalityDef(
            source_column="action",
            start=6,
            end=21,
            absolute=True,
        )
        return modalities


DATA_CONFIG_MAP = {
    "libero_nonorm": LiberoNoNormDataConfig,
    "robocasa_human": RobocasaHumanDataConfig,
    "shadow_grasp_wrist_gesture": ShadowGraspWristGestureDataConfig,
    "shadow_grasp_wrist_gesture_q99": (
        ShadowGraspWristGestureQ99DataConfig
    ),
    "shadow_grasp_wrist_gesture_minmax": (
        ShadowGraspWristGestureMinMaxDataConfig
    ),
    "shadow_grasp_wrist_minmax_gesture_raw": (
        ShadowGraspWristMinMaxGestureRawDataConfig
    ),
    "shadow_grasp_wrist_euler_minmax_gesture_raw": (
        ShadowGraspWristEulerMinMaxGestureRawDataConfig
    ),
    "shadow_grasp_2cam_wrist_euler_minmax_gesture_raw": (
        ShadowGraspTwoCameraWristEulerMinMaxGestureRawDataConfig
    ),
    "shadow_grasp_wrist_rot6d_minmax_gesture_raw": (
        ShadowGraspWristRot6DMinMaxGestureRawDataConfig
    ),
    "shadow_grasp_2cam_wrist_rot6d_minmax_gesture_raw": (
        ShadowGraspTwoCameraWristRot6DMinMaxGestureRawDataConfig
    ),
    "shadow_grasp_2cam_wrist_rot6d_minmax_joints": (
        ShadowGraspTwoCameraWristRot6DMinMaxJointsDataConfig
    ),
    "sharpa_grasp_2cam_wrist_rot6d_minmax_joints": (
        SharpaGraspTwoCameraWristRot6DMinMaxJointsDataConfig
    ),
    "gaia_grasp_2cam_wrist_rot6d_minmax_joints": (
        GaiaGraspTwoCameraWristRot6DMinMaxJointsDataConfig
    ),
}
