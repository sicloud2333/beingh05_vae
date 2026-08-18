import os
from pathlib import Path


from BeingH.dataset.datasets.vla_dataset import LeRobotIterableDataset
from BeingH.dataset.datasets.vlm_dataset import SftJSONLIterableDataset


DATA_ROOT = Path(os.environ.get("BEINGH_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
LIBERO_ROOT = Path(os.environ.get("LIBERO_DATA_ROOT", DATA_ROOT / "libero"))
ROBOCASA_ROOT = Path(os.environ.get("ROBOCASA_DATA_ROOT", DATA_ROOT / "robocasa"))
REAL_ROOT = Path(os.environ.get("REAL_DATA_ROOT", DATA_ROOT / "real"))


DATASET_REGISTRY = {
    'libero_posttrain': LeRobotIterableDataset,
    'robocasa_human_posttrain': LeRobotIterableDataset,
    'shadow_grasp_posttrain': LeRobotIterableDataset,
    'uni_posttrain': LeRobotIterableDataset,
}


DATASET_INFO = {
    'shadow_grasp_posttrain': {
        'shadow_grasp_bottle22249179_aug100_2cam': {
            'dataset_path': str(DATA_ROOT / "shadow_grasp_bottle22249179_aug100_2cam"),
        },
        'shadow_grasp_bottle22249179_aug100_npuvae_2cam': {
            'dataset_path': str(DATA_ROOT / "shadow_grasp_bottle22249179_aug100_npuvae_2cam"),
        },
        'sharpa_grasp_bottle22249179_geo_visual100_2cam': {
            'dataset_path': str(DATA_ROOT / "sharpa_grasp_bottle22249179_geo_visual100_2cam"),
        },
        'gaia_grasp_bottle22249179_geo_visual100_2cam': {
            'dataset_path': str(DATA_ROOT / "gaia_grasp_bottle22249179_geo_visual100_2cam"),
        },
        'shadow_grasp_0725_core_bottle_1071': {
            'dataset_path': str(DATA_ROOT / "shadow_grasp_0725_core_bottle_1071"),
        },
        'shadow_grasp_0725': {
            'dataset_path': str(DATA_ROOT / "shadow_grasp_0725"),
        },
    },

    'libero_posttrain': {
        'libero_spatial': {
            'dataset_path': str(LIBERO_ROOT / "libero_spatial_no_noops_1.0.0_lerobot"),
        },
        'libero_object': {
            'dataset_path': str(LIBERO_ROOT / "libero_object_no_noops_1.0.0_lerobot"),
        },
        'libero_goal': {
            'dataset_path': str(LIBERO_ROOT / "libero_goal_no_noops_1.0.0_lerobot"),
        },
        'libero_10': {
            'dataset_path': str(LIBERO_ROOT / "libero_10_no_noops_1.0.0_lerobot"),
        },
    },

    'robocasa_human_posttrain': {
        'single_panda_gripper.CloseDoubleDoor': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "CloseDoubleDoor"),
        },
        'single_panda_gripper.CloseDrawer': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "CloseDrawer"),
        },
        'single_panda_gripper.CloseSingleDoor': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "CloseSingleDoor"),
        },

        'single_panda_gripper.CoffeePressButton': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "CoffeePressButton"),
        },
        'single_panda_gripper.CoffeeServeMug': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "CoffeeServeMug"),
        },
        'single_panda_gripper.CoffeeSetupMug': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "CoffeeSetupMug"),
        },

        'single_panda_gripper.OpenDoubleDoor': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "OpenDoubleDoor"),
        },
        'single_panda_gripper.OpenDrawer': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "OpenDrawer"),
        },
        'single_panda_gripper.OpenSingleDoor': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "OpenSingleDoor"),
        },

        'single_panda_gripper.PnPCabToCounter': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPCabToCounter"),
        },
        'single_panda_gripper.PnPCounterToCab': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPCounterToCab"),
        },
        'single_panda_gripper.PnPCounterToMicrowave': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPCounterToMicrowave"),
        },
        'single_panda_gripper.PnPCounterToSink': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPCounterToSink"),
        },
        'single_panda_gripper.PnPCounterToStove': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPCounterToStove"),
        },
        'single_panda_gripper.PnPMicrowaveToCounter': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPMicrowaveToCounter"),
        },
        'single_panda_gripper.PnPSinkToCounter': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPSinkToCounter"),
        },
        'single_panda_gripper.PnPStoveToCounter': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "PnPStoveToCounter"),
        },

        'single_panda_gripper.TurnOffMicrowave': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOffMicrowave"),
        },
        'single_panda_gripper.TurnOffSinkFaucet': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOffSinkFaucet"),
        },
        'single_panda_gripper.TurnOffStove': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOffStove"),
        },
        'single_panda_gripper.TurnOnMicrowave': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOnMicrowave"),
        },
        'single_panda_gripper.TurnOnSinkFaucet': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOnSinkFaucet"),
        },
        'single_panda_gripper.TurnOnStove': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOnStove"),
        },
        'single_panda_gripper.TurnSinkSpout': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnSinkSpout"),
        },
    },

    'uni_posttrain': {
        # ========================================================================
        # ROBOCASA datasets
        # ========================================================================
        'single_panda_gripper.CloseDoubleDoor': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "CloseDoubleDoor"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseDoubleDoor',
        },

        'single_panda_gripper.CloseDrawer': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "CloseDrawer"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseDrawer',
        },

        'single_panda_gripper.CloseSingleDoor': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "CloseSingleDoor"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CloseSingleDoor',
        },

        'single_panda_gripper.CoffeePressButton': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "CoffeePressButton"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeePressButton',
        },

        'single_panda_gripper.CoffeeServeMug': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "CoffeeServeMug"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeeServeMug',
        },

        'single_panda_gripper.CoffeeSetupMug': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "CoffeeSetupMug"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.CoffeeSetupMug',
        },

        'single_panda_gripper.OpenDoubleDoor': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "OpenDoubleDoor"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenDoubleDoor',
        },

        'single_panda_gripper.OpenDrawer': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "OpenDrawer"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenDrawer',
        },

        'single_panda_gripper.OpenSingleDoor': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "OpenSingleDoor"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.OpenSingleDoor',
        },

        'single_panda_gripper.PnPCabToCounter': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPCabToCounter"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCabToCounter',
        },

        'single_panda_gripper.PnPCounterToCab': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPCounterToCab"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToCab',
        },

        'single_panda_gripper.PnPCounterToMicrowave': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPCounterToMicrowave"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToMicrowave',
        },

        'single_panda_gripper.PnPCounterToSink': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPCounterToSink"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToSink',
        },

        'single_panda_gripper.PnPCounterToStove': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPCounterToStove"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPCounterToStove',
        },

        'single_panda_gripper.PnPMicrowaveToCounter': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPMicrowaveToCounter"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPMicrowaveToCounter',
        },

        'single_panda_gripper.PnPSinkToCounter': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPSinkToCounter"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPSinkToCounter',
        },

        'single_panda_gripper.PnPStoveToCounter': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "PnPStoveToCounter"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.PnPStoveToCounter',
        },

        'single_panda_gripper.TurnOffMicrowave': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "TurnOffMicrowave"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffMicrowave',
        },

        'single_panda_gripper.TurnOffSinkFaucet': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "TurnOffSinkFaucet"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffSinkFaucet',
        },

        'single_panda_gripper.TurnOffStove': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "TurnOffStove"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOffStove',
        },

        'single_panda_gripper.TurnOnMicrowave': {
            'dataset_path': str(ROBOCASA_ROOT / "single_stage" / "TurnOnMicrowave"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnMicrowave',
        },

        'single_panda_gripper.TurnOnSinkFaucet': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "TurnOnSinkFaucet"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnSinkFaucet',
        },

        'single_panda_gripper.TurnOnStove': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "TurnOnStove"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnOnStove',
        },

        'single_panda_gripper.TurnSinkSpout': {
            'dataset_path': str(REAL_ROOT / "posttrain" / "ROBOCASA" / "TurnSinkSpout"),
            'embodiment': 'ROBOCASA',
            'embodiment_tag': 'robocasa',
            'subtask': 'single_panda_gripper.TurnSinkSpout',
        },
    },  
}
