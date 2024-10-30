import pytest
import json
import os
import torch
from vit_prisma.sae.config import VisionModelSAERunnerConfig


tmp_path = "/tmp"


def test_load_old_config(tmp_path):
    """
    Test that old configuration files can still be loaded correctly.
    """
    # Create the old configuration JSON content
    old_config_data = {
        "model_class_name": "HookedViT",
        "model_name": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "hook_point": "blocks.11.hook_resid_post",
        "hook_point_layer": 11,
        "hook_point_head_index": None,
        "context_size": 50,
        "use_cached_activations": False,
        "use_patches_only": False,
        "cached_activations_path": "activations/data_ImageNet-complete_/wkcn_TinyCLIP-ViT-40M-32-Text-19M-LAION400M/blocks.11.hook_resid_post",
        "d_in": 512,
        "activation_fn_str": "relu",
        "activation_fn_kwargs": {},
        "cls_token_only": False,  # Adjusted to match attribute name in the class
        "max_grad_norm": 1.0,
        "initialization_method": "encoder_transpose_decoder",
        "normalize_activations": "layer_norm",
        "n_batches_in_buffer": 20,
        "store_batch_size": 32,
        "num_workers": 16,
        "num_epochs": 5,
        "total_training_images": 13000000,
        "total_training_tokens": 650000000,
        "image_size": 224,
        "device": {"__type__": "torch.device", "value": "cuda:0"},
        "seed": 42,
        "dtype": {"__type__": "torch.dtype", "value": "torch.float32"},
        "architecture": "standard",
        "sparsity_loss": "l1",
        "verbose": False,
        "b_dec_init_method": "geometric_median",
        "expansion_factor": 16,
        "from_pretrained_path": None,
        "d_sae": 8192,
        "l1_coefficient": 0.0002,
        "lp_norm": 1,
        "lr": 0.001,
        "lr_scheduler_name": "cosineannealingwarmup",
        "lr_warm_up_steps": 10000,
        "beta1": 0.9,
        "beta2": 0.999,
        "train_batch_size": 4096,
        "dataset_name": "imagenet1k",
        "dataset_path": "data/ImageNet-complete/",
        "dataset_train_path": "data/ImageNet-complete/train",
        "dataset_val_path": "data/ImageNet-complete/val",
        "use_ghost_grads": True,
        "feature_sampling_window": 1000,
        "dead_feature_window": 5000,
        "dead_feature_threshold": 1e-08,
        "log_to_wandb": True,
        "wandb_project": "tinyclip_sae_16_hyperparam_sweep_lr",
        "wandb_entity": "lolorenz",
        "wandb_log_frequency": 10,
        "n_checkpoints": 50,
        "checkpoint_path": "models/sae",
    }

    # Save the old config to a temporary file
    old_config_path = tmp_path / "old_config.json"
    with open(old_config_path, "w") as f:
        json.dump(old_config_data, f, indent=4)

    # Load the old config
    loaded_config = VisionModelSAERunnerConfig.load_config(str(old_config_path))

    # Expected values from the old config
    expected_values = old_config_data.copy()
    # Adjust special types
    expected_values["device"] = torch.device(expected_values["device"]["value"])
    expected_values["dtype"] = torch.float32  # Since it's torch.float32
    expected_values["hook_point_head_index"] = None  # Ensure None is NoneType
    expected_values["from_pretrained_path"] = None  # Ensure None is NoneType

    # Since 'hook_point' is computed in __post_init__, we need to check it separately
    expected_hook_point = expected_values.pop("hook_point")

    # Check that each attribute matches the expected value
    for attr, expected_value in expected_values.items():
        actual_value = getattr(loaded_config, attr)
        assert (
            actual_value == expected_value
        ), f"Attribute '{attr}' does not match expected value"

    # Check 'hook_point' separately
    assert (
        loaded_config.hook_point == expected_hook_point
    ), f"Attribute 'hook_point' does not match expected value"


def test_default_values():
    """
    Test that the default values are the same and can be accessed.
    """
    # Create a config instance with default values
    config = VisionModelSAERunnerConfig()

    # Expected default values
    expected_values = {
        "model_class_name": "HookedViT",
        "model_name": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
        "hook_point_layer": 9,
        "layer_subtype": "hook_resid_post",
        "context_size": 50,
        "device": torch.device("cpu"),
        "seed": 42,
        "dtype": torch.float32,
        "cls_token_only": False,
        "activation_fn_str": "relu",
        "expansion_factor": 16,
        "l1_coefficient": 0.0002,
        # Add any other expected default values here
    }

    # Check that each default attribute matches the expected value
    for attr, expected_value in expected_values.items():
        actual_value = getattr(config, attr)
        assert (
            actual_value == expected_value
        ), f"Default value for '{attr}' does not match expected value"


def test_save_and_load_config(tmp_path):
    """
    Test that saving and loading a configuration works correctly.
    """
    # Create a configuration instance
    config = VisionModelSAERunnerConfig()

    # Save the configuration to a temporary file
    config_path = tmp_path / "config.json"
    config.save_config(str(config_path))

    # Load the configuration from the file
    loaded_config = VisionModelSAERunnerConfig.load_config(str(config_path))

    # Verify that the loaded configuration matches the original
    assert config == loaded_config, "Loaded config does not match the original"

    # Optionally, print the configuration
    config.pretty_print()
