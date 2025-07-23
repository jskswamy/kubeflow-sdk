import pytest
from unittest.mock import Mock, patch
from kubeflow.trainer.utils import utils
from kubeflow.trainer.types import types
from kubeflow.trainer.constants import constants


class TestTrainerDetection:
    """Test cases for trainer detection logic."""

    @pytest.mark.parametrize(
        "image_name,expected_framework",
        [
            # Known images from ALL_TRAINERS
            ("pytorch/pytorch", types.Framework.TORCH),
            ("ghcr.io/kubeflow/trainer/mlx-runtime", types.Framework.MLX),
            (
                "ghcr.io/kubeflow/trainer/deepspeed-runtime",
                types.Framework.DEEPSPEED,
            ),
            (
                "ghcr.io/kubeflow/trainer/torchtune-trainer",
                types.Framework.TORCHTUNE,
            ),
            # Custom images with pattern matching - lowercase
            ("my-org/deepspeed-custom:latest", types.Framework.DEEPSPEED),
            ("custom-mlx-runtime:v1.0", types.Framework.MLX),
            ("pytorch-training:latest", types.Framework.TORCH),
            ("torchtune-finetuning:latest", types.Framework.TORCHTUNE),
            # Custom images with pattern matching - uppercase
            ("my-org/DeepSpeed-Custom:latest", types.Framework.DEEPSPEED),
            ("custom-MLX-runtime:v1.0", types.Framework.MLX),
            ("PyTorch-training:latest", types.Framework.TORCH),
            ("TorchTune-finetuning:latest", types.Framework.TORCHTUNE),
            # Custom images with pattern matching - mixed case
            ("my-org/DeepSpeed-custom:latest", types.Framework.DEEPSPEED),
            ("custom-Mlx-runtime:v1.0", types.Framework.MLX),
            ("PyTorch-Training:latest", types.Framework.TORCH),
            ("TorchTune-Finetuning:latest", types.Framework.TORCHTUNE),
            # Custom images with pattern matching - all caps
            ("my-org/DEEPSPEED-CUSTOM:latest", types.Framework.DEEPSPEED),
            ("custom-MLX-RUNTIME:v1.0", types.Framework.MLX),
            ("PYTORCH-TRAINING:latest", types.Framework.TORCH),
            ("TORCHTUNE-FINETUNING:latest", types.Framework.TORCHTUNE),
            # Edge cases - partial matches
            ("my-deepspeed-runtime:latest", types.Framework.DEEPSPEED),
            ("mlx-custom:latest", types.Framework.MLX),
            ("pytorch-torch-custom:latest", types.Framework.TORCH),
            ("pytorch-custom:latest", types.Framework.TORCH),
            ("torchtune-custom:latest", types.Framework.TORCHTUNE),
            # Edge cases - with numbers and special characters
            ("deepspeed-v2.1:latest", types.Framework.DEEPSPEED),
            ("mlx_runtime_1.0:latest", types.Framework.MLX),
            ("pytorch_2.0_cuda:latest", types.Framework.TORCH),
            ("torchtune-llama-3b:latest", types.Framework.TORCHTUNE),
            # Edge cases - with registry prefixes
            ("docker.io/myorg/deepspeed:latest", types.Framework.DEEPSPEED),
            ("ghcr.io/myorg/mlx-runtime:latest", types.Framework.MLX),
            ("quay.io/myorg/pytorch-training:latest", types.Framework.TORCH),
            (
                "registry.example.com/myorg/torchtune:latest",
                types.Framework.TORCHTUNE,
            ),
            # Edge cases - with ports and complex paths
            (
                "registry.example.com:5000/myorg/deepspeed:latest",
                types.Framework.DEEPSPEED,
            ),
            ("ghcr.io/myorg/mlx/runtime:v1.0", types.Framework.MLX),
            ("docker.io/myorg/pytorch/training:latest", types.Framework.TORCH),
            (
                "quay.io/myorg/torchtune/finetuning:latest",
                types.Framework.TORCHTUNE,
            ),
            # Edge cases - no match (including generic torch without pytorch)
            (
                "torch-custom:latest",
                None,
            ),  # Generic torch should not match (requires pytorch)
            ("unknown-image:latest", None),
            ("", None),
            ("nginx:latest", None),
            ("ubuntu:20.04", None),
        ],
    )
    def test_trainer_detection_from_image_patterns(
        self, image_name, expected_framework
    ):
        """Test trainer detection using image pattern matching with various case scenarios."""
        trainer = utils.get_trainer_from_image(image_name)

        if expected_framework is None:
            # When no pattern matches, should return DEFAULT_TRAINER (PyTorch)
            assert trainer.framework == types.Framework.TORCH
        else:
            assert trainer.framework == expected_framework

    @pytest.mark.parametrize(
        "image_name,expected_framework",
        [
            # Official Kubeflow images (should be detected by regex)
            ("pytorch/pytorch", types.Framework.TORCH),
            ("ghcr.io/kubeflow/trainer/mlx-runtime", types.Framework.MLX),
            (
                "ghcr.io/kubeflow/trainer/deepspeed-runtime",
                types.Framework.DEEPSPEED,
            ),
            (
                "ghcr.io/kubeflow/trainer/torchtune-trainer",
                types.Framework.TORCHTUNE,
            ),
            # Custom images with pattern matching - various cases
            ("my-deepspeed-runtime:latest", types.Framework.DEEPSPEED),
            ("custom-pytorch:latest", types.Framework.TORCH),
            ("mlx-custom:latest", types.Framework.MLX),
            ("torchtune-custom:latest", types.Framework.TORCHTUNE),
            ("DeepSpeed-Custom:latest", types.Framework.DEEPSPEED),
            ("PyTorch-Custom:latest", types.Framework.TORCH),
            ("MLX-Custom:latest", types.Framework.MLX),
            ("TorchTune-Custom:latest", types.Framework.TORCHTUNE),
            # Fallback to default
            ("completely-unknown:latest", types.Framework.TORCH),
            ("nginx:latest", types.Framework.TORCH),
        ],
    )
    def test_trainer_detection_precedence(self, image_name, expected_framework):
        """Test the trainer detection logic with pattern matching and fallback."""
        # Create mock trainer container
        trainer_container = Mock()
        trainer_container.image = image_name

        trainer = utils.detect_trainer(trainer_container)
        assert trainer is not None
        assert trainer.framework == expected_framework

    def test_official_kubeflow_images_detected_by_regex(self):
        """Test that official Kubeflow trainer images are correctly detected by regex patterns."""
        # Official Kubeflow images that should be detected by regex patterns
        official_images = [
            ("pytorch/pytorch", types.Framework.TORCH),
            ("ghcr.io/kubeflow/trainer/mlx-runtime", types.Framework.MLX),
            ("ghcr.io/kubeflow/trainer/deepspeed-runtime", types.Framework.DEEPSPEED),
            ("ghcr.io/kubeflow/trainer/torchtune-trainer", types.Framework.TORCHTUNE),
        ]

        for image_name, expected_framework in official_images:
            trainer = utils.get_trainer_from_image(image_name)
            assert trainer is not None, (
                f"Failed to detect trainer for official Kubeflow image: {image_name}"
            )
            assert trainer.framework == expected_framework, (
                f"Wrong framework detected for {image_name}: got {trainer.framework}, expected {expected_framework}"
            )

    def test_returns_default_trainer_when_no_pattern_matches(self):
        """Test that function returns DEFAULT_TRAINER when no pattern matches."""
        trainer = utils.get_trainer_from_image("unknown-image:latest")
        assert trainer is not None
        assert trainer.framework == types.Framework.TORCH  # DEFAULT_TRAINER is PyTorch

    def test_returns_deep_copy_of_default_trainer(self):
        """Test that function returns a deep copy of DEFAULT_TRAINER when no pattern matches."""
        trainer1 = utils.get_trainer_from_image("unknown-image-1:latest")
        trainer2 = utils.get_trainer_from_image("unknown-image-2:latest")

        assert trainer1 is not None
        assert trainer2 is not None
        assert trainer1.framework == types.Framework.TORCH
        assert trainer2.framework == types.Framework.TORCH
        # Verify they are different objects (deep copies)
        assert trainer1 is not trainer2

    def test_pattern_matching_takes_precedence_over_default(self):
        """Test that pattern matching takes precedence over default fallback."""
        trainer = utils.get_trainer_from_image("deepspeed-custom:latest")
        assert trainer is not None
        assert trainer.framework == types.Framework.DEEPSPEED  # Pattern match wins
        assert trainer.framework != types.Framework.TORCH  # Not the default


class TestAcceleratorCountLogic:
    """Test cases for accelerator count logic in get_runtime_trainer."""

    @pytest.mark.parametrize(
        "ml_policy_config,expected_accelerator_count",
        [
            # Torch policies with different num_proc_per_node values
            ({"torch": {"num_proc_per_node": 4}}, 4),
            ({"torch": {"num_proc_per_node": 8}}, 8),
            (
                {"torch": {"num_proc_per_node": "auto"}},
                None,
            ),  # String values should not set accelerator count
            (
                {"torch": {"num_proc_per_node": "gpu"}},
                None,
            ),  # String values should not set accelerator count
            (
                {"torch": {"num_proc_per_node": "cpu"}},
                None,
            ),  # String values should not set accelerator count
            # MPI policies with different num_proc_per_node values
            ({"mpi": {"num_proc_per_node": 2}}, 2),
            ({"mpi": {"num_proc_per_node": 16}}, 16),
            ({"mpi": {"num_proc_per_node": 1}}, 1),
            # No policies
            ({}, None),
            ({"torch": {}}, None),
            ({"mpi": {}}, None),
        ],
    )
    def test_accelerator_count_from_ml_policy(
        self, ml_policy_config, expected_accelerator_count
    ):
        """Test that accelerator count is correctly set from ML policy."""
        with patch.object(
            utils, "get_container_devices", return_value=None
        ) as mock_get_devices:
            # Create mock replicated jobs with proper structure
            mock_container = Mock()
            mock_container.image = "pytorch/pytorch:latest"
            mock_resources = Mock()
            mock_resources.limits = None
            mock_container.resources = mock_resources
            mock_container.name = constants.NODE

            mock_replicated_job = Mock()
            mock_replicated_job.template = Mock()
            mock_replicated_job.template.spec = Mock()
            mock_replicated_job.template.spec.template = Mock()
            mock_replicated_job.template.spec.template.spec = Mock()
            mock_replicated_job.template.spec.template.spec.containers = [
                mock_container
            ]
            mock_replicated_job.template.metadata = Mock()
            mock_replicated_job.template.metadata.labels = {
                constants.TRAINJOB_ANCESTOR_LABEL: "trainer"
            }
            replicated_jobs = [mock_replicated_job]

            # Create mock ML policy
            ml_policy = Mock()
            ml_policy.num_nodes = None

            if "torch" in ml_policy_config:
                ml_policy.torch = Mock()
                if "num_proc_per_node" in ml_policy_config["torch"]:
                    mock_nppp_obj = Mock()
                    mock_nppp_obj.actual_instance = ml_policy_config["torch"][
                        "num_proc_per_node"
                    ]
                    ml_policy.torch.num_proc_per_node = mock_nppp_obj
                else:
                    ml_policy.torch.num_proc_per_node = None  # Explicitly None
            else:
                ml_policy.torch = None

            if "mpi" in ml_policy_config:
                ml_policy.mpi = Mock()
                if "num_proc_per_node" in ml_policy_config["mpi"]:
                    ml_policy.mpi.num_proc_per_node = ml_policy_config["mpi"][
                        "num_proc_per_node"
                    ]
                else:
                    ml_policy.mpi.num_proc_per_node = None  # Explicitly None
            else:
                ml_policy.mpi = None

            # Create mock runtime metadata
            runtime_metadata = Mock()
            runtime_metadata.labels = {}

            # Call the function
            trainer = utils.get_runtime_trainer(
                replicated_jobs, ml_policy, runtime_metadata
            )

            # Check accelerator count
            if expected_accelerator_count is not None:
                assert trainer.accelerator_count == expected_accelerator_count
            else:
                assert trainer.accelerator_count == constants.UNKNOWN

    @pytest.mark.parametrize(
        "ml_policy_config,num_nodes,expected_accelerator_count",
        [
            # Torch with num_nodes
            ({"torch": {"num_proc_per_node": 4}}, 2, 8),  # 4 * 2 = 8
            ({"torch": {"num_proc_per_node": 8}}, 3, 24),  # 8 * 3 = 24
            # MPI with num_nodes
            ({"mpi": {"num_proc_per_node": 2}}, 4, 8),  # 2 * 4 = 8
            ({"mpi": {"num_proc_per_node": 16}}, 2, 32),  # 16 * 2 = 32
            # String values should not be multiplied
            ({"torch": {"num_proc_per_node": "auto"}}, 2, None),
            ({"torch": {"num_proc_per_node": "gpu"}}, 3, None),
        ],
    )
    def test_accelerator_count_with_num_nodes(
        self, ml_policy_config, num_nodes, expected_accelerator_count
    ):
        """Test that accelerator count is correctly multiplied by number of nodes."""
        with patch.object(
            utils, "get_container_devices", return_value=None
        ) as mock_get_devices:
            # Create mock replicated jobs with proper structure
            mock_container = Mock()
            mock_container.image = "pytorch/pytorch:latest"
            mock_resources = Mock()
            mock_resources.limits = None
            mock_container.resources = mock_resources
            mock_container.name = constants.NODE
            mock_replicated_job = Mock()
            mock_replicated_job.template = Mock()
            mock_replicated_job.template.spec = Mock()
            mock_replicated_job.template.spec.template = Mock()
            mock_replicated_job.template.spec.template.spec = Mock()
            mock_replicated_job.template.spec.template.spec.containers = [
                mock_container
            ]
            mock_replicated_job.template.metadata = Mock()
            mock_replicated_job.template.metadata.labels = {
                constants.TRAINJOB_ANCESTOR_LABEL: "trainer"
            }
            replicated_jobs = [mock_replicated_job]

            # Create mock ML policy
            ml_policy = Mock()
            ml_policy.num_nodes = num_nodes  # Use the num_nodes parameter

            if "torch" in ml_policy_config:
                ml_policy.torch = Mock()
                if "num_proc_per_node" in ml_policy_config["torch"]:
                    mock_nppp_obj = Mock()
                    mock_nppp_obj.actual_instance = ml_policy_config["torch"][
                        "num_proc_per_node"
                    ]
                    ml_policy.torch.num_proc_per_node = mock_nppp_obj
                else:
                    ml_policy.torch.num_proc_per_node = None  # Explicitly None
            else:
                ml_policy.torch = None

            if "mpi" in ml_policy_config:
                ml_policy.mpi = Mock()
                if "num_proc_per_node" in ml_policy_config["mpi"]:
                    ml_policy.mpi.num_proc_per_node = ml_policy_config["mpi"][
                        "num_proc_per_node"
                    ]
                else:
                    ml_policy.mpi.num_proc_per_node = None  # Explicitly None
            else:
                ml_policy.mpi = None

            # Create mock runtime metadata
            runtime_metadata = Mock()
            runtime_metadata.labels = {}

            # Call the function
            trainer = utils.get_runtime_trainer(
                replicated_jobs, ml_policy, runtime_metadata
            )

            # Check accelerator count
            if expected_accelerator_count is not None:
                assert trainer.accelerator_count == expected_accelerator_count
            else:
                assert trainer.accelerator_count == constants.UNKNOWN

    def test_accelerator_count_precedence(self):
        """Test that torch policy takes precedence over mpi policy for accelerator count."""
        with patch.object(
            utils, "get_container_devices", return_value=None
        ) as mock_get_devices:
            # Create mock replicated jobs with proper structure
            mock_container = Mock()
            mock_container.image = "pytorch/pytorch:latest"
            mock_resources = Mock()
            mock_resources.limits = None
            mock_container.resources = mock_resources
            mock_container.name = constants.NODE

            mock_replicated_job = Mock()
            mock_replicated_job.template = Mock()
            mock_replicated_job.template.spec = Mock()
            mock_replicated_job.template.spec.template = Mock()
            mock_replicated_job.template.spec.template.spec = Mock()
            mock_replicated_job.template.spec.template.spec.containers = [
                mock_container
            ]
            mock_replicated_job.template.metadata = Mock()
            mock_replicated_job.template.metadata.labels = {
                constants.TRAINJOB_ANCESTOR_LABEL: "trainer"
            }
            replicated_jobs = [mock_replicated_job]

            # Create mock ML policy with both torch and mpi
            ml_policy = Mock()
            ml_policy.torch = Mock()
            mock_nppp_torch = Mock()
            mock_nppp_torch.actual_instance = 4  # Should take precedence
            ml_policy.torch.num_proc_per_node = mock_nppp_torch

            ml_policy.mpi = Mock()
            ml_policy.mpi.num_proc_per_node = 8  # Should be ignored

            ml_policy.num_nodes = None

            # Create mock runtime metadata
            runtime_metadata = Mock()
            runtime_metadata.labels = {}

            # Call the function
            trainer = utils.get_runtime_trainer(
                replicated_jobs, ml_policy, runtime_metadata
            )

            # Torch policy should take precedence
            assert trainer.accelerator_count == 4
