from kubeflow.trainer.types import types


class TestTrainerConfigurations:
    """Test cases for trainer configurations and types."""

    def test_centralized_trainer_configs(self):
        """Test that centralized trainer configurations are properly defined."""
        # Verify all trainer frameworks have configurations
        for framework in types.Framework:
            assert framework in types.TRAINER_CONFIGS
            trainer = types.TRAINER_CONFIGS[framework]
            assert trainer.framework == framework

    def test_default_trainer_uses_centralized_config(self):
        """Test that DEFAULT_TRAINER uses centralized configuration."""
        assert types.DEFAULT_TRAINER == types.TRAINER_CONFIGS[types.Framework.TORCH]
        assert types.DEFAULT_TRAINER.framework == types.Framework.TORCH

    def test_custom_trainer_python_file_with_args(self):
        """Test CustomTrainer with python_file and python_args."""
        # Test basic python_file without args
        trainer = types.CustomTrainer(python_file="train.py")
        assert trainer.python_file == "train.py"
        assert trainer.python_args is None

        # Test python_file with args
        trainer = types.CustomTrainer(
            python_file="train.py",
            python_args=["--epochs", "100", "--batch-size", "32"]
        )
        assert trainer.python_file == "train.py"
        assert trainer.python_args == ["--epochs", "100", "--batch-size", "32"]

        # Test python_file with complex args
        trainer = types.CustomTrainer(
            python_file="train.py",
            python_args=["--epochs", "100", "--batch-size", "32", "--lr", "0.001", "--model-path", "/workspace/model"]
        )
        assert trainer.python_file == "train.py"
        assert trainer.python_args == ["--epochs", "100", "--batch-size", "32", "--lr", "0.001", "--model-path", "/workspace/model"]

    def test_custom_trainer_mutual_exclusivity(self):
        """Test that func and python_file are mutually exclusive."""
        # This should work
        trainer = types.CustomTrainer(python_file="train.py")
        assert trainer.func is None
        assert trainer.python_file == "train.py"

        # This should work
        def dummy_func():
            pass
        trainer = types.CustomTrainer(func=dummy_func)
        assert trainer.func == dummy_func
        assert trainer.python_file is None

    def test_custom_trainer_python_args_only(self):
        """Test CustomTrainer with python_args but no python_file (should be None)."""
        trainer = types.CustomTrainer(python_args=["--epochs", "100"])
        assert trainer.python_file is None
        assert trainer.python_args == ["--epochs", "100"]

    def test_custom_trainer_python_args_with_func(self):
        """Test CustomTrainer with func and python_args (should be allowed)."""
        def dummy_func():
            pass

        trainer = types.CustomTrainer(
            func=dummy_func,
            python_args=["--epochs", "100"]
        )
        assert trainer.func == dummy_func
        assert trainer.python_file is None
        assert trainer.python_args == ["--epochs", "100"]
