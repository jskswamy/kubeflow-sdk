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