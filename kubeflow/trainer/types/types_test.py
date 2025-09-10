from kubeflow.trainer.types import types


class TestCommandTrainerType:
    def test_command_trainer_dataclass_minimal(self):
        trainer = types.CommandTrainer(command=["python"], args=["train.py"])

        assert trainer.command == ["python"]
        assert trainer.args == ["train.py"]
        assert trainer.pip_index_urls and isinstance(trainer.pip_index_urls, list)
        assert trainer.packages_to_install is None
        assert trainer.env is None
