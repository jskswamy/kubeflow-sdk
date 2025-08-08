# Copyright 2024 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from unittest.mock import patch

from kubeflow.trainer.utils import utils
from kubeflow.trainer.types import types


class TestCustomTrainerPythonFileSupport:
    """Test cases for the new python_file and python_args functionality in CustomTrainer."""

    @pytest.mark.parametrize(
        "framework,expected_command,python_file,python_args,num_nodes",
        [
            ("torch", ["torchrun"], "train.py", ["--epochs", "100", "--batch-size", "32"], 2),
            ("plainml", ["python"], "train.py", None, 2),
            (
                "mpi",
                ["mpirun", "--hostfile", "/etc/mpi/hostfile"],
                "train.py",
                ["--epochs", "100"],
                4,
            ),
        ],
    )
    def test_get_trainer_crd_from_custom_trainer_python_file_frameworks(
        self, framework, expected_command, python_file, python_args, num_nodes
    ):
        """Test get_trainer_crd_from_custom_trainer with different framework commands."""
        # Create real Runtime and RuntimeTrainer objects instead of mocking
        trainer = types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework=framework,
            num_nodes=num_nodes,
        )

        # Set the command based on framework
        if framework == "torch":
            trainer.set_command(("torchrun",))
        elif framework == "mpi":
            trainer.set_command(("mpirun", "--hostfile", "/etc/mpi/hostfile"))
        else:  # plainml
            trainer.set_command(("python",))

        runtime = types.Runtime(
            name="test-runtime",
            trainer=trainer,
        )

        custom_trainer = types.CustomTrainer(
            python_file=python_file,
            python_args=python_args,
            num_nodes=num_nodes,
            resources_per_node={"gpu": "4"},
        )

        result = utils.get_trainer_crd_from_custom_trainer(runtime, custom_trainer)

        assert result.num_nodes == num_nodes
        assert result.command == expected_command

        expected_args = [python_file]
        if python_args:
            expected_args.extend(python_args)
        assert result.args == expected_args

    def test_get_trainer_crd_from_custom_trainer_mutual_exclusivity_both_specified(self):
        """Test that func and python_file cannot be specified together."""
        # Create a minimal real runtime
        trainer = types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
        )
        trainer.set_command(("torchrun",))

        runtime = types.Runtime(
            name="test-runtime",
            trainer=trainer,
        )

        custom_trainer = types.CustomTrainer(func=lambda: None, python_file="train.py")

        with pytest.raises(ValueError) as excinfo:
            utils.get_trainer_crd_from_custom_trainer(runtime, custom_trainer)

        assert "Specify only one of func or python_file in CustomTrainer" in str(excinfo.value)

    def test_get_trainer_crd_from_custom_trainer_mutual_exclusivity_neither_specified(self):
        """Test that either func or python_file must be specified."""
        # Create a minimal real runtime
        trainer = types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="torch",
        )
        trainer.set_command(("torchrun",))

        runtime = types.Runtime(
            name="test-runtime",
            trainer=trainer,
        )

        custom_trainer = types.CustomTrainer()

        with pytest.raises(ValueError) as excinfo:
            utils.get_trainer_crd_from_custom_trainer(runtime, custom_trainer)

        assert "You must specify either func or python_file in CustomTrainer" in str(excinfo.value)

    def test_get_trainer_crd_from_custom_trainer_with_func_unchanged(self):
        """Test that existing func functionality remains unchanged."""
        # Create real runtime objects instead of mocking
        trainer = types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="plainml",
        )
        trainer.set_command(("python",))

        runtime = types.Runtime(
            name="test-runtime",
            trainer=trainer,
        )

        def dummy_func():
            pass

        custom_trainer = types.CustomTrainer(
            func=dummy_func, func_args={"lr": 0.001}, num_nodes=2, resources_per_node={"gpu": "4"}
        )

        with patch("kubeflow.trainer.utils.utils.get_command_using_train_func") as mock_get_command:
            mock_get_command.return_value = ["python", "script.py"]
            result = utils.get_trainer_crd_from_custom_trainer(runtime, custom_trainer)

        assert result.num_nodes == 2
        # Verify that the existing func path still works
        mock_get_command.assert_called_once()
