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

import unittest
from unittest.mock import Mock

from kubeflow.trainer.utils import utils
from kubeflow.trainer.types import types


class TestCustomTrainerPythonFileSupport(unittest.TestCase):
    """Test cases for the new python_file and python_args functionality in CustomTrainer."""

    def test_get_trainer_crd_from_custom_trainer_python_file_with_args(self):
        """Test get_trainer_crd_from_custom_trainer with python_file and python_args."""
        runtime = Mock()
        trainer = types.CustomTrainer(
            python_file="train.py",
            python_args=["--epochs", "100", "--batch-size", "32"],
            num_nodes=2,
            resources_per_node={"gpu": "4"},
        )

        result = utils.get_trainer_crd_from_custom_trainer(runtime, trainer)

        self.assertEqual(result.num_nodes, 2)
        self.assertEqual(result.command, ["python"])
        self.assertEqual(result.args, ["train.py", "--epochs", "100", "--batch-size", "32"])

    def test_get_trainer_crd_from_custom_trainer_python_file_no_args(self):
        """Test get_trainer_crd_from_custom_trainer with python_file but no args."""
        runtime = Mock()
        trainer = types.CustomTrainer(
            python_file="train.py", num_nodes=2, resources_per_node={"gpu": "4"}
        )

        result = utils.get_trainer_crd_from_custom_trainer(runtime, trainer)

        self.assertEqual(result.num_nodes, 2)
        self.assertEqual(result.command, ["python"])
        self.assertEqual(result.args, ["train.py"])

    def test_get_trainer_crd_from_custom_trainer_mutual_exclusivity_both_specified(self):
        """Test that func and python_file cannot be specified together."""
        runtime = Mock()
        trainer = types.CustomTrainer(func=lambda: None, python_file="train.py")

        with self.assertRaises(ValueError) as context:
            utils.get_trainer_crd_from_custom_trainer(runtime, trainer)

        self.assertIn(
            "Specify only one of func or python_file in CustomTrainer", str(context.exception)
        )

    def test_get_trainer_crd_from_custom_trainer_mutual_exclusivity_neither_specified(self):
        """Test that either func or python_file must be specified."""
        runtime = Mock()
        trainer = types.CustomTrainer()

        with self.assertRaises(ValueError) as context:
            utils.get_trainer_crd_from_custom_trainer(runtime, trainer)

        self.assertIn(
            "You must specify either func or python_file in CustomTrainer", str(context.exception)
        )

    def test_get_trainer_crd_from_custom_trainer_with_func_unchanged(self):
        """Test that existing func functionality remains unchanged."""
        runtime = Mock()
        runtime.trainer = Mock()
        runtime.trainer.command = ["python", "script.py"]

        def dummy_func():
            pass

        trainer = types.CustomTrainer(
            func=dummy_func, func_args={"lr": 0.001}, num_nodes=2, resources_per_node={"gpu": "4"}
        )

        with unittest.mock.patch(
            "kubeflow.trainer.utils.utils.get_command_using_train_func"
        ) as mock_get_command:
            mock_get_command.return_value = ["python", "script.py"]
            result = utils.get_trainer_crd_from_custom_trainer(runtime, trainer)

        self.assertEqual(result.num_nodes, 2)
        # Verify that the existing func path still works
        mock_get_command.assert_called_once()
