# Copyright 2025 The Kubeflow Authors.
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

import textwrap

import pytest

from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils


def _build_runtime() -> types.Runtime:
    runtime_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="torch",
        device="cpu",
        device_count="1",
    )
    runtime_trainer.set_command(constants.DEFAULT_COMMAND)
    return types.Runtime(name="test-runtime", trainer=runtime_trainer)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="multiple pip index URLs",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://private.repo.com/simple",
                    "https://internal.company.com/simple",
                ],
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "--extra-index-url https://private.repo.com/simple "
                "--extra-index-url https://internal.company.com/simple "
                "torch numpy custom-package\n"
            ),
        ),
        TestCase(
            name="single pip index URL (backward compatibility)",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": ["https://pypi.org/simple"],
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "torch numpy custom-package\n"
            ),
        ),
        TestCase(
            name="multiple pip index URLs with MPI",
            config={
                "packages_to_install": ["torch", "numpy", "custom-package"],
                "pip_index_urls": [
                    "https://pypi.org/simple",
                    "https://private.repo.com/simple",
                    "https://internal.company.com/simple",
                ],
                "is_mpi": True,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                "--no-warn-script-location --index-url https://pypi.org/simple "
                "--extra-index-url https://private.repo.com/simple "
                "--extra-index-url https://internal.company.com/simple "
                "--user torch numpy custom-package\n"
            ),
        ),
        TestCase(
            name="default pip index URLs",
            config={
                "packages_to_install": ["torch", "numpy"],
                "pip_index_urls": constants.DEFAULT_PIP_INDEX_URLS,
                "is_mpi": False,
            },
            expected_output=(
                '\nif ! [ -x "$(command -v pip)" ]; then\n'
                "    python -m ensurepip || python -m ensurepip --user || "
                "apt-get install python-pip\n"
                "fi\n\n"
                "PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet "
                f"--no-warn-script-location --index-url "
                f"{constants.DEFAULT_PIP_INDEX_URLS[0]} torch numpy\n"
            ),
        ),
    ],
)
def test_get_script_for_python_packages(test_case):
    """Test get_script_for_python_packages with various configurations."""

    script = utils.get_script_for_python_packages(
        packages_to_install=test_case.config["packages_to_install"],
        pip_index_urls=test_case.config["pip_index_urls"],
        is_mpi=test_case.config["is_mpi"],
    )

    assert test_case.expected_output == script


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="with args dict always unpacks kwargs",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": {"batch_size": 128, "learning_rate": 0.001, "epochs": 20},
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>(**{'batch_size': 128, 'learning_rate': 0.001, 'epochs': 20})\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
        TestCase(
            name="without args calls function with no params",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": None,
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>()\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
        TestCase(
            name="raises when runtime has no trainer",
            expected_status=FAILED,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": None,
                "runtime": types.Runtime(name="no-trainer", trainer=None),
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="raises when train_func is not callable",
            expected_status=FAILED,
            config={
                "func": "not callable",
                "func_args": None,
                "runtime": _build_runtime(),
            },
            expected_error=ValueError,
        ),
        TestCase(
            name="single dict param also unpacks kwargs",
            expected_status=SUCCESS,
            config={
                "func": (lambda: print("Hello World")),
                "func_args": {"a": 1, "b": 2},
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda: print("Hello World")),\n\n'
                    "<lambda>(**{'a': 1, 'b': 2})\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
        TestCase(
            name="multi-param function uses kwargs-unpacking",
            expected_status=SUCCESS,
            config={
                "func": (lambda **kwargs: "ok"),
                "func_args": {"a": 3, "b": "hi", "c": 0.2},
                "runtime": _build_runtime(),
            },
            expected_output=[
                "bash",
                "-c",
                (
                    "\nread -r -d '' SCRIPT << EOM\n\n"
                    '"func": (lambda **kwargs: "ok"),\n\n'
                    "<lambda>(**{'a': 3, 'b': 'hi', 'c': 0.2})\n\n"
                    "EOM\n"
                    'printf "%s" "$SCRIPT" > "utils_test.py"\n'
                    'python "utils_test.py"'
                ),
            ],
        ),
    ],
)
def test_get_command_using_train_func(test_case: TestCase):
    print("Executing test:", test_case.name)

    try:
        command = utils.get_command_using_train_func(
            runtime=test_case.config["runtime"],
            train_func=test_case.config.get("func"),
            train_func_parameters=test_case.config.get("func_args"),
            pip_index_urls=constants.DEFAULT_PIP_INDEX_URLS,
            packages_to_install=[],
        )

        assert test_case.expected_status == SUCCESS
        assert command == test_case.expected_output

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")

def _build_plain_runtime() -> types.Runtime:
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="plainml",
        num_nodes=1,
    )
    trainer.set_command(constants.DEFAULT_COMMAND)
    return types.Runtime(name="test-runtime", trainer=trainer)

def _build_mpi_runtime() -> types.Runtime:
    trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="mpi",
        num_nodes=2,
    )
    trainer.set_command(constants.MPI_COMMAND)
    return types.Runtime(name="mpi-runtime", trainer=trainer)


class TestGetCommandUsingUserCommand:
    def test_plain_with_installs(self):
        runtime = _build_plain_runtime()
        command = ["python"]
        command_args = ["train.py", "--epochs", "2"]
        pip_index_urls = [
            "https://pypi.org/simple",
            "https://private.repo.com/simple",
        ]
        packages_to_install = ["torch", "numpy"]

        result = utils.get_command_using_user_command(
            runtime=runtime,
            command=command,
            command_args=command_args,
            pip_index_urls=pip_index_urls,
            packages_to_install=packages_to_install,
        )

        expected = textwrap.dedent(
            """bash
-c

if ! [ -x "$(command -v pip)" ]; then
    python -m ensurepip || python -m ensurepip --user || apt-get install python-pip
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet --no-warn-script-location --index-url https://pypi.org/simple --extra-index-url https://private.repo.com/simple torch numpy
python train.py --epochs 2"""
        )
        assert "\n".join(result) == expected

    def test_plain_with_installs_and_pip_extra_args(self):
        runtime = _build_plain_runtime()
        command = ["python"]
        command_args = ["train.py"]
        pip_index_urls = [
            "https://pypi.org/simple",
        ]
        packages_to_install = ["torch"]

        result = utils.get_command_using_user_command(
            runtime=runtime,
            command=command,
            command_args=command_args,
            pip_index_urls=pip_index_urls,
            packages_to_install=packages_to_install,
            pip_extra_args=["--no-cache-dir", "--find-links", "/wheels"],
        )

        joined = "\n".join(result)
        assert "--no-warn-script-location --index-url https://pypi.org/simple torch --no-cache-dir --find-links /wheels" in joined


class TestGetTrainerCRDFromCommandTrainer:
    def test_plain_runtime_builds_crd_with_env_and_resources(self):
        runtime = _build_plain_runtime()
        trainer = types.CommandTrainer(
            command=["python"],
            args=["main.py", "--epochs", "3"],
            packages_to_install=["numpy"],
            pip_index_urls=["https://pypi.org/simple"],
            num_nodes=2,
            resources_per_node={"gpu": "1"},
            env={"FOO": "bar"},
        )

        crd = utils.get_trainer_crd_from_command_trainer(runtime, trainer)
        expected_command = utils.get_command_using_user_command(
            runtime=runtime,
            command=trainer.command,
            command_args=trainer.args,
            pip_index_urls=trainer.pip_index_urls,
            packages_to_install=trainer.packages_to_install,
        )

        assert crd.num_nodes == 2
        assert crd.command == expected_command
        assert any(ev.name == "FOO" and ev.value == "bar" for ev in (crd.env or []))
        assert crd.resources_per_node is not None

    def test_mpi_runtime_builds_crd_uses_user_flag(self):
        runtime = _build_mpi_runtime()
        trainer = types.CommandTrainer(
            command=["python"],
            args=["train.py"],
            packages_to_install=["torch"],
            pip_index_urls=["https://pypi.org/simple"],
            num_nodes=4,
        )

        crd = utils.get_trainer_crd_from_command_trainer(runtime, trainer)

        expected_command = utils.get_command_using_user_command(
            runtime=runtime,
            command=trainer.command,
            command_args=trainer.args,
            pip_index_urls=trainer.pip_index_urls,
            packages_to_install=trainer.packages_to_install,
        )

        assert crd.num_nodes == 4
        assert crd.command == expected_command

    def test_defaults_to_bash_when_command_missing(self):
        runtime = _build_plain_runtime()
        trainer = types.CommandTrainer(
            args=["-lc", "echo hello"],
        )
        crd = utils.get_trainer_crd_from_command_trainer(runtime, trainer)
        expected_command = utils.get_command_using_user_command(
            runtime=runtime,
            command=[],
            command_args=["-lc", "echo hello"],
            pip_index_urls=trainer.pip_index_urls,
            packages_to_install=trainer.packages_to_install,
        )
        assert crd.command == expected_command

    def test_always_bash_wrapped_even_without_installs(self):
        runtime = _build_plain_runtime()
        trainer = types.CommandTrainer(
            command=["python"],
            args=["main.py"],
        )
        crd = utils.get_trainer_crd_from_command_trainer(runtime, trainer)
        # Should wrap into bash -c preserving python main.py
        assert crd.command == ["bash", "-c", "python main.py"]

    def test_preserves_prefix_plain(self):
        runtime = _build_plain_runtime()

        result = utils.get_command_using_user_command(
            runtime=runtime,
            command=["python"],
            command_args=["main.py"],
            pip_index_urls=[constants.DEFAULT_PIP_INDEX_URLS[0]],
            packages_to_install=None,
        )

        expected = "bash\n-c\npython main.py"
        assert "\n".join(result) == expected

    def test_preserves_prefix_mpi_and_user_flag(self):
        runtime = _build_mpi_runtime()

        result = utils.get_command_using_user_command(
            runtime=runtime,
            command=["python"],
            command_args=["train.py"],
            pip_index_urls=["https://pypi.org/simple"],
            packages_to_install=["torch"],
        )

        expected = textwrap.dedent(
            """mpirun
--hostfile
/etc/mpi/hostfile
bash
-c

if ! [ -x "$(command -v pip)" ]; then
    python -m ensurepip || python -m ensurepip --user || apt-get install python-pip
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --quiet --no-warn-script-location --index-url https://pypi.org/simple --user torch
python train.py"""
        )
        assert "\n".join(result) == expected

    def test_fallback_when_no_runtime_command(self):
        trainer = types.RuntimeTrainer(
            trainer_type=types.TrainerType.CUSTOM_TRAINER,
            framework="plainml",
            num_nodes=1,
        )
        # Explicitly set launcher for this runtime
        trainer.set_command(constants.DEFAULT_COMMAND)
        runtime = types.Runtime(name="with-launcher", trainer=trainer)

        result = utils.get_command_using_user_command(
            runtime=runtime,
            command=["echo"],
            command_args=["hello"],
            pip_index_urls=[constants.DEFAULT_PIP_INDEX_URLS[0]],
            packages_to_install=None,
        )

        expected = textwrap.dedent(
            """bash
-c
echo hello"""
        )
        assert "\n".join(result) == expected