"""Tests for ProcessComponent base class."""

import pytest
import torch
import json
from pathlib import Path
from unittest.mock import patch

from dcoupler.wrappers.process import ProcessComponent, FiniteDifferenceProcess
from dcoupler.core.component import (
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)


class MockProcess(ProcessComponent):
    """Mock process component that reads/writes JSON files."""

    @property
    def input_fluxes(self):
        return [
            FluxSpec("precip", "mm/d", FluxDirection.INPUT, "hru", 86400, ("time",)),
        ]

    @property
    def output_fluxes(self):
        return [
            FluxSpec("runoff", "mm/d", FluxDirection.OUTPUT, "hru", 86400, ("time",),
                     conserved_quantity="water_mass"),
        ]

    def write_inputs(self, inputs, work_dir):
        data = {k: v.tolist() for k, v in inputs.items()}
        with open(work_dir / "inputs.json", "w") as f:
            json.dump(data, f)

    def execute(self, work_dir):
        with open(work_dir / "inputs.json") as f:
            data = json.load(f)
        precip = data["precip"]
        # Simple model: runoff = 0.5 * precip
        runoff = [0.5 * p for p in precip]
        with open(work_dir / "outputs.json", "w") as f:
            json.dump({"runoff": runoff}, f)
        return 0  # success

    def read_outputs(self, work_dir):
        with open(work_dir / "outputs.json") as f:
            data = json.load(f)
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}


class FailingProcess(ProcessComponent):
    """Process component that always fails."""

    @property
    def input_fluxes(self):
        return [FluxSpec("x", "m", FluxDirection.INPUT, "hru", 1, ("time",))]

    @property
    def output_fluxes(self):
        return [FluxSpec("y", "m", FluxDirection.OUTPUT, "hru", 1, ("time",))]

    def write_inputs(self, inputs, work_dir):
        pass

    def execute(self, work_dir):
        return 1  # failure

    def read_outputs(self, work_dir):
        return {}


class TestProcessComponent:
    def test_requires_batch(self):
        proc = MockProcess("mock")
        assert proc.requires_batch is True

    def test_step_raises(self):
        proc = MockProcess("mock")
        with pytest.raises(RuntimeError, match="batch execution"):
            proc.step({}, torch.empty(0), 1.0)

    def test_run_write_execute_read_cycle(self, tmp_path):
        proc = MockProcess("mock", work_dir=tmp_path)
        precip = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        inputs = {"precip": precip}
        state = torch.empty(0)

        outputs, new_state = proc.run(inputs, state, 86400, 5)

        assert "runoff" in outputs
        expected = precip * 0.5
        assert torch.allclose(outputs["runoff"], expected)

    def test_failed_execution_raises(self, tmp_path):
        proc = FailingProcess("failing", work_dir=tmp_path)
        with pytest.raises(RuntimeError, match="exit code 1"):
            proc.run({"x": torch.tensor([1.0])}, torch.empty(0), 1.0, 1)

    def test_no_torch_parameters(self):
        proc = MockProcess("mock")
        assert proc.get_torch_parameters() == []
        assert proc.get_physical_parameters() == {}

    def test_gradient_method_none(self):
        proc = MockProcess("mock")
        assert proc.gradient_method == GradientMethod.NONE

    def test_bmi_lifecycle(self, tmp_path):
        proc = MockProcess("mock", work_dir=tmp_path)
        proc.bmi_initialize({})

        result = proc.bmi_update_batch(
            {"precip": [1.0, 2.0, 3.0]}, dt=86400, n_timesteps=3
        )
        assert "runoff" in result

        state = proc.bmi_get_state()
        proc.bmi_finalize()

    def test_bmi_update_raises_for_process(self, tmp_path):
        proc = MockProcess("mock", work_dir=tmp_path)
        proc.bmi_initialize({})
        with pytest.raises(RuntimeError, match="batch execution"):
            proc.bmi_update({"precip": 1.0}, dt=86400)

    def test_bmi_var_names(self):
        proc = MockProcess("mock")
        assert proc.bmi_get_output_var_names() == ["runoff"]
        assert proc.bmi_get_input_var_names() == ["precip"]

    def test_bmi_get_value(self, tmp_path):
        proc = MockProcess("mock", work_dir=tmp_path)
        proc.bmi_initialize({})
        proc.bmi_update_batch({"precip": [1.0, 2.0]}, dt=86400, n_timesteps=2)
        val = proc.bmi_get_value("runoff")
        assert val is not None

    def test_bmi_get_value_unknown_raises(self):
        proc = MockProcess("mock")
        with pytest.raises(KeyError):
            proc.bmi_get_value("nonexistent")
