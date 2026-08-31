from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qairt_agent.contracts import WorkflowSpec
from qairt_agent.families import presets
from qairt_agent.vectors import VectorPreparer

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "tools" / "make_smoke_fixture.py"


def _generate(destination: Path) -> dict[str, str]:
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--output-dir",
            str(destination),
            "--artifacts-root",
            str(destination.parent / "artifacts"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_documented_first_run_fixture_still_generates(tmp_path: Path) -> None:
    # docs/first-run.md sends a newcomer straight here, and the repository
    # commits no model payload, so this generator is the only runnable entry
    # point. If it breaks, a fresh clone can run nothing at all.
    produced = _generate(tmp_path / "smoke")

    for key in ("model", "encodings", "vector_manifest", "spec", "layer_debug_spec"):
        assert Path(produced[key]).is_file(), f"{key} was not written"


def test_the_generated_spec_resolves_to_a_verified_target(tmp_path: Path) -> None:
    # A spec that generates but does not resolve would send the newcomer to a
    # failure with nothing to compare against.
    produced = _generate(tmp_path / "smoke")
    spec = json.loads(Path(produced["spec"]).read_text())

    resolved = presets.to_build_spec(WorkflowSpec.model_validate(spec))

    assert resolved.family.value == "vit"
    assert resolved.target.chipset
    assert resolved.target.soc_model is not None
    # Artifacts must not land under the models directory: the worker mounts it
    # read-only and the build dies with EROFS.
    assert "models" not in Path(spec["output_root"]).parts[-2:]


def test_the_fixture_is_byte_identical_across_machines(tmp_path: Path) -> None:
    # The generator exists so an acceptance result recorded in docs/plan/ can be
    # reproduced by someone else; that only holds if the inputs hash the same.
    first = _generate(tmp_path / "a")
    second = _generate(tmp_path / "b")

    for key in ("model", "encodings"):
        assert (
            Path(first[key]).read_bytes() == Path(second[key]).read_bytes()
        ), f"{key} is not deterministic"

    left = VectorPreparer.load_manifest(first["vector_manifest"])
    right = VectorPreparer.load_manifest(second["vector_manifest"])
    assert left.inputs["input"].sha256 == right.inputs["input"].sha256
    assert left.goldens["output"].sha256 == right.goldens["output"].sha256


def test_the_golden_is_the_float_graphs_own_output(tmp_path: Path) -> None:
    # Validation compares the device against this golden, so a golden that did
    # not come from the graph would make every SQNR number meaningless.
    produced = _generate(tmp_path / "smoke")
    manifest_path = Path(produced["vector_manifest"])
    manifest = VectorPreparer.load_manifest(manifest_path)
    tensors = VectorPreparer.load_tensors(manifest_path, section="inputs")
    goldens = VectorPreparer.load_tensors(manifest_path, section="goldens")

    onnxruntime = __import__("onnxruntime")
    session = onnxruntime.InferenceSession(
        produced["model"], providers=["CPUExecutionProvider"]
    )
    expected = session.run(["output"], {"input": tensors["input"]})[0]

    np.testing.assert_allclose(goldens["output"], expected, rtol=1e-5, atol=1e-6)
    assert manifest.goldens["output"].shape == (1, 32)


def test_the_layer_debug_spec_actually_asks_for_a_drilldown(tmp_path: Path) -> None:
    # The layer reference fails closed without a build that emitted diagnostic
    # contexts, so the generated debug spec must turn that on itself.
    produced = _generate(tmp_path / "smoke")
    debug = json.loads(Path(produced["layer_debug_spec"]).read_text())

    assert debug["compile"]["enable_intermediate_outputs"] is True
    float_reference = debug["stage_configs"]["validation"]["float_reference"]
    assert float_reference["granularity"] == "layer"
    assert float_reference["ar"] == 1
    assert Path(float_reference["model_path"]).is_file()


def test_the_fixture_target_defaults_to_the_active_harness_target(
    tmp_path: Path,
) -> None:
    """A first run must not silently plan for a chip the harness is not on.

    The generator used to hardcode its default target, so flipping
    `harness/constraints.json` to another chip left the fixture planning for
    the old one -- with everything downstream looking perfectly healthy.
    """

    constraints = json.loads(
        (REPO_ROOT / "harness" / "constraints.json").read_text(encoding="utf-8")
    )
    produced = _generate(tmp_path / "smoke")

    for key in ("spec", "layer_debug_spec"):
        spec = json.loads(Path(produced[key]).read_text())
        assert spec["target"]["name"] == constraints["target"]["name"]
