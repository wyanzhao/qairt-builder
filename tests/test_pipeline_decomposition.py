"""The decomposition's invariants, kept by a test rather than by discipline.

`pipeline.py` was 8,285 lines with ~110 methods on one class, and 55 inline
family-string mentions carrying routing logic inside stage bodies. Nothing
stopped it growing back except intent, so the shape is asserted here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "qairt_agent"
STAGES = SRC / "pipeline_stages"

#: The facade delegates; it does not hold stage bodies.
FACADE_LINE_BUDGET = 1500
#: A stage module you can still read end to end.
STAGE_LINE_BUDGET = 1300


def test_the_facade_stays_a_facade() -> None:
    lines = len((SRC / "pipeline.py").read_text(encoding="utf-8").splitlines())
    assert lines <= FACADE_LINE_BUDGET, (
        f"pipeline.py is {lines} lines; extract the next stage into "
        "pipeline_stages/ rather than growing the facade"
    )


@pytest.mark.parametrize(
    "module", sorted(p.name for p in STAGES.glob("*.py") if p.name != "__init__.py")
)
def test_no_stage_module_outgrows_a_reading(module: str) -> None:
    lines = len((STAGES / module).read_text(encoding="utf-8").splitlines())
    assert lines <= STAGE_LINE_BUDGET, (
        f"{module} is {lines} lines; split the concern it grew"
    )


@pytest.mark.parametrize(
    "path",
    sorted(
        [SRC / "pipeline.py"]
        + [p for p in STAGES.glob("*.py") if p.name != "__init__.py"]
    ),
    ids=lambda p: p.name,
)
def test_no_family_string_conditional_in_a_stage_body(path: Path) -> None:
    """Family branching goes through a named capability, not an enum compare.

    `if family is ModelFamily.VIT` says nothing about *why*; `if not
    has_decoder_lane(family)` does, and a second decoder-less family then needs
    no new comparison here at all.
    """

    pattern = re.compile(
        r'family(?:\.value)?\s*(?:is|==)\s*(?:ModelFamily\.|FamilyId\.|"qwen)'
    )
    offenders = [
        f"{path.name}:{index}"
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, (
        "family-string conditionals belong behind a named capability in "
        f"family_registry: {offenders}"
    )


def test_stage_modules_never_import_the_facade() -> None:
    """The facade imports the stages; the reverse would be a cycle."""

    for path in STAGES.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from qairt_agent.pipeline import" not in text, path.name
        assert "import qairt_agent.pipeline\n" not in text, path.name


def test_the_public_facade_surface_is_unchanged() -> None:
    """The CLI, job worker and MCP layers must not notice the split."""

    from qairt_agent.pipeline import QairtAgent

    for name in (
        "plan",
        "generate_config",
        "build",
        "build_genai_container",
        "validate",
        "benchmark",
        "diagnose_quality",
        "diagnose_latency",
        "prepare_vectors",
        "ar_convert",
        "split",
        "mha2sha",
        "convert",
        "quantize",
        "compile_context",
        "run_graph",
        "run_chain",
        "profile",
    ):
        assert callable(getattr(QairtAgent, name, None)), name


def test_the_benchmark_inner_loop_is_a_named_unit() -> None:
    # It was an 870-line closure inside `benchmark`; nothing could call it and
    # nothing could read it in isolation.
    from qairt_agent.pipeline import QairtAgent

    assert callable(QairtAgent._benchmark_one)
    source = (STAGES / "benchmark.py").read_text(encoding="utf-8")
    assert "def run_one(" not in source
