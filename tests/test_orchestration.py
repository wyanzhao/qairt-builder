from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from qairt_agent.artifacts import ManifestStore, canonical_json_bytes
from qairt_agent.contracts import (
    ArtifactKind,
    ArtifactRef,
    BuildSpec,
    StageRecord,
    StageStatus,
    utc_now,
)
from qairt_agent.contracts import (
    JobState,
    StageProvenance,
    StageReceipt,
    WorkflowSpec,
    to_workflow_spec,
)
from qairt_agent.agent import QairtAgentClient, spec_sha256
from qairt_agent.errors import JobConflictError, UnsupportedSdkCapabilityError
from qairt_agent.jobs.engine_runner import EngineStageRunner
from qairt_agent.jobs.journal import JobJournal
from qairt_agent.jobs.worker import StageOutcome, WorkflowWorker
from qairt_agent.contracts import ToolResult
from qairt_agent.families.presets import resolve_workflow

PROV = StageProvenance(
    sdk_build="260730134355",
    adapter_capability="explicit_factory",
    platform_abi="ubuntu22.04-x86_64",
    resolved_preset_sha256="p" * 64,
)


def spec_dict(**overrides) -> dict:
    spec = {
        "family": "qwen3",
        "sources": {"text": {"onnx_path": "/m/model.onnx", "encodings_path": "/m/model.encodings"}},
        "output_root": "/artifacts/out",
        "vectors": {"mode": "provided", "validation_manifest": "/v/golden.json"},
    }
    spec.update(overrides)
    return spec


class FakeEngine:
    """Mimics the synchronous engine, publishing a real manifest per stage."""

    def __init__(self, workdir: Path) -> None:
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.calls: list[str] = []
        self._n = 0

    def _result(self, stage: str) -> ToolResult:
        self.calls.append(stage)
        self._n += 1
        path = self.workdir / f"{stage}-{self._n}.manifest.json"
        path.write_text(json.dumps({"stage": stage, "n": self._n}), encoding="utf-8")
        ref = ArtifactRef.from_path(path, kind=ArtifactKind.MANIFEST)
        return ToolResult.success({"stage": stage}, manifest=ref)

    def build(self, spec):
        return self._result("build")

    def build_genai_container(self, spec, config=None):
        return self._result("build")

    def validate(self, uri, sha, vector_manifest=None, config=None):
        return self._result("validate")

    def benchmark(self, uri, sha, config=None):
        return self._result("benchmark")

    def diagnose_quality(self, uri, sha, config=None):
        return self._result("diagnose")


class RealManifestEngine:
    """Small real ManifestStore engine that rejects same-run branching."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def build(self, spec):
        self.calls.append("build")
        build_spec = BuildSpec.model_validate(spec)
        store = ManifestStore(build_spec.output_root / "manifests")
        _, initial_ref = store.create(build_spec)
        _, build_ref = store.revise(
            initial_ref,
            stage=StageRecord(
                name="build",
                status=StageStatus.SUCCEEDED,
                completed_at=utc_now(),
            ),
        )
        return ToolResult.success({"stage": "build"}, manifest=build_ref)

    @staticmethod
    def _continue(stage, uri, sha, config, execution_context=None):
        store = ManifestStore(Path(uri).resolve().parent.parent)
        _, ref = store.revise(
            uri,
            expected_sha256=sha,
            stage=StageRecord(
                name=stage,
                attempt=(
                    execution_context.attempt
                    if execution_context is not None
                    else 1
                ),
                status=StageStatus.SUCCEEDED,
                completed_at=utc_now(),
                metrics={"config": dict(config or {})},
            ),
        )
        return ToolResult.success({"stage": stage}, manifest=ref)

    def validate(
        self,
        uri,
        sha,
        vector_manifest=None,
        config=None,
        execution_context=None,
    ):
        self.calls.append("validate")
        return self._continue(
            "validate",
            uri,
            sha,
            config,
            execution_context,
        )

    def benchmark(self, uri, sha, config=None, execution_context=None):
        self.calls.append("benchmark")
        return self._continue(
            "benchmark",
            uri,
            sha,
            config,
            execution_context,
        )

    def diagnose_quality(
        self,
        uri,
        sha,
        config=None,
        execution_context=None,
    ):
        self.calls.append("diagnose")
        return self._continue(
            "diagnose",
            uri,
            sha,
            config,
            execution_context,
        )


class CrashOnceRealManifestEngine(RealManifestEngine):
    def __init__(self) -> None:
        super().__init__()
        self.crash_benchmark_once = True

    def benchmark(self, uri, sha, config=None, execution_context=None):
        self.calls.append("benchmark")
        if self.crash_benchmark_once:
            self.crash_benchmark_once = False
            raise KeyboardInterrupt("simulated hard worker loss after fork")
        return self._continue(
            "benchmark",
            uri,
            sha,
            config,
            execution_context,
        )


class CrashOnceDiagnoseRealManifestEngine(RealManifestEngine):
    def __init__(self) -> None:
        super().__init__()
        self.crash_diagnose_once = True

    def diagnose_quality(
        self,
        uri,
        sha,
        config=None,
        execution_context=None,
    ):
        self.calls.append("diagnose")
        if self.crash_diagnose_once:
            self.crash_diagnose_once = False
            raise KeyboardInterrupt(
                "simulated hard worker loss after initial-manifest fork"
            )
        return self._continue(
            "diagnose",
            uri,
            sha,
            config,
            execution_context,
        )


class OutputRecordingRealManifestEngine(RealManifestEngine):
    """Real manifest engine whose build publishes an output-root artifact."""

    def build(self, spec):
        self.calls.append("build")
        build_spec = BuildSpec.model_validate(spec)
        store = ManifestStore(build_spec.output_root / "manifests")
        initial, initial_ref = store.create(build_spec)
        output = (
            build_spec.output_root
            / "runs"
            / str(initial.run_id)
            / "build"
            / "compiled.bin"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(str(build_spec.output_root).encode("utf-8"))
        output_ref = ArtifactRef.from_path(
            output,
            kind=ArtifactKind.CONTEXT_BINARY,
            logical_name="compiled-context",
        )
        _, build_ref = store.revise(
            initial_ref,
            stage=StageRecord(
                name="build",
                status=StageStatus.SUCCEEDED,
                completed_at=utc_now(),
                outputs=(output_ref,),
            ),
            artifacts=(output_ref,),
        )
        return ToolResult.success(
            {"stage": "build"},
            manifest=build_ref,
        )


class StrictStageConfigEngine(FakeEngine):
    """Reject missing, shared, or incorrectly routed continuation config."""

    def validate(self, uri, sha, vector_manifest=None, config=None):
        assert vector_manifest is None
        assert config == {
            "actual_manifest": "/v/device.json",
            "vector_manifest": "/v/golden.json",
        }
        return self._result("validate")

    def benchmark(self, uri, sha, config=None):
        assert config == {
            "context_path": "/contexts/decoder.bin",
            "graph_name": "decoder_ar1",
            "vector_manifest": "/v/benchmark.json",
            "warmup_runs": 10,
            "measured_runs": 50,
            "optrace": False,
        }
        return self._result("benchmark")

    def diagnose_quality(self, uri, sha, config=None):
        raise AssertionError("latency diagnosis must not call diagnose_quality")

    def diagnose_latency(self, uri, sha, config=None):
        assert config == {
            "baseline_ops": [{"op": "MatMul", "cycles": 10}],
            "candidate_ops": [{"op": "MatMul", "cycles": 12}],
        }
        return self._result("diagnose_latency")


def make_client(tmp_path, fake, *, background=False):
    return QairtAgentClient(
        jobs_root=tmp_path / "jobs",
        engine_factory=lambda: fake,
        background=background,
        provenance=PROV,
    )


def test_submit_build_inline_succeeds(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    handle = client.submit(spec_dict(), stages=("build",))
    status = handle.wait()

    assert status.state == JobState.SUCCEEDED
    assert fake.calls == ["build"]
    assert [r.stage_name for r in handle.journal.receipts()] == ["build"]
    assert status.manifest is not None
    submission = handle.submission()
    assert set(submission) == {"job_id", "state", "status_path"}


def test_workflow_runs_core_stages(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    handle = client.workflow(spec_dict())
    status = handle.wait()

    assert status.state == JobState.SUCCEEDED
    assert fake.calls == ["build", "validate", "benchmark"]
    assert len(handle.journal.receipts()) == 3


def test_runner_receives_quality_modes_and_effective_diagnostic_compile(
    tmp_path,
) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    workflow = client._normalize_spec(  # noqa: SLF001 - contract wiring test
        spec_dict(
            quality={
                "sqnr_modes": ["full_reference", "teacher_forced", "chain"],
                "dump_intermediates_on_failure": True,
            }
        )
    )

    runner = client._make_runner(  # noqa: SLF001 - contract wiring test
        workflow,
        resolve_workflow(workflow),
        fake,
    )

    assert runner.validate_config == {
        "sqnr_modes": ["full_reference", "teacher_forced", "chain"],
        "dump_intermediates_on_failure": True,
    }
    assert runner.build_spec["compile"]["enable_intermediate_outputs"] is True


def test_workflow_passes_distinct_stage_configs_and_routes_latency_diagnosis(
    tmp_path,
) -> None:
    fake = StrictStageConfigEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    spec = spec_dict(
        stage_configs={
            "build": {"build_only": True},
            "validation": {
                "actual_manifest": "/v/device.json",
                "vector_manifest": "/v/golden.json",
            },
            "benchmark": {
                "context_path": "/contexts/decoder.bin",
                "graph_name": "decoder_ar1",
                "vector_manifest": "/v/benchmark.json",
            },
            "diagnose": {
                "kind": "latency",
                "config": {
                    "baseline_ops": [{"op": "MatMul", "cycles": 10}],
                    "candidate_ops": [{"op": "MatMul", "cycles": 12}],
                },
            },
        }
    )

    status = client.submit(
        spec,
        stages=("build", "validate", "benchmark", "diagnose"),
    ).wait()

    assert status.state == JobState.SUCCEEDED
    assert fake.calls == ["build", "validate", "benchmark", "diagnose_latency"]


def test_background_submit_returns_handle_then_completes(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake, background=True)
    handle = client.submit(spec_dict(), stages=("build",))
    # The handle is usable immediately.
    assert handle.job_id
    assert handle.status_path.endswith("state.json")
    status = handle.wait(timeout=10)
    assert status.state == JobState.SUCCEEDED


def test_rerun_with_identical_spec_reuses_every_stage(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    first = client.workflow(spec_dict())
    first.wait()
    assert fake.calls == ["build", "validate", "benchmark"]

    fake.calls.clear()
    second = client.rerun(first.job_id)  # same spec (read from parent journal)
    status = second.wait()

    assert status.state == JobState.SUCCEEDED
    assert status.parent_job_id == first.job_id
    assert fake.calls == []  # everything reused from the parent
    reused = [r for r in second.journal.receipts() if r.metrics.get("reused_from_parent")]
    assert len(reused) == 3


def test_rerun_with_benchmark_only_change_partially_reuses(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    first = client.workflow(spec_dict())
    first.wait()

    fake.calls.clear()
    adjusted = spec_dict(benchmark={"warmup_runs": 10, "measured_runs": 99, "optrace": False})
    second = client.rerun(first.job_id, adjusted)
    second.wait()

    # Build and validate are unaffected by a benchmark-only change -> reused.
    assert "build" not in fake.calls
    assert "validate" not in fake.calls
    # Benchmark config changed -> rerun (and diagnose depends on its output).
    assert "benchmark" in fake.calls


def test_real_manifest_rerun_forks_after_last_reuse_before_changed_benchmark(
    tmp_path,
) -> None:
    engine = RealManifestEngine()
    client = make_client(tmp_path, engine)
    base = spec_dict(output_root=str(tmp_path / "artifacts"))
    first = client.workflow(base)
    first_status = first.wait()
    assert first_status.state == JobState.SUCCEEDED
    assert first_status.manifest is not None

    engine.calls.clear()
    adjusted = {
        **base,
        "benchmark": {
            "warmup_runs": 10,
            "measured_runs": 99,
            "optrace": False,
        },
        "stage_configs": {
            "benchmark": {"profile_level": "detailed"},
        },
    }
    second = client.rerun(first.job_id, adjusted)
    second_status = second.wait()

    assert second_status.state == JobState.SUCCEEDED
    assert engine.calls == ["benchmark"]
    assert second_status.manifest is not None
    store = ManifestStore(tmp_path / "artifacts" / "manifests")
    first_final = store.load(first_status.manifest)
    second_final = store.load(second_status.manifest)
    expected_build_spec = BuildSpec.model_validate(adjusted)
    assert first_final.build_spec.benchmark.measured_runs == 50
    assert second_final.run_id != first_final.run_id
    assert second_final.revision == 1
    assert second_final.parent_manifest is not None
    assert second_final.build_spec == expected_build_spec

    snapshot = store.load(second_final.parent_manifest)
    assert snapshot.revision == 0
    assert snapshot.parent_manifest is None
    assert [stage.name for stage in snapshot.stages] == ["build", "validate"]
    assert snapshot.metadata["fork_reason"] == "before_stage:benchmark"
    assert snapshot.metadata["forked_from_job_id"] == first.job_id
    assert snapshot.metadata["forked_for_job_id"] == second.job_id
    assert snapshot.metadata["build_spec_rebased"] is True
    assert snapshot.metadata["forked_from_build_spec_sha256"] != (
        snapshot.metadata["effective_build_spec_sha256"]
    )
    assert snapshot.build_spec == expected_build_spec
    assert snapshot.build_spec.benchmark.measured_runs == 99
    assert snapshot.build_spec.stage_configs.benchmark == {
        "profile_level": "detailed"
    }
    assert [stage.name for stage in second_final.stages] == [
        "build",
        "validate",
        "benchmark",
    ]
    benchmark_receipt = next(
        receipt
        for receipt in second.journal.receipts()
        if receipt.stage_name == "benchmark"
    )
    assert benchmark_receipt.inputs
    worker = client._worker_for(second.journal, engine=engine)
    assert benchmark_receipt.stage_key == worker._stage_key(
        "benchmark",
        benchmark_receipt.inputs[0],
    )

    # The old validate revision still has exactly its original benchmark child;
    # the adjusted benchmark lives only under the new run.
    old_run_dir = first_status.manifest.path.parent
    assert len(tuple(old_run_dir.glob("manifest-r000003-*.json"))) == 1


def test_real_manifest_all_reused_rerun_still_mints_new_run_snapshot(
    tmp_path,
) -> None:
    engine = RealManifestEngine()
    client = make_client(tmp_path, engine)
    spec = spec_dict(output_root=str(tmp_path / "artifacts"))
    first = client.workflow(spec)
    first_status = first.wait()
    assert first_status.manifest is not None

    engine.calls.clear()
    second = client.rerun(first.job_id)
    second_status = second.wait()

    assert second_status.state == JobState.SUCCEEDED
    assert second_status.manifest is not None
    assert engine.calls == []
    store = ManifestStore(tmp_path / "artifacts" / "manifests")
    first_final = store.load(first_status.manifest)
    snapshot = store.load(second_status.manifest)
    assert snapshot.run_id != first_final.run_id
    assert snapshot.revision == 0
    assert snapshot.parent_manifest is None
    assert snapshot.stages == first_final.stages
    assert snapshot.artifacts == first_final.artifacts
    assert snapshot.metadata["fork_reason"] == "all_stages_reused"


def test_output_root_change_forces_real_rebuild_into_new_root(tmp_path) -> None:
    engine = OutputRecordingRealManifestEngine()
    client = make_client(tmp_path, engine)
    root_a = tmp_path / "artifacts-a"
    root_b = tmp_path / "artifacts-b"
    first = client.workflow(spec_dict(output_root=str(root_a)))
    first_status = first.wait()
    assert first_status.state == JobState.SUCCEEDED
    assert first_status.manifest is not None

    engine.calls.clear()
    second = client.rerun(
        first.job_id,
        spec_dict(output_root=str(root_b)),
    )
    second_status = second.wait()

    assert second_status.state == JobState.SUCCEEDED
    assert engine.calls == ["build", "validate", "benchmark"]
    assert second_status.manifest is not None
    assert second_status.manifest.path.is_relative_to(root_b)
    final = ManifestStore(root_b / "manifests").load(second_status.manifest)
    assert final.build_spec.output_root == root_b
    compiled = next(
        artifact
        for artifact in final.artifacts
        if artifact.logical_name == "compiled-context"
    )
    assert compiled.path.is_relative_to(root_b)
    assert compiled.path.read_bytes() == str(root_b).encode("utf-8")
    assert not second.journal.receipts()[0].metrics.get(
        "reused_from_parent",
        False,
    )

    # Relocating a build never copies A's BuildSpec or artifact paths into B.
    first_final = ManifestStore(root_a / "manifests").load(
        first_status.manifest
    )
    assert first_final.build_spec.output_root == root_a
    assert all(
        artifact.path.is_relative_to(root_a)
        for artifact in first_final.artifacts
    )


@pytest.mark.parametrize("change", ["family", "source"])
def test_build_relevant_change_rebuilds_without_manifest_rebase(
    tmp_path,
    change: str,
) -> None:
    engine = RealManifestEngine()
    client = make_client(tmp_path, engine)
    root = tmp_path / "artifacts"
    base = spec_dict(output_root=str(root))
    parent = client.workflow(base)
    assert parent.wait().state == JobState.SUCCEEDED

    if change == "family":
        adjusted = {**base, "family": "qwen3_moe"}
    else:
        adjusted = {
            **base,
            "sources": {
                "text": {
                    "onnx_path": "/m/replacement.onnx",
                    "encodings_path": "/m/replacement.encodings",
                }
            },
        }
    engine.calls.clear()
    rerun = client.rerun(parent.job_id, adjusted)
    status = rerun.wait()

    assert status.state == JobState.SUCCEEDED
    assert status.manifest is not None
    assert engine.calls == ["build", "validate", "benchmark"]
    assert not any(
        event.get("type") == "manifest_forked"
        for event in rerun.journal.events()
    )
    final = ManifestStore(root / "manifests").load(status.manifest)
    assert final.build_spec == BuildSpec.model_validate(adjusted)
    assert "forked_from_manifest" not in final.metadata


def test_two_diagnose_from_job_calls_fork_independent_runs(
    tmp_path,
) -> None:
    engine = RealManifestEngine()
    client = make_client(tmp_path, engine)
    spec = spec_dict(output_root=str(tmp_path / "artifacts"))
    parent = client.workflow(spec)
    parent_status = parent.wait()
    assert parent_status.manifest is not None
    parent_ref = parent_status.manifest
    parent_paths_before = {
        path: path.read_bytes()
        for path in parent_ref.path.parent.glob("manifest-*.json")
    }

    diagnoses = []
    for _ in range(2):
        handle = client.prepare(
            spec,
            stages=("diagnose",),
            initial_manifest_job=parent.job_id,
        )
        status = client.execute(handle.job_id)
        assert status.state == JobState.SUCCEEDED
        assert status.manifest is not None
        diagnoses.append((handle, status.manifest))

    store = ManifestStore(tmp_path / "artifacts" / "manifests")
    parent_manifest = store.load(parent_ref)
    first = store.load(diagnoses[0][1])
    second = store.load(diagnoses[1][1])
    assert first.run_id != parent_manifest.run_id
    assert second.run_id != parent_manifest.run_id
    assert first.run_id != second.run_id
    assert first.revision == second.revision == 1
    assert first.parent_manifest is not None
    assert second.parent_manifest is not None
    first_snapshot = store.load(first.parent_manifest)
    second_snapshot = store.load(second.parent_manifest)
    assert first_snapshot.revision == second_snapshot.revision == 0
    assert first_snapshot.metadata["forked_from_manifest"]["sha256"] == parent_ref.sha256
    assert second_snapshot.metadata["forked_from_manifest"]["sha256"] == parent_ref.sha256
    assert first_snapshot.metadata["fork_reason"] == (
        "initial_manifest_before_stage:diagnose"
    )
    assert second_snapshot.metadata["fork_reason"] == (
        "initial_manifest_before_stage:diagnose"
    )
    assert [stage.name for stage in first.stages][-1] == "diagnose"
    assert [stage.name for stage in second.stages][-1] == "diagnose"
    assert engine.calls.count("diagnose") == 2

    # Neither child is a revision of the parent's immutable run.
    assert {
        path: path.read_bytes()
        for path in parent_ref.path.parent.glob("manifest-*.json")
    } == parent_paths_before


def test_adjusted_benchmark_from_job_rebases_effective_build_spec(
    tmp_path,
) -> None:
    engine = RealManifestEngine()
    client = make_client(tmp_path, engine)
    root = tmp_path / "artifacts"
    base = spec_dict(output_root=str(root))
    parent = client.workflow(base)
    assert parent.wait().state == JobState.SUCCEEDED

    adjusted = {
        **base,
        "benchmark": {
            "warmup_runs": 10,
            "measured_runs": 99,
            "optrace": False,
        },
    }
    engine.calls.clear()
    handle = client.prepare(
        adjusted,
        stages=("benchmark",),
        initial_manifest_job=parent.job_id,
    )
    status = client.execute(handle.job_id)

    assert status.state == JobState.SUCCEEDED
    assert status.manifest is not None
    assert engine.calls == ["benchmark"]
    final = ManifestStore(root / "manifests").load(status.manifest)
    assert final.parent_manifest is not None
    snapshot = ManifestStore(root / "manifests").load(
        final.parent_manifest
    )
    expected = BuildSpec.model_validate(adjusted)
    assert snapshot.build_spec == expected
    assert final.build_spec == expected
    assert snapshot.metadata["build_spec_rebased"] is True
    assert snapshot.metadata["fork_reason"] == (
        "initial_manifest_before_stage:benchmark"
    )


def test_rerun_reuses_same_fork_snapshot_after_hard_loss(
    tmp_path,
) -> None:
    parent_engine = RealManifestEngine()
    client = make_client(tmp_path, parent_engine)
    base = spec_dict(output_root=str(tmp_path / "artifacts"))
    parent = client.workflow(base)
    assert parent.wait().state == JobState.SUCCEEDED

    adjusted = {
        **base,
        "benchmark": {
            "warmup_runs": 10,
            "measured_runs": 77,
            "optrace": False,
        },
    }
    handle = client.prepare(
        adjusted,
        stages=("build", "validate", "benchmark"),
        parent_job_id=parent.job_id,
        reuse_from_job=parent.job_id,
    )
    crash_engine = CrashOnceRealManifestEngine()
    first_worker = client._worker_for(handle.journal, engine=crash_engine)
    with pytest.raises(KeyboardInterrupt, match="simulated hard worker loss"):
        first_worker.run()

    fork_events = [
        event
        for event in handle.journal.events()
        if event.get("type") == "manifest_forked"
    ]
    assert len(fork_events) == 1
    fork_payload = fork_events[0]["payload"]

    resumed = client._worker_for(handle.journal, engine=crash_engine)
    resumed.heartbeat_stale_after = 1.0
    resumed.clock = lambda: utc_now() + timedelta(minutes=5)
    status = resumed.run()

    assert status.state == JobState.SUCCEEDED
    assert status.manifest is not None
    assert crash_engine.calls == ["benchmark", "benchmark"]
    assert len(
        [
            event
            for event in handle.journal.events()
            if event.get("type") == "manifest_forked"
        ]
    ) == 1
    final = ManifestStore(tmp_path / "artifacts" / "manifests").load(
        status.manifest
    )
    assert final.parent_manifest is not None
    assert str(final.parent_manifest.path) == fork_payload["fork_manifest"]
    benchmark_receipt = next(
        receipt
        for receipt in handle.journal.receipts()
        if receipt.stage_name == "benchmark"
    )
    assert benchmark_receipt.attempt == 2
    assert benchmark_receipt.inputs[0].path == final.parent_manifest.path
    assert benchmark_receipt.inputs[0].sha256 == final.parent_manifest.sha256
    assert benchmark_receipt.stage_key == resumed._stage_key(
        "benchmark",
        benchmark_receipt.inputs[0],
    )


def test_from_job_reuses_same_initial_fork_after_hard_loss(tmp_path) -> None:
    parent_engine = RealManifestEngine()
    client = make_client(tmp_path, parent_engine)
    base = spec_dict(output_root=str(tmp_path / "artifacts"))
    parent = client.workflow(base)
    assert parent.wait().state == JobState.SUCCEEDED

    handle = client.prepare(
        base,
        stages=("diagnose",),
        initial_manifest_job=parent.job_id,
    )
    crash_engine = CrashOnceDiagnoseRealManifestEngine()
    first_worker = client._worker_for(handle.journal, engine=crash_engine)
    with pytest.raises(
        KeyboardInterrupt,
        match="simulated hard worker loss after initial-manifest fork",
    ):
        first_worker.run()

    fork_events = [
        event
        for event in handle.journal.events()
        if event.get("type") == "manifest_forked"
    ]
    assert len(fork_events) == 1
    fork_payload = fork_events[0]["payload"]
    assert handle.status().manifest is not None
    assert str(handle.status().manifest.path) == fork_payload["fork_manifest"]

    resumed = client._worker_for(handle.journal, engine=crash_engine)
    resumed.heartbeat_stale_after = 1.0
    resumed.clock = lambda: utc_now() + timedelta(minutes=5)
    status = resumed.run()

    assert status.state == JobState.SUCCEEDED
    assert status.manifest is not None
    assert crash_engine.calls == ["diagnose", "diagnose"]
    assert len(
        [
            event
            for event in handle.journal.events()
            if event.get("type") == "manifest_forked"
        ]
    ) == 1
    final = ManifestStore(tmp_path / "artifacts" / "manifests").load(
        status.manifest
    )
    assert final.parent_manifest is not None
    assert str(final.parent_manifest.path) == fork_payload["fork_manifest"]
    diagnose_receipt = next(
        receipt
        for receipt in handle.journal.receipts()
        if receipt.stage_name == "diagnose"
    )
    assert diagnose_receipt.attempt == 2
    assert diagnose_receipt.inputs[0].path == final.parent_manifest.path
    assert diagnose_receipt.inputs[0].sha256 == final.parent_manifest.sha256
    assert diagnose_receipt.stage_key == resumed._stage_key(
        "diagnose",
        diagnose_receipt.inputs[0],
    )


def test_failed_stage_records_failed_receipt_and_fails_job(tmp_path) -> None:
    class FailingEngine(FakeEngine):
        def validate(self, uri, sha, vector_manifest=None, config=None):
            raise RuntimeError("device disconnected")

    fake = FailingEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    handle = client.workflow(spec_dict())
    status = handle.wait()

    assert status.state == JobState.FAILED
    assert status.error is not None
    receipts = {r.stage_name: r for r in handle.journal.receipts()}
    assert receipts["build"].status == StageStatus.SUCCEEDED
    assert receipts["validate"].status == StageStatus.FAILED
    assert receipts["validate"].error is not None


def test_cancel_stops_before_next_stage(tmp_path) -> None:
    journal_ref: dict[str, JobJournal] = {}

    def runner(ctx):
        if ctx.stage_name == "build":
            ctx.journal.request_cancel()
        out = ctx.output_dir
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{ctx.stage_name}.out"
        f.write_bytes(b"x")
        ref = ArtifactRef.from_path(f)
        return StageOutcome(artifacts=(ref,), manifest=ref, metrics={})

    workflow_spec = to_workflow_spec(BuildSpec.model_validate(spec_dict()))
    journal = JobJournal.create(
        tmp_path / "jobs",
        "job-cancel",
        spec_original=json.loads(workflow_spec.model_dump_json()),
        spec_resolved={"preset_id": "qwen3_dense"},
        spec_sha256=spec_sha256(workflow_spec),
    )
    worker = WorkflowWorker(
        journal,
        spec=workflow_spec.model_dump(mode="json"),
        resolved={"preset_id": "qwen3_dense"},
        provenance=PROV,
        stage_runner=runner,
        stages=("build", "validate"),
    )
    status = worker.run()
    assert status.state == JobState.CANCELLED
    # build ran and was receipted; validate was skipped due to cancel.
    assert [r.stage_name for r in journal.receipts()] == ["build"]


def test_resume_continues_from_last_verified_receipt(tmp_path) -> None:
    workflow_spec = to_workflow_spec(BuildSpec.model_validate(spec_dict()))
    spec_payload = workflow_spec.model_dump(mode="json")
    journal = JobJournal.create(
        tmp_path / "jobs",
        "job-resume",
        spec_original=json.loads(workflow_spec.model_dump_json()),
        spec_resolved={"preset_id": "qwen3_dense"},
        spec_sha256=spec_sha256(workflow_spec),
    )

    calls: list[str] = []

    def runner(ctx):
        calls.append(ctx.stage_name)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        f = ctx.output_dir / f"{ctx.stage_name}.out"
        f.write_bytes(b"x")
        ref = ArtifactRef.from_path(f)
        return StageOutcome(artifacts=(ref,), manifest=ref, metrics={})

    # Compute the build key the worker will use, then plant a verified receipt
    # and mark the job orphaned to simulate a crashed worker.
    probe = WorkflowWorker(
        journal, spec=spec_payload, resolved={}, provenance=PROV, stage_runner=runner, stages=("build",)
    )
    build_key = probe._stage_key("build", None)  # noqa: SLF001 - test introspection
    build_out = tmp_path / "build.out"
    build_out.write_bytes(b"build")
    journal.record_receipt(
        StageReceipt(
            stage_key=build_key,
            stage_name="build",
            status=StageStatus.SUCCEEDED,
            started_at=utc_now(),
            completed_at=utc_now(),
            outputs=(ArtifactRef.from_path(build_out),),
            provenance=PROV,
        )
    )
    journal.set_state(JobState.ORPHANED)

    worker = WorkflowWorker(
        journal,
        spec=spec_payload,
        resolved={},
        provenance=PROV,
        stage_runner=runner,
        stages=("build", "validate", "benchmark", "diagnose"),
    )
    status = worker.run()

    assert status.state == JobState.SUCCEEDED
    # build was reused (not re-run); the remaining stages ran.
    assert "build" not in calls
    assert calls == ["validate", "benchmark", "diagnose"]


def test_prepare_resume_rejects_fresh_inflight_job(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = QairtAgentClient(
        jobs_root=tmp_path / "jobs",
        engine_factory=lambda: fake,
        background=False,
        provenance=PROV,
        heartbeat_interval=0.02,
        heartbeat_stale_after=10.0,
    )
    handle = client.prepare(spec_dict(), stages=("build",))
    handle.journal.set_state(JobState.STAGING, current_stage="build")
    handle.journal.set_state(JobState.RUNNING, current_stage="build")
    handle.journal.touch_heartbeat()

    with pytest.raises(JobConflictError, match="fresh worker heartbeat"):
        client.prepare_resume(handle.job_id)

    assert fake.calls == []
    assert handle.status().state == JobState.RUNNING


def test_resume_marks_stale_inflight_job_orphaned_then_completes(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = QairtAgentClient(
        jobs_root=tmp_path / "jobs",
        engine_factory=lambda: fake,
        background=False,
        provenance=PROV,
        heartbeat_interval=0.02,
        heartbeat_stale_after=1.0,
    )
    handle = client.prepare(spec_dict(), stages=("build",))
    handle.journal.set_state(JobState.STAGING, current_stage="build")
    handle.journal.set_state(JobState.RUNNING, current_stage="build")
    handle.journal.touch_heartbeat(now=utc_now() - timedelta(seconds=2))

    resumed = client.resume(handle.job_id)

    assert resumed.status().state == JobState.SUCCEEDED
    assert fake.calls == ["build"]
    orphan_events = [
        event
        for event in handle.events()
        if event["type"] == "state_changed"
        and event["payload"]["state"] == JobState.ORPHANED.value
    ]
    assert len(orphan_events) == 1
    assert orphan_events[0]["payload"]["previous_state"] == JobState.RUNNING.value


def test_resume_rejects_terminal_failed_job(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    journal = JobJournal.create(
        tmp_path / "jobs",
        "job-failed",
        spec_original=spec_dict(),
        spec_resolved={},
        spec_sha256="s" * 64,
    )
    from qairt_agent.errors import ToolErrorData
    from qairt_agent.errors import ErrorCode

    journal.set_state(
        JobState.FAILED,
        error=ToolErrorData(code=ErrorCode.STAGE_FAILED, message="boom"),
    )
    with pytest.raises(JobConflictError, match="terminal"):
        client.resume("job-failed")


def test_submit_omni_routes_component_packaging_but_keeps_runtime_gated(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    client = make_client(tmp_path, fake)
    omni = {
        "preset": "qwen3_5_omni",
        "sources": {
            "text": {
                "onnx_path": "/m/text.onnx",
                "encodings_path": "/m/text.enc",
            },
            "audio": {
                "onnx_path": "/m/audio.onnx",
                "encodings_path": "/m/audio.enc",
            },
        },
        "metadata": {
            "attached_models_by_ar": {
                "1": {
                    "model_path": "/m/text-ar1.onnx",
                    "encodings_path": "/m/text-ar1.enc",
                },
                "128": {
                    "model_path": "/m/text-ar128.onnx",
                    "encodings_path": "/m/text-ar128.enc",
                },
            }
        },
        "output_root": "/artifacts/omni",
    }
    handle = client.submit(omni, stages=("build",))
    assert handle.status().state is JobState.SUCCEEDED
    assert handle.journal.spec_resolved()["runtime_supported"] is False
    assert fake.calls == ["build"]


def test_engine_runner_maps_stages_and_guards(tmp_path) -> None:
    fake = FakeEngine(tmp_path / "engine")
    runner = EngineStageRunner(
        engine=fake,
        build_spec=spec_dict(),
        pipeline="low_level",
    )

    class Ctx:
        def __init__(self, stage_name, current_manifest=None):
            self.stage_name = stage_name
            self.current_manifest = current_manifest
            self.output_dir = tmp_path
            self.attempt = 1

    build_outcome = runner(Ctx("build"))
    assert build_outcome.manifest is not None
    assert fake.calls == ["build"]

    # validate without a prior manifest is guarded.
    from qairt_agent.errors import InvalidSpecError

    with pytest.raises(InvalidSpecError, match="requires a prior build"):
        runner(Ctx("validate"))

    with pytest.raises(InvalidSpecError, match="unknown workflow stage"):
        runner(Ctx("frobnicate"))


def test_engine_runner_surfaces_structured_failure(tmp_path) -> None:
    class FailedEngine(FakeEngine):
        def build(self, spec):
            from qairt_agent.errors import ErrorCode, ToolErrorData

            return ToolResult.failure(
                ToolErrorData(code=ErrorCode.QAIRT_UNAVAILABLE, message="no sdk", stage="build")
            )

    fake = FailedEngine(tmp_path / "engine")
    runner = EngineStageRunner(engine=fake, build_spec=spec_dict(), pipeline="low_level")

    class Ctx:
        stage_name = "build"
        current_manifest = None
        output_dir = tmp_path
        attempt = 1

    from qairt_agent.errors import ToolError

    with pytest.raises(ToolError):
        runner(Ctx())
