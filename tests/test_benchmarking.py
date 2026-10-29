import json
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from mlpf.conf import BenchmarkConfig
from mlpf.model import benchmarking as bm
from mlpf.model.PFDataset import PFBatch


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "provenance", lambda *args: {})
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(threading.Thread, "join", lambda self: None)
    result = bm.BenchmarkRecorder(tmp_path, SimpleNamespace(benchmark=BenchmarkConfig(warmup_steps=1)), rank="cpu")
    yield result
    result.close()


def gpu(busy=50.0, power=10.0):
    return {"device_id": "GPU-a", "metrics": {"gpu_busy_percent": busy, "power_watts": power}}


def test_nvidia_values_units_and_unsupported_metrics():
    result = bm.NvidiaTelemetry.parse("0,GPU-a,0000:01:00.0,GH200,595,50,N/A,100,96000,120.5,900,35,1900,3000,P0,0x0\n")
    assert result["device_id"] == "GPU-a"
    assert result["metrics"]["gpu_busy_percent"] == 50
    assert result["metrics"]["memory_busy_percent"] is None
    assert result["metrics"]["memory_used_bytes"] == 100 * 1024**2
    assert result["metrics"]["power_watts"] == 120.5
    with pytest.raises(ValueError, match="exactly one"):
        bm.NvidiaTelemetry.parse("0,GPU-a\n1,GPU-b\n")


def test_old_nvidia_driver_falls_back_once(monkeypatch):
    calls = []

    def query(args):
        calls.append(args)
        if "clocks_throttle_reasons.active" in args[2]:
            raise subprocess.CalledProcessError(1, args)
        return "0,GPU-a,0000:01:00.0,GH200,595,50,N/A,100,96000,120.5,900,35,1900,3000\n"

    monkeypatch.setattr(bm, "command", query)
    reader = bm.NvidiaTelemetry("GPU-a")
    assert reader.sample()["metrics"]["gpu_busy_percent"] == 50
    assert reader.sample()["clock_throttle_reason_mask"] == "N/A"
    assert len(calls) == 3
    assert all("--id=GPU-a" in args for args in calls)


def test_rocm_selects_one_device_and_preserves_missing_values():
    output = json.dumps(
        {
            "card0": {"Unique ID": "a", "GPU use (%)": "99"},
            "card1": {
                "Unique ID": "b",
                "GPU use (%)": "42",
                "VRAM Total Used Memory (B)": "1024",
                "Current Socket Graphics Package Power (W)": "250",
                "Temperature (Sensor junction) (C)": "55",
            },
        }
    )
    for selector in ("1", "b", "GPU-b", "GPU-000b"):
        result = bm.RocmTelemetry.parse(output, selector)
        assert result["device_id"] == "b"
        assert result["metrics"]["gpu_busy_percent"] == 42
        assert result["metrics"]["memory_used_bytes"] == 1024
        assert result["metrics"]["memory_total_bytes"] is None
        assert result["metrics"]["power_watts"] == 250
    with pytest.raises(ValueError, match="uniquely"):
        bm.RocmTelemetry.parse(output, "unknown")


def test_visible_device_mapping():
    assert bm.device_selector(1, "nvidia", {"CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a"}) == "GPU-a"
    assert bm.device_selector(0, "amd", {"ROCR_VISIBLE_DEVICES": "3"}) == "3"
    with pytest.raises(ValueError, match="outside"):
        bm.device_selector(1, "amd", {"ROCR_VISIBLE_DEVICES": "3"})


def test_nvidia_backend_normalizes_cuda_uuid_for_smi(monkeypatch):
    monkeypatch.setattr(bm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bm.torch.cuda, "get_device_properties", lambda rank: SimpleNamespace(uuid="1234-abcd"))
    monkeypatch.setattr(bm.torch.version, "hip", None)
    monkeypatch.setattr(bm.shutil, "which", lambda name: name)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert bm.telemetry_backend(0).selector == "GPU-1234-abcd"


def test_energy_and_utilization_weight_time_not_number_of_samples(recorder):
    recorder.ingest_gpu(gpu(0, 10), 0, "training")
    recorder.ingest_gpu(gpu(100, 30), 2, "training")
    recorder.ingest_gpu(gpu(100, 50), 3, "training")
    assert recorder.gpu_integrals["training/power_watts/joules"] == 80
    assert recorder.gpu_integrals["training/gpu_busy_percent/covered_seconds"] == 3
    assert recorder.gpu_integrals["training/gpu_busy_percent/percent_seconds"] == 200
    window = recorder.gpu_window()
    assert window["sample_count"] == 3
    assert window["sum_percent"] == 200
    assert recorder.gpu_window()["mean_percent"] is None


def test_no_energy_interpolation_across_phases_missing_power_or_long_gaps(recorder):
    recorder.ingest_gpu(gpu(), 0, "training")
    recorder.ingest_gpu(gpu(), 1, "validation")
    recorder.ingest_gpu(gpu(power=None), 2, "validation")
    recorder.ingest_gpu(gpu(), 10, "validation")
    recorder.phase_epoch += 1
    recorder.ingest_gpu(gpu(), 11, "validation")
    assert not any(key.endswith("joules") for key in recorder.gpu_integrals)


def test_telemetry_failure_is_recorded_and_breaks_coverage(recorder, monkeypatch):
    monkeypatch.setattr(bm, "host_metrics", lambda: {"rss_bytes": 42})
    recorder.backend = SimpleNamespace(sample=Mock(side_effect=subprocess.TimeoutExpired("smi", 3)))
    recorder.previous_sample = (0, "startup", 0, gpu())
    recorder.sample()
    assert recorder.previous_sample is None
    assert recorder.telemetry_errors == 1
    record = json.loads((recorder.directory / "telemetry.jsonl").read_text())
    assert record["gpu"] is None
    assert record["host"]["rss_bytes"] == 42
    assert record["errors"]


def test_every_batch_counts_real_work_and_resume_warmup_is_per_session(recorder):
    batch = PFBatch(X=torch.tensor([[[1.0], [1.0], [0.0]], [[1.0], [0.0], [0.0]]]))
    for step in (1000, 1001):
        state = recorder.begin_step(step, batch)
        recorder.end_step(state, 0.001, {}, {"Total": 1.0, "unsupported": float("nan")})
    assert recorder.events == 4 and recorder.hits == 6
    assert len(recorder.steady_steps) == 1
    records = [json.loads(line) for line in (recorder.directory / "steps.jsonl").read_text().splitlines()]
    assert records[0]["sequence_lengths"] == [2, 1]
    assert records[0]["padded_hits"] == 6
    assert records[0]["input_tensor_bytes"] == 24
    assert records[0]["losses"]["unsupported"] is None


def test_checkpoint_artifact_and_phase_boundaries(recorder, tmp_path):
    @bm.record_phase("checkpoint_save", artifact=True)
    def save(path):
        path.write_bytes(b"checkpoint")

    token = bm._active.set(recorder)
    try:
        save(tmp_path / "model.pth")
    finally:
        bm._active.reset(token)
    records = [json.loads(line) for line in (recorder.directory / "phases.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["begin", "end", "artifact"]
    assert records[-1]["bytes"] == 10
    assert recorder.phase_name == "startup"


def test_quantiles_and_nonfinite_vendor_values():
    assert bm.distribution([1, 2, 3, 4])["p50"] == 2.5
    assert bm.distribution([1, 2, 3, 4])["p99"] == pytest.approx(3.97)
    assert bm.distribution([]) is None
    for value in ("N/A", "[Not Supported]", "nan", "inf", None):
        assert bm.number(value) is None


def test_cpu_interval_accounts_for_new_workers_and_flags_exited_workers():
    old = {"cpu_seconds_by_pid": {"1": 5, "2": 100}}
    new = {"cpu_seconds_by_pid": {"1": 6, "3": 0.5}, "affinity_cpu_count": 8}
    result = bm.cpu_interval(new, old, 2)
    assert result["cpu_core_equivalents"] == 0.75
    assert result["cpu_percent_of_affinity_capacity"] == 9.375
    assert result["unobserved_exited_processes"] == 1


def test_live_cpu_collector_records_host_metrics_without_device_telemetry(recorder):
    recorder.sample()
    recorder.sample()
    samples = [json.loads(line) for line in (recorder.directory / "telemetry.jsonl").read_text().splitlines()]
    assert all(row["gpu"] is None and not row["errors"] for row in samples)
    assert samples[-1]["host"]["rss_bytes"] > 0
    assert samples[-1]["host"]["cpu_core_equivalents"] >= 0
    assert samples[-1]["collection_seconds"] > 0


def test_distributed_report_sums_work_and_gpu_hours_but_not_elapsed_time():
    from mlpf.benchmark_report import allocation_totals

    common = {"run_dir": "experiment", "job_id": "42", "array_task": "0", "hostname": "node1", "world_size": 2, "status": "completed"}
    rows = [
        {**common, "rank": 0, "events": 10, "hits": 100, "worker_elapsed_seconds": 2, "started_at_unix_time": 100, "finished_at_unix_time": 102},
        {**common, "rank": 1, "events": 20, "hits": 200, "worker_elapsed_seconds": 3, "started_at_unix_time": 100.1, "finished_at_unix_time": 103.1},
    ]
    report = allocation_totals(rows)[0]
    assert report["events_processed_including_retries"] == 30
    assert report["observed_worker_span_seconds"] == pytest.approx(3.1)
    assert report["recorded_gpu_worker_hours"] == pytest.approx(5 / 3600)
    assert report["observed_workers"] == report["expected_world_size"] == 2


def test_thread_shutdown_and_failure_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "provenance", lambda *args: {})
    sampled = threading.Event()
    monkeypatch.setattr(bm, "host_metrics", lambda: sampled.set() or {"rss_bytes": 1})

    @bm.record_benchmark("worker")
    def fail(rank, world_size, config, outdir):
        assert sampled.wait(2)
        raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed"):
        fail("cpu", 1, SimpleNamespace(benchmark=BenchmarkConfig(sample_interval_seconds=0.1)), tmp_path)
    assert bm.active_benchmark() is None
    summary = json.loads(next(tmp_path.glob("benchmark/*/summary.json")).read_text())
    assert summary["status"] == "failed"
    assert summary["error"] == "ValueError: training failed"
    assert summary["gpu_sample_count"] == 0
    assert not any(thread.name == "mlpf-benchmark" and thread.is_alive() for thread in threading.enumerate())


def test_disabled_recorder_creates_no_files(tmp_path):
    @bm.record_benchmark("worker")
    def run(rank, world_size, config, outdir):
        return 42

    assert run("cpu", 1, SimpleNamespace(benchmark=BenchmarkConfig(enabled=False)), tmp_path) == 42
    assert not (tmp_path / "benchmark").exists()


def test_validation_history_uses_cycle_duration_not_elapsed_training(tmp_path, monkeypatch):
    from mlpf.model import training

    monkeypatch.setattr(training, "evaluate", lambda **kwargs: {"Total": 1.0, "Total_unweighted": 2.0})
    monkeypatch.setattr(training, "get_train_epoch", lambda loader: 0.5)
    for name in ("log_memory", "_log_validation_results_to_tensorboard", "_log_group_losses_to_tensorboard", "_log_validation_probe_results"):
        monkeypatch.setattr(training, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(training.time, "perf_counter", Mock(side_effect=[10, 11, 13, 14, 15, 16, 17, 18]))
    monkeypatch.setattr(training.time, "time", lambda: 1000)
    training._run_validation_cycle(
        rank="cpu",
        world_size=1,
        model=None,
        optimizer=None,
        lr_schedule=None,
        valid_loader=Mock(),
        step=10,
        num_steps=20,
        losses_train={"Total": 1.0, "Total_unweighted": 2.0},
        best_val_loss=10,
        stale_steps=0,
        outdir=str(tmp_path),
        config=SimpleNamespace(enabled_test_datasets=[]),
        device_type="cpu",
        dtype=torch.float32,
        tensorboard_writer_valid=Mock(),
        comet_experiment=None,
        checkpoint_dir=str(tmp_path),
        train_time=0.5,
        t0_initial=100,
        use_ray=False,
        train_loader=None,
        valid_sampler=None,
        train_sampler=None,
        losses_train_group={},
    )
    history = json.loads((tmp_path / "history/step_10.json").read_text())
    assert history["step_valid_time"] == 2
    assert history["step_validation_cycle_time"] == 8
    assert history["step_test_time"] == history["step_plot_time"] == 1
    assert history["elapsed_training_loop_seconds"] == 900
