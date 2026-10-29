"""Per-worker benchmark evidence. No optional vendor Python packages are required.

JSONL streams use SI seconds/watts/joules/bytes and explicit measurement scopes.
Device-wide telemetry is not process attribution or achieved FLOP efficiency.
"""

import csv
import functools
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import psutil
import torch

_active = ContextVar("mlpf_benchmark", default=None)


def active_benchmark():
    return _active.get()


def number(value):
    """Vendor N/A/unsupported readings are missing, never zero."""
    try:
        result = float(str(value).strip().split()[0])
        return result if math.isfinite(result) else None
    except (ValueError, IndexError, TypeError):
        return None


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    return value


def command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=3).stdout


def device_selector(rank, vendor, environ=None):
    """Map a process-local CUDA/HIP ordinal through scheduler visibility masks."""
    environ = os.environ if environ is None else environ
    keys = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if vendor == "amd" else ("CUDA_VISIBLE_DEVICES",)
    for key in keys:
        if key in environ:
            devices = [item.strip() for item in environ[key].split(",") if item.strip()]
            if int(rank) >= len(devices):
                raise ValueError(f"Rank device {rank} is outside {key}={environ[key]!r}")
            return devices[int(rank)]
    return str(rank)


class NvidiaTelemetry:
    vendor = "nvidia"
    fields = (
        "index",
        "uuid",
        "pci.bus_id",
        "name",
        "driver_version",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "clocks.sm",
        "clocks.mem",
        "pstate",
        "clocks_throttle_reasons.active",
    )

    def __init__(self, selector, executable="nvidia-smi"):
        self.selector, self.executable = selector, executable

    @classmethod
    def parse(cls, output):
        rows = list(csv.reader(io.StringIO(output)))
        if len(rows) != 1 or len(rows[0]) != len(cls.fields):
            raise ValueError("Expected exactly one NVIDIA device with the requested fields")
        row = dict(zip(cls.fields, (value.strip() for value in rows[0])))
        metrics = {
            "gpu_busy_percent": number(row["utilization.gpu"]),
            "memory_busy_percent": number(row["utilization.memory"]),
            "power_watts": number(row["power.draw"]),
            "power_limit_watts": number(row["power.limit"]),
            "temperature_celsius": number(row["temperature.gpu"]),
            "sm_clock_mhz": number(row["clocks.sm"]),
            "memory_clock_mhz": number(row["clocks.mem"]),
        }
        for source, dest in (("memory.used", "memory_used_bytes"), ("memory.total", "memory_total_bytes")):
            value = number(row[source])
            metrics[dest] = value * 1024**2 if value is not None else None
        return {
            "vendor": cls.vendor,
            "scope": "device",
            "device_id": row["uuid"],
            "pci_bus_id": row["pci.bus_id"],
            "name": row["name"],
            "driver_version": row["driver_version"],
            "metrics": metrics,
            "performance_state": row["pstate"],
            "clock_throttle_reason_mask": row["clocks_throttle_reasons.active"],
        }

    def sample(self):
        fields = getattr(self, "available_fields", self.fields)
        try:
            output = command([self.executable, f"--id={self.selector}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
        except subprocess.CalledProcessError:
            # Older drivers lack throttle-reason fields. Keep basic telemetry.
            fields = self.fields[:-2]
            output = command([self.executable, f"--id={self.selector}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
        self.available_fields = fields
        if len(fields) != len(self.fields):
            output = output.strip() + ",N/A,N/A\n"
        return self.parse(output)


class RocmTelemetry:
    vendor = "amd"

    def __init__(self, selector, executable="rocm-smi"):
        self.selector, self.executable = selector, executable

    @staticmethod
    def parse(output, selector):
        decoded = json.loads(output)
        if not isinstance(decoded, dict):
            raise TypeError("Expected a ROCm device dictionary")
        cards = {key: value for key, value in decoded.items() if key.startswith("card") and isinstance(value, dict)}
        if str(selector).isdigit():
            cards = {key: value for key, value in cards.items() if key == f"card{selector}"}
        else:

            def normalized_id(value):
                return str(value).lower().removeprefix("gpu-").removeprefix("0x").lstrip("0")

            cards = {key: value for key, value in cards.items() if normalized_id(value.get("Unique ID", "")) == normalized_id(selector)}
        if len(cards) != 1:
            raise ValueError("Cannot uniquely match ROCm telemetry to the worker's visible GPU")
        card, values = next(iter(cards.items()))

        def read(*keys):
            return next((number(values[key]) for key in keys if key in values), None)

        return {
            "vendor": "amd",
            "scope": "device",
            "device_id": values.get("Unique ID", card),
            "card": card,
            "metrics": {
                "gpu_busy_percent": read("GPU use (%)"),
                "memory_busy_percent": read("GPU memory use (%)"),
                "memory_used_bytes": read("VRAM Total Used Memory (B)"),
                "memory_total_bytes": read("VRAM Total Memory (B)"),
                "power_watts": read("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)"),
                "temperature_celsius": read("Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)"),
                "sm_clock_mhz": read("sclk clock speed:", "sclk clock speed (Mhz)"),
                "memory_clock_mhz": read("mclk clock speed:", "mclk clock speed (Mhz)"),
                "power_limit_watts": None,
            },
        }

    def sample(self):
        args = [
            self.executable,
            "--showuse",
            "--showmemuse",
            "--showmeminfo",
            "vram",
            "--showpower",
            "--showtemp",
            "--showuniqueid",
            "--showclocks",
            "--json",
        ]
        if self.selector.isdigit():
            args += ["--device", self.selector]
        return self.parse(command(args), self.selector)


def telemetry_backend(rank):
    if rank == "cpu" or not torch.cuda.is_available():
        return None
    vendor = "amd" if torch.version.hip else "nvidia"
    executable = shutil.which("rocm-smi" if vendor == "amd" else "nvidia-smi")
    if executable is None:
        return None
    selector = device_selector(rank, vendor)
    if vendor == "nvidia":
        # CUDA and nvidia-smi ordinal ordering can differ. The caller binds the
        # device first; the CUDA UUID avoids selecting another rank's GPU.
        device_uuid = getattr(torch.cuda.get_device_properties(int(rank)), "uuid", None)
        selector = str(device_uuid) if device_uuid is not None else selector
        if device_uuid is not None and not selector.startswith(("GPU-", "MIG-")):
            selector = "GPU-" + selector
        return NvidiaTelemetry(selector, executable)
    return RocmTelemetry(selector, executable)


def host_metrics():
    """Process tree counters, not node-wide RAM or filesystem device counters."""
    process = psutil.Process()
    result = {
        "scope": "process_tree",
        "rss_bytes": 0,
        "pss_bytes": 0,
        "cpu_seconds": 0.0,
        "read_bytes": 0,
        "write_bytes": 0,
        "read_chars": 0,
        "write_chars": 0,
        "processes": 0,
    }
    pss_available = True
    result["cpu_seconds_by_pid"] = {}
    try:
        result["affinity_cpu_count"] = len(process.cpu_affinity())
    except (AttributeError, psutil.Error):
        result["affinity_cpu_count"] = None
    for child in [process, *process.children(recursive=True)]:
        try:
            result["rss_bytes"] += child.memory_info().rss
            cpu = child.cpu_times()
            result["cpu_seconds"] += cpu.user + cpu.system
            result["cpu_seconds_by_pid"][str(child.pid)] = cpu.user + cpu.system
            result["processes"] += 1
            try:
                result["pss_bytes"] += child.memory_full_info().pss
            except (AttributeError, psutil.Error):
                pss_available = False
            try:
                counters = child.io_counters()
                for key in ("read_bytes", "write_bytes", "read_chars", "write_chars"):
                    if result[key] is not None:
                        result[key] += getattr(counters, key, 0)
            except (AttributeError, psutil.Error):
                for key in ("read_bytes", "write_bytes", "read_chars", "write_chars"):
                    result[key] = None
        except psutil.Error:
            continue
    if not pss_available:
        result["pss_bytes"] = None
    return result


def cpu_interval(current, previous, duration):
    """Estimate average occupied cores; flag exited workers with unobserved tails."""
    if duration <= 0 or "cpu_seconds_by_pid" not in current or "cpu_seconds_by_pid" not in previous:
        return {}
    new, old = current["cpu_seconds_by_pid"], previous["cpu_seconds_by_pid"]
    cores = sum(max(0, value - old.get(pid, 0)) for pid, value in new.items()) / duration
    capacity = current.get("affinity_cpu_count")
    return {
        "cpu_core_equivalents": cores,
        "cpu_percent_of_affinity_capacity": cores / capacity * 100 if capacity else None,
        "cpu_interval_seconds": duration,
        "unobserved_exited_processes": len(old.keys() - new.keys()),
    }


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(q):
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (pos - lower)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def provenance(config, rank, world_size, scope):
    root = Path(__file__).resolve().parents[2]
    git = {}
    try:
        git["revision"] = command(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
        git["status"] = command(["git", "-C", str(root), "status", "--porcelain"])
        git["tracked_diff_sha256"] = hashlib.sha256(command(["git", "-C", str(root), "diff", "HEAD"]).encode()).hexdigest()
        untracked = command(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"]).split("\0")
        git["untracked_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in untracked
            if name and (root / name).is_file() and (root / name).stat().st_size < 2_000_000
        }
    except (OSError, subprocess.SubprocessError):
        git["unavailable"] = True
    packages = {}
    for name in ("numpy", "tensorflow", "tensorflow-datasets", "array-record", "triton", "flash-attn", "pydantic"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    device = None
    if rank != "cpu":
        props = torch.cuda.get_device_properties(int(rank))
        device = {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "uuid": str(getattr(props, "uuid", "unavailable")),
        }
    return {
        "schema_version": 1,
        "scope": scope,
        "rank": rank,
        "world_size": world_size,
        "world_rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "architecture": platform.machine(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": torch.version.hip,
        "packages": packages,
        "device": device,
        "cudnn": torch.backends.cudnn.version(),
        "config": config.model_dump(mode="json"),
        "git": git,
        "environment": {
            key: os.environ[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "PF_SITE",
                "IMG",
                "LUMI_VENV",
                "LUMI_PROJECT_ROOT",
                "ROCM_PATH",
                "CUDA_DEVICE_ORDER",
                "RANK",
                "LOCAL_RANK",
            )
            if key in os.environ
        },
        "advanced_counters": {
            "sm_activity": None,
            "tensor_activity": None,
            "memory_bandwidth_bytes_per_second": None,
            "coherent_memory_traffic": None,
            "interconnect_traffic": None,
            "reason": "requires separate DCGM/Nsight/rocprof profiling; not inferred from GPU busy time",
        },
    }


class BenchmarkRecorder:
    def __init__(self, outdir, config, rank=0, world_size=1, scope="worker", backend=None):
        self.config, self.rank = config.benchmark, rank
        self.start_time = time.perf_counter()
        self.start_unix_time = time.time()
        self.session_id = uuid.uuid4().hex[:12]
        self.directory = Path(outdir) / "benchmark" / f"{socket.gethostname()}_rank{rank}_{os.getpid()}_{self.session_id}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.lock = threading.RLock()
        self.files = {name: (self.directory / f"{name}.jsonl").open("w", buffering=1) for name in ("steps", "phases", "telemetry")}
        self.phase_name, self.step = "startup", None
        self.phase_totals = defaultdict(float)
        self.phase_exclusive_totals = defaultdict(float)
        self.phase_stack = []
        self.steps, self.steady_steps = [], []
        self.events = self.hits = self.steady_events = self.steady_hits = 0
        self.window = []
        self.gpu_integrals = defaultdict(float)
        self.gpu_phase_counts = defaultdict(int)
        self.gpu_peaks = {}
        self.phase_epoch = 0
        self.previous_sample = None
        self.previous_host_sample = None
        self.gpu_sample_count = self.telemetry_errors = 0
        self.pending_cuda = []
        self.stop_event = threading.Event()
        self.backend = backend
        if backend is None:
            try:
                self.backend = telemetry_backend(rank)
            except (ValueError, RuntimeError) as error:
                self.backend_error = str(error)
        metadata = provenance(config, rank, world_size, scope)
        metadata.update(
            session_id=self.session_id,
            telemetry_backend=getattr(self.backend, "vendor", None),
            telemetry_unavailable_reason=getattr(self, "backend_error", "no supported device telemetry backend") if self.backend is None else None,
        )
        self.write_json("metadata.json", metadata)
        self.thread = threading.Thread(target=self._monitor, name="mlpf-benchmark", daemon=True)
        self.thread.start()

    def write_json(self, name, data):
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(clean_json(data), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def emit(self, stream, data):
        with self.lock:
            self.files[stream].write(
                json.dumps(
                    clean_json(
                        {
                            "schema_version": 1,
                            "unix_time": time.time(),
                            "elapsed_seconds": time.perf_counter() - self.start_time,
                            "rank": self.rank,
                            **data,
                        }
                    ),
                    allow_nan=False,
                )
                + "\n"
            )

    @contextmanager
    def phase(self, name, **details):
        started = time.perf_counter()
        with self.lock:
            previous = self.phase_name
            self.phase_name = name
            self.phase_epoch += 1
            frame = {"children": 0.0}
            self.phase_stack.append(frame)
        self.emit("phases", {"event": "begin", "phase": name, "step": self.step, **details})
        try:
            yield
        finally:
            duration = time.perf_counter() - started
            with self.lock:
                self.phase_name = previous
                self.phase_epoch += 1
                self.phase_totals[name] += duration
                self.phase_exclusive_totals[name] += max(0.0, duration - frame["children"])
                self.phase_stack.pop()
                if self.phase_stack:
                    self.phase_stack[-1]["children"] += duration
            self.emit("phases", {"event": "end", "phase": name, "step": self.step, "duration_seconds": duration, **details})

    def ingest_gpu(self, record, timestamp, phase):
        """Trapezoidal estimates only across consecutive valid, same-phase readings."""
        with self.lock:
            self.gpu_sample_count += 1
            self.gpu_phase_counts[phase] += 1
            metrics = record["metrics"]
            for key, value in metrics.items():
                if value is not None:
                    self.gpu_peaks[key] = max(self.gpu_peaks.get(key, value), value)
            busy = metrics.get("gpu_busy_percent")
            if busy is not None:
                self.window.append(busy)
            if self.previous_sample is not None:
                old_time, old_phase, old_epoch, old_record = self.previous_sample
                duration = timestamp - old_time
                # Do not bridge failed samples, long polling gaps, or phase changes.
                if (
                    0 < duration <= self.config.sample_interval_seconds * 3
                    and phase == old_phase
                    and old_epoch == self.phase_epoch
                    and old_record.get("device_id") == record.get("device_id")
                ):
                    for metric, unit in (("gpu_busy_percent", "percent_seconds"), ("power_watts", "joules")):
                        old, new = old_record["metrics"].get(metric), metrics.get(metric)
                        if old is not None and new is not None:
                            self.gpu_integrals[f"{phase}/{metric}/{unit}"] += (old + new) * 0.5 * duration
                            self.gpu_integrals[f"{phase}/{metric}/covered_seconds"] += duration
            self.previous_sample = (timestamp, phase, self.phase_epoch, record)

    def sample(self):
        collection_started = time.perf_counter()
        with self.lock:
            phase, epoch, step = self.phase_name, self.phase_epoch, self.step
        data = {"phase": phase, "step": step, "gpu": None, "host": None, "errors": []}
        try:
            data["host"] = host_metrics()
            host_timestamp = time.perf_counter()
            if self.previous_host_sample is not None:
                old_time, old_host = self.previous_host_sample
                data["host"].update(cpu_interval(data["host"], old_host, host_timestamp - old_time))
            self.previous_host_sample = (host_timestamp, data["host"])
        except (OSError, psutil.Error) as error:
            data["errors"].append(f"host: {error}")
        if self.backend is not None:
            try:
                data["gpu"] = self.backend.sample()
                with self.lock:
                    same_phase = self.phase_epoch == epoch
                if same_phase:
                    self.ingest_gpu(data["gpu"], time.perf_counter(), phase)
                else:
                    data["phase"] = "transition"
                    with self.lock:
                        self.previous_sample = None
            except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
                data["errors"].append(f"gpu: {error}")
                with self.lock:
                    self.previous_sample = None
        if data["errors"]:
            self.telemetry_errors += 1
        data["collection_seconds"] = time.perf_counter() - collection_started
        self.emit("telemetry", data)

    def _monitor(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(self.config.sample_interval_seconds)

    def gpu_window(self):
        with self.lock:
            values, self.window = self.window, []
        return {
            "sample_count": len(values),
            "sum_percent": sum(values),
            "max_percent": max(values) if values else None,
            "mean_percent": statistics.mean(values) if values else None,
        }

    def begin_step(self, step, batch):
        started = time.perf_counter()
        self.step = step
        lengths = batch.mask.sum(dim=-1).tolist()
        pending = None
        if self.rank != "cpu" and self.config.gpu_timing_interval and step % self.config.gpu_timing_interval == 0:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            pending = (step, start, end)
        return {
            "step": step,
            "events": len(lengths),
            "hits": sum(lengths),
            "sequence_lengths": lengths,
            "padded_hits": batch.mask.numel(),
            "input_tensor_bytes": sum(
                value.numel() * value.element_size() for key in batch.attrs if isinstance(value := getattr(batch, key), torch.Tensor)
            ),
            "started": started,
            "cuda": pending,
        }

    def end_step(self, state, data_wait_seconds, diagnostics, losses):
        state = dict(state)
        cuda = state.pop("cuda")
        duration = time.perf_counter() - state.pop("started") + data_wait_seconds
        if cuda is not None:
            cuda[2].record()
            self.pending_cuda.append(cuda)
        self.drain_cuda()
        self.steps.append(duration)
        self.events += state["events"]
        self.hits += state["hits"]
        if len(self.steps) > self.config.warmup_steps:
            self.steady_steps.append(duration)
            self.steady_events += state["events"]
            self.steady_hits += state["hits"]
        self.emit(
            "steps",
            {
                **state,
                "duration_seconds": duration,
                "data_wait_seconds": data_wait_seconds,
                "sampled_diagnostics": diagnostics,
                "losses": losses,
                "timing_scope": "host wall including existing training synchronizations; excludes validation/checkpoints",
            },
        )

    def drain_cuda(self, final=False):
        remaining = []
        for step, start, end in self.pending_cuda:
            if final:
                end.synchronize()
            if end.query():
                self.emit(
                    "steps",
                    {
                        "event": "cuda_timing",
                        "step": step,
                        "cuda_stream_elapsed_seconds": start.elapsed_time(end) / 1000,
                        "timing_scope": "stream interval includes CPU submission gaps, not sum of kernel execution times",
                    },
                )
            else:
                remaining.append((step, start, end))
        self.pending_cuda = remaining

    def close(self, status="completed", error=None):
        self.stop_event.set()
        # Each device command has a three-second timeout. Let the sampler finish
        # before closing its JSONL stream; never leave a thread writing closed files.
        self.thread.join()
        try:
            self.drain_cuda(final=status == "completed")
        except RuntimeError:
            pass  # A failed CUDA context must not mask the original training error.
        elapsed = time.perf_counter() - self.start_time
        framework_memory = None
        if self.rank != "cpu":
            try:
                framework_memory = {
                    "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.rank),
                    "process_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.rank),
                }
            except RuntimeError:
                pass
        device_phases = {}
        for phase, samples in self.gpu_phase_counts.items():
            busy_coverage = self.gpu_integrals.get(f"{phase}/gpu_busy_percent/covered_seconds", 0)
            power_coverage = self.gpu_integrals.get(f"{phase}/power_watts/covered_seconds", 0)
            device_phases[phase] = {
                "sample_count": samples,
                "time_weighted_gpu_busy_percent": self.gpu_integrals[f"{phase}/gpu_busy_percent/percent_seconds"] / busy_coverage
                if busy_coverage
                else None,
                "gpu_busy_covered_seconds": busy_coverage,
                "estimated_energy_joules": self.gpu_integrals[f"{phase}/power_watts/joules"] if power_coverage else None,
                "power_covered_seconds": power_coverage,
            }
        self.write_json(
            "summary.json",
            {
                "schema_version": 1,
                "session_id": self.session_id,
                "rank": self.rank,
                "status": status,
                "error": error,
                "worker_elapsed_seconds": elapsed,
                "steps": len(self.steps),
                "events": self.events,
                "hits": self.hits,
                "started_at_unix_time": self.start_unix_time,
                "finished_at_unix_time": time.time(),
                "events_per_worker_second": self.events / elapsed if elapsed else None,
                "training_step_seconds": distribution(self.steps),
                "steady_training_step_seconds": distribution(self.steady_steps),
                "steady_events_per_training_second": self.steady_events / sum(self.steady_steps) if self.steady_steps else None,
                "steady_hits_per_training_second": self.steady_hits / sum(self.steady_steps) if self.steady_steps else None,
                "warmup_steps_excluded": min(self.config.warmup_steps, len(self.steps)),
                "phase_inclusive_seconds": dict(self.phase_totals),
                "phase_exclusive_seconds": dict(self.phase_exclusive_totals),
                "gpu_sample_count": self.gpu_sample_count,
                "telemetry_error_count": self.telemetry_errors,
                "gpu_trapezoidal_integrals": dict(self.gpu_integrals),
                "device_phase_metrics": device_phases,
                "device_metric_peaks": self.gpu_peaks,
                "framework_memory_bytes": framework_memory,
                "scope_notes": [
                    "per-rank work; sum actual events across ranks, use max elapsed, not summed times",
                    "nested phase times overlap; do not sum inclusive durations",
                    "device power scope depends on hardware; energy is a sampled estimate with explicit coverage",
                    "worker elapsed excludes launcher/import time before recorder creation",
                ],
            },
        )
        for handle in self.files.values():
            handle.close()


@contextmanager
def benchmark_phase(name, **details):
    recorder = active_benchmark()
    if recorder is None:
        yield
    else:
        with recorder.phase(name, **details):
            yield


def record_phase(name, artifact=False):
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with benchmark_phase(name):
                result = function(*args, **kwargs)
            recorder = active_benchmark()
            if recorder is not None and artifact:
                path = Path(args[0])
                recorder.emit("phases", {"event": "artifact", "phase": name, "path": str(path), "bytes": path.stat().st_size})
            return result

        return wrapper

    return decorate


def record_benchmark(scope):
    """Run workers own a recorder; direct Ray training loops get their own too."""

    def decorate(function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            config = bound["config"]
            if active_benchmark() is not None or not config.benchmark.enabled:
                with benchmark_phase(scope):
                    return function(*args, **kwargs)
            rank = bound["rank"]
            if rank != "cpu":
                torch.cuda.set_device(rank)
            recorder = BenchmarkRecorder(bound["outdir"], config, rank, bound["world_size"], scope)
            token = _active.set(recorder)
            status, error = "completed", None
            try:
                with recorder.phase(scope):
                    return function(*args, **kwargs)
            except BaseException as exception:
                status, error = "failed", f"{type(exception).__name__}: {exception}"
                raise
            finally:
                _active.reset(token)
                recorder.close(status, error)

        return wrapper

    return decorate
