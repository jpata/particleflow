import resource
from collections import defaultdict

import psutil

from mlpf.logger import _logger
from mlpf.model.benchmarking import active_benchmark

try:
    import torch
except ImportError:
    torch = None


def log_gpu_utilization_to_tensorboard(tensorboard_writer, step):
    recorder = active_benchmark()
    if recorder is None or tensorboard_writer is None:
        return
    metrics = recorder.gpu_window()
    tensorboard_writer.add_scalar("system/gpu_utilization_sample_count", metrics["sample_count"], step)
    tensorboard_writer.add_scalar("system/gpu_utilization_sum_percent", metrics["sum_percent"], step)
    if metrics["sample_count"]:
        tensorboard_writer.add_scalar("system/gpu_utilization_max", metrics["max_percent"], step)
        tensorboard_writer.add_scalar("system/gpu_utilization_avg", metrics["mean_percent"], step)


def monitor_open_files():
    """
    Monitors the number of open files for the current process and its children.
    Returns a dictionary with different file descriptor metrics.
    """
    metrics = defaultdict(int)
    metrics["error"] = False

    # Get the current process
    current_process = psutil.Process()

    try:
        # Get open files for current process
        open_files = current_process.open_files()
        metrics["process_open_files"] = len(open_files)

        # Get open files for child processes
        try:
            children = current_process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            children = []

        child_files = 0
        for child in children:
            try:
                child_files += len(child.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        metrics["child_process_open_files"] = child_files

        # Total open files
        metrics["total_open_files"] = metrics["process_open_files"] + metrics["child_process_open_files"]

        # Get soft and hard limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        metrics["soft_limit"] = soft_limit
        metrics["hard_limit"] = hard_limit

        # Calculate percentage of soft limit used
        if soft_limit > 0:
            metrics["soft_limit_usage_percent"] = (metrics["total_open_files"] / soft_limit) * 100

    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
        _logger.error(f"Error monitoring file descriptors: {e}")
        metrics["error"] = True

    return metrics


def log_open_files_to_tensorboard(tensorboard_writer, step):
    """
    Logs the open file metrics to tensorboard.

    Args:
        tensorboard_writer: SummaryWriter instance
        step: Current training step
    """
    try:
        metrics = monitor_open_files()
        if not metrics["error"]:
            _logger.debug(
                f"Open files: total={metrics['total_open_files']} "
                f"(process={metrics['process_open_files']}, child={metrics['child_process_open_files']}), "
                f"soft_limit={metrics['soft_limit']}"
            )

            if tensorboard_writer is not None:
                # Log all metrics to tensorboard
                tensorboard_writer.add_scalar("system/process_open_files", metrics["process_open_files"], step)
                tensorboard_writer.add_scalar("system/child_process_open_files", metrics["child_process_open_files"], step)
                tensorboard_writer.add_scalar("system/total_open_files", metrics["total_open_files"], step)
                tensorboard_writer.add_scalar("system/soft_limit", metrics["soft_limit"], step)
                tensorboard_writer.add_scalar("system/hard_limit", metrics["hard_limit"], step)

                if "soft_limit_usage_percent" in metrics:
                    tensorboard_writer.add_scalar("system/soft_limit_usage_percent", metrics["soft_limit_usage_percent"], step)
    except Exception:
        _logger.error("Error monitoring open files, this can happen in singularity on lxplus")


def log_step_to_tensorboard(batch, loss_accum, lr_schedule, tensorboard_writer, step):
    # get the number of elements, excluding padded elements
    num_elems = batch.X[batch.mask].shape[0]

    tensorboard_writer.add_scalar("step/num_elems", num_elems, step)
    tensorboard_writer.add_scalar("step/num_batch", batch.X.shape[0], step)
    tensorboard_writer.add_scalar("step/learning_rate", lr_schedule.get_last_lr()[0], step)


def collect_process_memory_metrics():
    process = psutil.Process()
    mem = process.memory_info()
    metrics = {
        "rss_MB": mem.rss / 1024**2,
        "vms_MB": mem.vms / 1024**2,
        "num_threads": float(process.num_threads()),
    }
    try:
        full_mem = process.memory_full_info()
        if hasattr(full_mem, "uss"):
            metrics["uss_MB"] = full_mem.uss / 1024**2
        if hasattr(full_mem, "pss"):
            metrics["pss_MB"] = full_mem.pss / 1024**2
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass

    child_rss = 0.0
    child_vms = 0.0
    child_count = 0
    for child in process.children(recursive=True):
        try:
            child_mem = child.memory_info()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        child_count += 1
        child_rss += child_mem.rss / 1024**2
        child_vms += child_mem.vms / 1024**2
    metrics["child_count"] = float(child_count)
    metrics["child_rss_MB"] = child_rss
    metrics["child_vms_MB"] = child_vms

    try:
        metrics["num_fds"] = float(process.num_fds())
    except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess):
        pass

    return metrics


def collect_cuda_memory_metrics(device=None):
    if torch is None or not torch.cuda.is_available():
        return {}

    metrics = {
        "allocated_MB": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_MB": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_MB": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved_MB": torch.cuda.max_memory_reserved(device) / 1024**2,
    }
    stats = torch.cuda.memory_stats(device)
    for source_key, target_key in [
        ("active_bytes.all.current", "active_MB"),
        ("inactive_split_bytes.all.current", "inactive_split_MB"),
        ("allocated_bytes.all.current", "allocated_stats_MB"),
        ("reserved_bytes.all.current", "reserved_stats_MB"),
        ("segment.all.current", "segments"),
        ("num_alloc_retries", "num_alloc_retries"),
        ("num_ooms", "num_ooms"),
    ]:
        if source_key not in stats:
            continue
        value = float(stats[source_key])
        if source_key.endswith("_bytes.all.current"):
            value /= 1024**2
        metrics[target_key] = value
    host_stats = getattr(torch.cuda.memory, "host_memory_stats", None)
    if host_stats is not None:
        try:
            for key, value in host_stats().items():
                if key in ("allocated_bytes.current", "allocated_bytes.peak", "active_bytes.current"):
                    metrics[f"pinned_host_{key.replace('.', '_')}_MB"] = value / 1024**2
        except RuntimeError:
            pass
    return metrics


def log_diagnostics_to_tensorboard(tensorboard_writer, diagnostics, step, prefix="diagnostic"):
    if tensorboard_writer is None:
        return
    for group, values in diagnostics.items():
        if isinstance(values, dict):
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    tensorboard_writer.add_scalar(f"{prefix}/{group}/{name}", value, step)
        elif isinstance(values, (int, float)):
            tensorboard_writer.add_scalar(f"{prefix}/{group}", values, step)


def log_gradients_to_tensorboard(model, tensorboard_writer, step):
    """
    Logs the gradient norms to tensorboard.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            tensorboard_writer.add_scalar(f"gradients/{name}", param.grad.norm(), step)


def log_residuals_to_tensorboard(model, tensorboard_writer, step):
    """
    Logs the residual norms of the attention layers to tensorboard.
    """
    for name, module in model.named_modules():
        if hasattr(module, "input_norm") and module.input_norm is not None:
            tensorboard_writer.add_scalar(f"stats/{name}_input_norm", module.input_norm, step)
            tensorboard_writer.add_scalar(f"stats/{name}_seq_len", module.seq_len, step)

        if hasattr(module, "mha_res_norm") and module.mha_res_norm is not None:
            tensorboard_writer.add_scalar(f"residuals/{name}_mha", module.mha_res_norm, step)
            if hasattr(module, "input_norm"):
                tensorboard_writer.add_scalar(f"residuals_ratio/{name}_mha", module.mha_res_norm / module.input_norm, step)

        if hasattr(module, "ffn_res_norm") and module.ffn_res_norm is not None:
            tensorboard_writer.add_scalar(f"residuals/{name}_ffn", module.ffn_res_norm, step)
            if hasattr(module, "input_norm"):
                tensorboard_writer.add_scalar(f"residuals_ratio/{name}_ffn", module.ffn_res_norm / module.input_norm, step)
