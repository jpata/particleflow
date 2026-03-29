import psutil
import resource
import subprocess
import shutil
import threading
import time
from collections import defaultdict
from mlpf.logger import _logger


class GPUMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.max_utilization = 0.0
        self.sum_utilization = 0.0
        self.count_utilization = 0
        self.stop_event = threading.Event()
        self.smi_command = shutil.which("nvidia-smi") or shutil.which("rocm-smi")
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        if self.smi_command:
            self.thread.start()

    def _monitor(self):
        while not self.stop_event.is_set():
            try:
                current_util = None
                if "nvidia-smi" in self.smi_command:
                    res = subprocess.run(
                        [self.smi_command, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    utils = [float(x) for x in res.stdout.strip().split("\n") if x.strip()]
                    if utils:
                        current_util = max(utils)
                elif "rocm-smi" in self.smi_command:
                    res = subprocess.run(
                        [self.smi_command, "--showuse"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    for line in res.stdout.split("\n"):
                        if "GPU use (%)" in line:
                            val = line.split(":")[-1].strip()
                            current_util = float(val)
                            break
                
                if current_util is not None:
                    self.max_utilization = max(self.max_utilization, current_util)
                    self.sum_utilization += current_util
                    self.count_utilization += 1
            except Exception:
                pass
            time.sleep(self.interval)

    def get_metrics_and_reset(self):
        max_val = self.max_utilization
        avg_val = self.sum_utilization / self.count_utilization if self.count_utilization > 0 else 0.0
        
        self.max_utilization = 0.0
        self.sum_utilization = 0.0
        self.count_utilization = 0
        
        return max_val, avg_val


# Initialize a global monitor
_gpu_monitor = None


def log_gpu_utilization_to_tensorboard(tensorboard_writer, step):
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()

    max_util, avg_util = _gpu_monitor.get_metrics_and_reset()
    if tensorboard_writer:
        tensorboard_writer.add_scalar("system/gpu_utilization_max", max_util, step)
        tensorboard_writer.add_scalar("system/gpu_utilization_avg", avg_util, step)


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


def log_dataloader_to_tensorboard(loader_state_dict, tensorboard_writer, step):
    for k in ["cur_index"]:
        tensorboard_writer.add_scalar("step/{}".format(k), loader_state_dict[k], step)
