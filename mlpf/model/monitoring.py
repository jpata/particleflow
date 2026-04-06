import psutil
import resource
from collections import defaultdict
from mlpf.logger import _logger


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
        children = current_process.children(recursive=True)
        child_files = sum(len(child.open_files()) for child in children)
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
