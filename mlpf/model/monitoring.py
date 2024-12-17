import os
import psutil
import resource
from collections import defaultdict

def monitor_open_files():
    """
    Monitors the number of open files for the current process and its children.
    Returns a dictionary with different file descriptor metrics.
    """
    metrics = defaultdict(int)
    metrics['error'] = False
    
    # Get the current process
    current_process = psutil.Process()
    
    try:
        # Get open files for current process
        open_files = current_process.open_files()
        metrics['process_open_files'] = len(open_files)
        
        # Get open files for child processes
        children = current_process.children(recursive=True)
        child_files = sum(len(child.open_files()) for child in children)
        metrics['child_process_open_files'] = child_files
        
        # Total open files
        metrics['total_open_files'] = metrics['process_open_files'] + metrics['child_process_open_files']
        
        # Get soft and hard limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        metrics['soft_limit'] = soft_limit
        metrics['hard_limit'] = hard_limit
        
        # Calculate percentage of soft limit used
        if soft_limit > 0:
            metrics['soft_limit_usage_percent'] = (metrics['total_open_files'] / soft_limit) * 100
        
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
        print(f"Error monitoring file descriptors: {e}")
        metrics['error'] = True
        
    return metrics

def log_open_files_to_tensorboard(tensorboard_writer, step):
    """
    Logs the open file metrics to tensorboard.
    
    Args:
        tensorboard_writer: SummaryWriter instance
        step: Current training step
    """
    if tensorboard_writer is None:
        return
        
    metrics = monitor_open_files()
    if not metrics['error']:
        # Log all metrics to tensorboard
        tensorboard_writer.add_scalar("system/process_open_files", metrics['process_open_files'], step)
        tensorboard_writer.add_scalar("system/child_process_open_files", metrics['child_process_open_files'], step)
        tensorboard_writer.add_scalar("system/total_open_files", metrics['total_open_files'], step)
        tensorboard_writer.add_scalar("system/soft_limit", metrics['soft_limit'], step)
        tensorboard_writer.add_scalar("system/hard_limit", metrics['hard_limit'], step)
        
        if 'soft_limit_usage_percent' in metrics:
            tensorboard_writer.add_scalar("system/soft_limit_usage_percent", metrics['soft_limit_usage_percent'], step)


def log_step_to_tensorboard(batch, loss_accum, lr_schedule, tensorboard_writer, step):
    #get the number of elements, excluding padded elements
    num_elems = batch.X[batch.mask].shape[0]

    tensorboard_writer.add_scalar("step/loss", loss_accum / num_elems, step)
    tensorboard_writer.add_scalar("step/num_elems", num_elems, step)
    tensorboard_writer.add_scalar("step/num_batch", batch.X.shape[0], step)
    tensorboard_writer.add_scalar("step/learning_rate", lr_schedule.get_last_lr()[0], step)
