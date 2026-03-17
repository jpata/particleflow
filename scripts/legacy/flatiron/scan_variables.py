import subprocess
import time


# Functions
def replaceline_and_save(fname, findln, newline, override=False):
    if findln not in newline and not override:
        raise ValueError("Detected inconsistency!")

    with open(fname, "r") as fid:
        lines = fid.readlines()

    found = False
    pos = None
    for ii, line in enumerate(lines):
        if findln in line:
            pos = ii
            found = True
            break

    if not found:
        raise ValueError("Not found!")

    if "\n" in newline:
        lines[pos] = newline
    else:
        lines[pos] = newline + "\n"

    with open(fname, "w") as fid:
        fid.writelines(lines)


def get_CVD_string(devices):
    s = ""
    for i in range(0, devices):
        s += f"{i},"
    return s[:-1]


def replaceline_CUDA_VISIBLE_DEVICES(fname, devices):
    replaceline_and_save(
        fname,
        findln="CUDA_VISIBLE_DEVICES",
        newline="export CUDA_VISIBLE_DEVICES={}".format(get_CVD_string(devices)),
    )


def replaceline_batch_multiplier(fname, bmultiplier):
    replaceline_and_save(
        fname,
        findln="--gpu-batch-multiplier",
        newline="    --gpu-batch-multiplier {} \\".format(bmultiplier),
    )


def replaceline_num_gpus(fname, num_gpus):
    replaceline_and_save(
        fname,
        findln="#SBATCH --gpus-per-node",
        newline="#SBATCH --gpus-per-node={}".format(num_gpus),
    )
    replaceline_and_save(
        fname,
        findln="#SBATCH --gpus-per-task",
        newline="#SBATCH --gpus-per-task={}".format(num_gpus),
    )


def replaceline_num_cpus_per_task(fname, num_cpus):
    replaceline_and_save(
        fname,
        findln="#SBATCH --cpus-per-task",
        newline="#SBATCH --cpus-per-task={}".format(num_cpus),
    )


def replaceline_ntrain(fname, ntrain):
    replaceline_and_save(
        fname,
        findln="--ntrain",
        newline="    --ntrain {} \\".format(ntrain),
    )


def replaceline_lr(fname, lr):
    replaceline_and_save(
        fname,
        findln="--lr",
        newline="    --lr {}".format(lr),
    )


def replaceline_epochs(fname, epochs):
    replaceline_and_save(
        fname,
        findln="--num-epochs",
        newline="    --num-epochs {} \\".format(epochs),
    )


# Configuration parameters
# batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_a100.slurm"
# batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_h100.slurm"
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_train_singularity.slurm"

# configuration file
# config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-cms.yaml"
# config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-clic.yaml"
# config_file = "/mnt/home/ewulff/ceph/particleflow/parameters/pytorch/pyg-clic-v230.yaml"
config_file = "/mnt/home/ewulff/ceph/particleflow/parameters/pytorch/pyg-cld.yaml"

# Batch size scan
# bs_list = [256, 128, 64, 32, 16, 8, 4, 2]


# Number of GPUs scan
# prefix_string = "largebatch_study_gpus{}_linearscaledLR{}_epochs{}_bsm256_lamb_a100_cu124_fulldataset_"
# num_gpus_list = [1, 4]
# num_cpus_list = [32, 64]
# epochs_list = [20, 30, 40, 60, 80, 100]
# base_lr = 0.0002

# Epoch scan
prefix_string = "cld_epoch_scan_epochs{}_"
epochs_list = [100, 80, 60, 40, 30, 20]

# LR scan
# lr_list = [0.0016, 0.0032, 0.0064, 0.0128, 0.0256, 0.0512]
# prefix_string = "largebatch_study_gpus8_LR{}_epochs40_bsm256_lamb_"

# Run the jobs
processes = []

# BS scan
# for bs in bs_list:
#     replaceline_batch_multiplier(batch_file, bs)
#     time.sleep(1)
#     process = subprocess.run(
#         ["sbatch", batch_file, config_file, prefix_string.format(bs)],
#         stdout=subprocess.PIPE,
#         universal_newlines=True,
#     )
#     processes.append(process)

# LR scan
# for lr in lr_list:
#     replaceline_lr(batch_file, lr)
#     time.sleep(1)
#     process = subprocess.run(
#         ["sbatch", batch_file, config_file, prefix_string.format(lr)],
#         stdout=subprocess.PIPE,
#         universal_newlines=True,
#     )
#     processes.append(process)

# GPU scan
# for num_gpus, num_cpus, epochs in zip(num_gpus_list, num_cpus_list, epochs_list):
#     replaceline_num_cpus_per_task(batch_file, num_cpus)
#     replaceline_num_gpus(batch_file, num_gpus)
#     scaled_lr = base_lr * num_gpus
#     replaceline_lr(batch_file, scaled_lr)
#     replaceline_epochs(batch_file, epochs)
#     time.sleep(1)
#     process = subprocess.run(
#         ["sbatch", batch_file, config_file, prefix_string.format(num_gpus, scaled_lr, epochs)],
#         stdout=subprocess.PIPE,
#         universal_newlines=True,
#     )
#     processes.append(process)


# Epoch scan
for epochs in epochs_list:
    replaceline_epochs(batch_file, epochs)
    time.sleep(1)
    process = subprocess.run(
        ["sbatch", batch_file, config_file, prefix_string.format(epochs)],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    processes.append(process)

# Print the submitted jobs
for proc in processes:
    print(proc)
