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


def replaceline_ntrain(fname, ntrain):
    replaceline_and_save(
        fname,
        findln="--ntrain",
        newline="    --ntrain {} \\".format(ntrain),
    )


# Configuration parameters
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_a100.slurm"

# configuration file
# config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-cms.yaml"
config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-clic.yaml"

# Batch size scan
# prefix_string = "BSscan_mlpf_bs{}_"
# bs_list = [4, 8, 12, 16, 24, 32]
# bs_list = [32, 64, 128, 256]

# Number of GPUs scan
prefix_string = "GPUscan_mlpf_gpus{}_"
num_gpus_list = [1, 2, 4]

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

# GPU scan
for num_gpus in num_gpus_list:
    replaceline_num_gpus(batch_file, num_gpus)
    time.sleep(1)
    process = subprocess.run(
        ["sbatch", batch_file, config_file, prefix_string.format(num_gpus)],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    processes.append(process)

# Print the submitted jobs
for proc in processes:
    print(proc)
