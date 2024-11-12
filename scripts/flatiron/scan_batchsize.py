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


def replaceline_ntrain(fname, ntrain):
    replaceline_and_save(
        fname,
        findln="--ntrain",
        newline="    --ntrain {} \\".format(ntrain),
    )


# Configuration parameters
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_h100.slurm"
config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-cms.yaml"
prefix_string = "BSscan_mlpf_bs{}_"
bs_list = [4, 8, 12, 16, 24, 32]
bs_list = [8]

# Run the jobs
processes = []
for bs in bs_list:
    replaceline_batch_multiplier(batch_file, bs)
    time.sleep(1)
    process = subprocess.run(
        ["sbatch", batch_file, config_file, prefix_string.format(bs)],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    processes.append(process)

for proc in processes:
    print(proc)
