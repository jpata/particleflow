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
        findln="--batch-multiplier",
        newline=" --batch-multiplier {}".format(bmultiplier),
    )


def replaceline_ntrain(fname, ntrain):
    replaceline_and_save(
        fname,
        findln="--ntrain",
        newline="    --ntrain {}".format(ntrain),
    )


# Configuration parameters
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_a100.slurm"
config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-clic-hits.yaml"
prefix_string = "Dscan_mlpf_gnnlsh_ntrain{}_lr1eneg4_run3"
# ntrain_list = [524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024]
ntrain_list = [65536, 32768, 16384, 8192, 4096, 2048, 1024]
ntrain_list = [1024]

# Run the jobs
processes = []
for ntrain in ntrain_list:
    replaceline_ntrain(batch_file, ntrain)
    time.sleep(1)
    process = subprocess.run(
        ["sbatch", batch_file, config_file, prefix_string.format(ntrain)],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    processes.append(process)

for proc in processes:
    print(proc)
