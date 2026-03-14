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
    replaceline_and_save(fname, findln="CUDA_VISIBLE_DEVICES", newline="export CUDA_VISIBLE_DEVICES={}".format(get_CVD_string(devices)))


def replaceline_hvd_script(fname, devices):
    replaceline_and_save(
        fname,
        findln="horovodrun -np",
        newline=" horovodrun -np {} -H localhost:{} \\".format(devices, devices),
    )


def replaceline_batch_multiplier(fname, bmultiplier):
    replaceline_and_save(
        fname,
        findln="--batch-multiplier",
        newline=" --batch-multiplier {}".format(bmultiplier),
    )


# Configuration parameters
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pipeline_train_8GPUs_singularity_hvd.slurm"
# config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/clic-gnn-tuned-v130.yaml"
config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/clic-test.yaml"
# config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/clic-transformer-tuned-v130.yaml"

# L40s vs H100 test
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_h100.slurm"
# batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_l40s.slurm"

prefix_string = "scale_test_dataV171_H100_{}GPUs_"
prefix_string = "scale_test_dataV171_L40s_{}GPUs_"
prefix_string = "scale_test_dataV171_H100_genoa_{}GPUs_"
min_gpus = 1
max_gpus = 8


# Run the jobs
processes = []
for i in range(min_gpus, max_gpus + 1):
    replaceline_CUDA_VISIBLE_DEVICES(batch_file, i)
    # replaceline_hvd_script(batch_file, devices=i)
    # replaceline_batch_multiplier(batch_file, bmultiplier=i)
    time.sleep(1)
    process = subprocess.run(["sbatch", batch_file, config_file, prefix_string.format(i)], stdout=subprocess.PIPE, universal_newlines=True)
    processes.append(process)

for proc in processes:
    print(proc)
