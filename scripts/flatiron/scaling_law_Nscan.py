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


def replaceline_ntrain(fname, ntrain):
    replaceline_and_save(
        fname,
        findln="--ntrain",
        newline="    --ntrain {}".format(ntrain),
    )


def replaceline_width(fname, width):
    replaceline_and_save(
        fname,
        findln="--width",
        newline="    --width {} \\".format(width),
    )


def replaceline_embedding_dim(fname, embedding_dim):
    replaceline_and_save(
        fname,
        findln="--embedding-dim",
        newline="    --embedding-dim {} \\".format(embedding_dim),
    )


def replaceline_num_convs(fname, num_convs):
    replaceline_and_save(
        fname,
        findln="--num-convs",
        newline="    --num-convs {} \\".format(num_convs),
    )


# Configuration parameters
batch_file = "/mnt/ceph/users/ewulff/particleflow/scripts/flatiron/pt_raytrain_a100.slurm"
config_file = "/mnt/ceph/users/ewulff/particleflow/parameters/pytorch/pyg-clic-hits.yaml"
prefix_string = "Nscan_mlpf_attention_nconvs{}_width{}_ntrain131072_epochs20"
# prefix_string = "Nscan_mlpf_gnnlsh_nconvs{}_width{}_ntrain131072"

nconvs_width_list = [
    # (1, 64),
    # (1, 128),
    # (1, 256),
    # (1, 512),
    (2, 64),
    (2, 128),
    (2, 256),
    (2, 512),
    # (3, 64),
    # (3, 128),
    # (3, 256),
    # (3, 512),
    # (4, 64),
    # (4, 128),
    # (4, 256),
    # (4, 512),
]

nconvs_width_list = list(reversed(nconvs_width_list))  # reverse list to submit largest models first

# Run the jobs
processes = []
for nconvs, width in nconvs_width_list:
    replaceline_width(batch_file, width)
    replaceline_embedding_dim(batch_file, width)
    replaceline_num_convs(batch_file, nconvs)
    time.sleep(0.5)
    process = subprocess.run(
        ["sbatch", batch_file, config_file, prefix_string.format(nconvs, width)],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    processes.append(process)

for proc in processes:
    print(proc)
