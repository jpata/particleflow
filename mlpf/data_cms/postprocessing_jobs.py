import os
import glob


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def write_script(infiles, outfiles):
    s = []
    s += ["#!/bin/bash"]
    s += ["#SBATCH --partition short"]
    s += ["#SBATCH --cpus-per-task 1"]
    s += ["#SBATCH --mem-per-cpu 4G"]
    s += ["#SBATCH -o logs/slurm-%x-%j-%N.out"]
    s += ["set -e"]

    for inf, outf in zip(infiles, outfiles):
        outpath = os.path.dirname(outf)

        outf_no_bzip = outf.replace(".pkl.bz2", ".pkl")
        s += [f"if [ ! -f {outf} ]; then"]
        s += [
            "  singularity exec -B /local /home/software/singularity/pytorch.simg:2024-08-18"
            + f" python3 mlpf/data_cms/postprocessing2.py --input {inf} --outpath {outpath}"
        ]
        s += [f"  bzip2 -z {outf_no_bzip}"]
        s += ["fi"]
    ret = "\n".join(s)
    return ret


samples = [
    "/local/joosep/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi",
]

ichunk = 1
for sample in samples:
    infiles = list(glob.glob(f"{sample}/root/pfntuple*.root"))
    for infiles_chunk in chunks(infiles, 10):
        outfiles_chunk = [inf.replace(".root", ".pkl.bz2").replace("/root/", "/raw2/") for inf in infiles_chunk]
        os.makedirs(os.path.dirname(outfiles_chunk[0]), exist_ok=True)
        scr = write_script(infiles_chunk, outfiles_chunk)
        ofname = f"jobscripts/postproc_{ichunk}.sh"
        with open(ofname, "w") as outfi:
            outfi.write(scr)
        ichunk += 1
