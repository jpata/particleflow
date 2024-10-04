import glob
import os


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def write_script(infiles, outpath):
    s = []
    s += ["#!/bin/bash"]
    s += ["#SBATCH --partition main"]
    s += ["#SBATCH --cpus-per-task 1"]
    s += ["#SBATCH --mem-per-cpu 4G"]
    s += ["#SBATCH -o logs/slurm-%x-%j-%N.out"]
    s += ["set -e"]

    for inf in infiles:
        s += [
            "singularity exec -B /local /home/software/singularity/pytorch.simg:2024-08-18 python3 "
            + f"scripts/clic/postprocessing.py --input {inf} --outpath {outpath}"
        ]
    ret = "\n".join(s)

    ret += "\n"
    return ret


samples = [
    ("/local/joosep/clic_edm4hep/2024_07/p8_ee_qq_ecm380/root/", "/local/joosep/mlpf/clic_edm4hep/p8_ee_qq_ecm380/"),
    # ("/local/joosep/clic_edm4hep/2024_07/p8_ee_tt_ecm380/root/", "/local/joosep/mlpf/clic_edm4hep/p8_ee_tt_ecm380/"),
    ("/local/joosep/clic_edm4hep/2024_07/p8_ee_WW_fullhad_ecm380/root/", "/local/joosep/mlpf/clic_edm4hep/p8_ee_WW_fullhad_ecm380/"),
    # ("/local/joosep/clic_edm4hep/2024_07/p8_ee_ZH_Htautau_ecm380/root/", "/local/joosep/mlpf/clic_edm4hep/p8_ee_ZH_Htautau_ecm380/"),
    # ("/local/joosep/clic_edm4hep/2024_07/p8_ee_Z_Ztautau_ecm380/root/", "/local/joosep/mlpf/clic_edm4hep/p8_ee_Z_Ztautau_ecm380/"),
]

ichunk = 1
for sample, outpath in samples:
    infiles = list(glob.glob(f"{sample}/*.root"))
    os.makedirs(outpath, exist_ok=True)
    for infiles_chunk in chunks(infiles, 100):
        scr = write_script(infiles_chunk, outpath)
        ofname = f"jobscripts/postproc_{ichunk}.sh"
        with open(ofname, "w") as outfi:
            outfi.write(scr)
        ichunk += 1
