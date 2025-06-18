#!/usr/bin/env python3
from __future__ import print_function

import os

outdir = "/local/joosep/mlpf/cms/20250618_cmssw_15_0_5_f8ae2f/"

samples = [
    # ("QCDForPF_13p6TeV_TuneCUETP8M1_cfi",                          700000, 705050, "genjob_pu55to75_val.sh", outdir + "/pu55to75_val"),
    # ("PhotonJet_Pt_10_13p6TeV_TuneCUETP8M1_cfi",                   900000, 905050, "genjob_pu55to75_val.sh", outdir + "/pu55to75_val"),
    ("TTbar_13p6TeV_TuneCUETP8M1_cfi", 800000, 801050, "genjob_pu55to75_val.sh", outdir + "/pu55to75_val"),
    ("TTbar_13p6TeV_TuneCUETP8M1_cfi", 800000, 801050, "genjob_nopu_val.sh", outdir + "/nopu_val"),
]

if __name__ == "__main__":

    for samp, seed0, seed1, script, this_outdir in samples:
        os.makedirs(this_outdir + "/" + samp + "/raw", exist_ok=True)
        os.makedirs(this_outdir + "/" + samp + "/root", exist_ok=True)

        for seed in range(seed0, seed1):
            p = this_outdir + "/" + samp + "/root/step2_{}.root".format(seed)
            if not os.path.isfile(p):
                print(
                    f"sbatch --mem-per-cpu 4G --partition main --time 20:00:00 --cpus-per-task 1 scripts/tallinn/cmssw-el8.sh mlpf/data/cms/{script} {samp} {seed}"
                )
