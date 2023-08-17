import os

# check for file presence in this path
outpath = "/local/joosep/clic_edm4hep/"

# pythia card, start seed, end seed
samples = [
    ("p8_ee_tt_ecm380", 1, 10011),
    ("p8_ee_qq_ecm380", 100001, 120011),
    ("p8_ee_ZH_Htautau_ecm380", 200001, 210011),
    ("p8_ee_WW_fullhad_ecm380", 300001, 310011),
]

samples_pu = [
    ("p8_ee_tt_ecm380", 1, 10001),
]

samples_gun = [
    ("neutron", 1, 101),
    ("kaon0L", 1, 101),
    ("pi-", 1, 101),
    ("pi+", 1, 101),
    ("pi0", 1, 101),
    ("mu-", 1, 101),
    ("mu+", 1, 101),
    ("e-", 1, 101),
    ("e+", 1, 101),
    ("gamma", 1, 101),
]

if __name__ == "__main__":
    # basic samples
    for sname, seed0, seed1 in samples:
        for seed in range(seed0, seed1):
            # check if output file exists, and print out batch submission if it doesn't
            if not os.path.isfile("{}/{}/reco_{}_{}.root".format(outpath, sname, sname, seed)):
                print("sbatch run_sim.sh {} {}".format(seed, sname))

    # PU
    for sname, seed0, seed1 in samples_pu:
        for seed in range(seed0, seed1):
            # check if output file exists, and print out batch submission if it doesn't
            if not os.path.isfile("{}/{}_PU10/reco_{}_{}.root".format(outpath, sname, sname, seed)):
                print("sbatch run_sim_pu.sh {} {} p8_ee_gg_ecm380".format(seed, sname))

    # gun
    for sname, seed0, seed1 in samples_gun:
        for seed in range(seed0, seed1):
            # check if output file exists, and print out batch submission if it doesn't
            if not os.path.isfile("{}/{}/reco_{}_{}.root".format(outpath, sname, sname, seed)):
                print("sbatch run_sim_gun.sh {} {}".format(seed, sname))
