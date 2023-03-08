import os

#check for file presence in this path
outpath = "/local/joosep/clic_edm4hep_2023_03_03/"

#pythia card, start seed, end seed
samples = [
    ("p8_ee_tt_ecm380",              1,   2011),
    ("p8_ee_qq_ecm380",         100001, 102011),
    ("p8_ee_ZH_Htautau_ecm380", 200001, 202011),
]

samples_pu = [
    ("p8_ee_tt_ecm380",              1,   101),
]

if __name__ == "__main__":
    #for sname, seed0, seed1 in samples:
    #    for seed in range(seed0, seed1):
    #        #check if output file exists, and print out batch submission if it doesn't
    #        if not os.path.isfile("{}/{}/reco_{}_{}.root".format(outpath, sname, sname, seed)):
    #            print("sbatch run_sim.sh {} {}".format(seed, sname)) 
    for sname, seed0, seed1 in samples_pu:
        for seed in range(seed0, seed1):
            #check if output file exists, and print out batch submission if it doesn't
            if not os.path.isfile("{}/{}_PU10/reco_{}_{}.root".format(outpath, sname, sname, seed)):
                print("sbatch run_sim_pu.sh {} {} p8_ee_gg_ecm380".format(seed, sname)) 
