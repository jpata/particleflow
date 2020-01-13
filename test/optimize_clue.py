import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from benchmark_solution import load_elements_candidates, CLUE

def run_model(args):
    model, els, els_blid, dm = args
    els_blid_pred_clue = model.predict_blocks(els, dm)
    score_blocks_clue = model.assess_blocks(els_blid, els_blid_pred_clue, dm)
    #final_score = 2*score_blocks_clue["edge_recall"] + score_blocks_clue["edge_precision"]
    final_score = score_blocks_clue["adjusted_rand_score"]
    return final_score

def objective_function(pars):
    global all_elements, all_elements_blid, all_dms

    rho_ecal = float(pars[0])
    rho_hcal = float(pars[1])
    rho_hf = float(pars[2])
    delta_ecal = float(pars[3])
    delta_hcal = float(pars[4])
    delta_hf = float(pars[5])

    model = CLUE(
        rho_ecal=rho_ecal, rho_hcal=rho_hcal, rho_hf=rho_hf,
        delta_ecal=delta_ecal, delta_hcal=delta_hcal, delta_hf=delta_hf
    )
    with ProcessPoolExecutor(max_workers=16) as executor:
        scores = executor.map(run_model, [(model, a[0], a[1], a[2]) for a in zip(all_elements, all_elements_blid, all_dms)])
    #scores = map(run_model, [(model, a[0], a[1], a[2]) for a in zip(all_elements, all_elements_blid, all_dms)])
    scores = list(scores)
    m = np.mean(scores)
    print(pars, m)
    return -m

if __name__ == "__main__":
    fns = []
    for i in range(1,2):
        for j in range(200):
            fn = "data/TTbar_run3/step3_ntuple_{0}_ev{1}.npz".format(i, j)
            fns += [fn]

    print("loading data from {0} files".format(len(fns)))
 
    all_elements = []
    all_elements_blid = []
    all_dms = []
    for fn in fns:
        els, els_blid, cands, cands_blid, dm = load_elements_candidates(fn)
        all_elements += [els.copy()]
        all_elements_blid += [els_blid.copy()]
        all_dms += [dm.copy()]
    print("done loading data") 

    from skopt import dummy_minimize, gp_minimize
    dims = [
        np.arange(0.1, 1.0, 0.1),
        np.arange(0.1, 1.0, 0.1),
        np.arange(0.1, 1.0, 0.1),
        np.arange(0.001, 0.2, 0.001),
        np.arange(0.001, 0.2, 0.001),
        np.arange(0.001, 0.2, 0.001),
    ]
    ret = gp_minimize(objective_function, dims, n_calls=100)
    print(ret)
