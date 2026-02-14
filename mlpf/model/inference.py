import os
import time
from pathlib import Path
import sys
import gc

import awkward
import fastjet
import mplhep
import numpy as np
import torch
import tqdm
import vector
from mlpf.jet_utils import match_two_jet_collections
from mlpf.plotting.plot_utils import (
    # get_class_names,
    # compute_met_and_ratio,
    load_eval_data,
    plot_jets,
    plot_jet_ratio,
    # plot_jet_response_binned,
    # plot_jet_response_binned_vstarget,
    # plot_jet_response_binned_eta,
    # plot_met,
    # plot_met_ratio,
    # plot_met_response_binned,
    plot_num_elements,
    # plot_particles,
    # plot_particle_ratio,
    # plot_particle_response,
    # plot_pu_fraction,
)

from mlpf.logger import _logger
from mlpf.model.utils import unpack_predictions, unpack_target


def predict_one_batch(conv_type, model, i, batch, rank, jetdef, jet_ptcut, jet_match_dr, outpath, dir_name, sample):

    # skip prediction if output exists
    outfile = f"{outpath}/preds{dir_name}/{sample}/pred_{rank}_{i}.parquet"
    if os.path.isfile(outfile):
        return

    # run model on batch
    batch = batch.to(rank)
    ypred = model(batch.X, batch.mask)

    # convert all outputs to float32 in case running in float16 or bfloat16
    ypred = tuple([y.to(torch.float32) for y in ypred])

    # transform log (pt/elempt) -> pt
    pred_cls = torch.argmax(ypred[0], axis=-1)
    ypred[2][..., 0] = torch.exp(ypred[2][..., 0]) * batch.X[..., 1]
    ypred[2][..., 0][pred_cls == 0] = 0

    # transform log (E/elemE) -> E
    ypred[2][..., 4] = torch.exp(ypred[2][..., 4]) * batch.X[..., 5]
    ypred[2][..., 4][pred_cls == 0] = 0

    batch.ytarget[..., 2] = batch.ytarget_pt_orig
    batch.ytarget[..., 6] = batch.ytarget_e_orig

    ytarget = unpack_target(batch.ytarget.to(torch.float32), model)
    ycand = unpack_target(batch.ycand.to(torch.float32), model)
    ypred = unpack_predictions(ypred)
    ypred["ispu"] = torch.softmax(ypred["ispu"], axis=-1)[:, :, -1]

    genjets_msk = batch.genjets[:, :, 0].cpu() > jet_ptcut
    genjets = awkward.unflatten(batch.genjets.cpu().to(torch.float64)[genjets_msk], torch.sum(genjets_msk, axis=1))
    genjets = vector.awk(
        awkward.zip(
            {
                "pt": genjets[:, :, 0],
                "eta": genjets[:, :, 1],
                "phi": genjets[:, :, 2],
                "e": genjets[:, :, 3],
            }
        )
    )
    genjets = vector.awk(awkward.zip({"px": genjets.px, "py": genjets.py, "pz": genjets.pz, "E": genjets.e}))

    jets_coll = {}
    jets_coll["gen"] = genjets

    # we need to turn 3d-padded arrays into awkward arrays
    # first, flatten events across batch dim with padding mask
    X = batch.X[batch.mask].cpu().contiguous().numpy()
    for k, v in ytarget.items():
        ytarget[k] = v[batch.mask].detach().cpu().contiguous().numpy()
    for k, v in ycand.items():
        ycand[k] = v[batch.mask].detach().cpu().contiguous().numpy()
    for k, v in ypred.items():
        ypred[k] = v[batch.mask].detach().cpu().contiguous().numpy()

    # second, create awkward arrays according to the counts of not padded elements
    counts = torch.sum(batch.mask, axis=1).cpu().numpy()
    awkvals = {}
    for flat_arr, typ in [(ytarget, "target"), (ycand, "cand"), (ypred, "pred")]:
        awk_arr = awkward.Array({k: flat_arr[k] for k in flat_arr.keys()})
        awkvals[typ] = awkward.unflatten(awk_arr, counts)
    Xs = awkward.unflatten(awkward.from_numpy(X), counts)

    # now cluster jets
    for typ, ydata in zip(["cand", "target", "pred", "pred_nopu"], [awkvals["cand"], awkvals["target"], awkvals["pred"], awkvals["pred"]]):
        msk = ydata["cls_id"] != 0
        # placeholder cut on the PU frac prediction
        if typ == "pred_nopu":
            msk1 = ydata["ispu"] < 0.8
            msk = msk & msk1
        vec = vector.awk(
            awkward.zip(
                {
                    "pt": ydata["pt"][msk],
                    "eta": ydata["eta"][msk],
                    "phi": ydata["phi"][msk],
                    "e": ydata["energy"][msk],
                }
            )
        )
        cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
        jets = cluster.inclusive_jets(min_pt=jet_ptcut)
        jets_coll[typ] = vector.awk(awkward.zip({"px": jets.px, "py": jets.py, "pz": jets.pz, "E": jets.e}))

    matched_jets = awkward.Array(
        {
            "gen_to_pred_nopu": match_two_jet_collections(jets_coll, "gen", "pred_nopu", jet_match_dr),
            "gen_to_pred": match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr),
            "gen_to_cand": match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr),
            "gen_to_target": match_two_jet_collections(jets_coll, "gen", "target", jet_match_dr),
            "target_to_cand": match_two_jet_collections(jets_coll, "target", "cand", jet_match_dr),
            "target_to_pred": match_two_jet_collections(jets_coll, "target", "pred", jet_match_dr),
        }
    )

    outdict = {
        "inputs": Xs,
        "particles": awkvals,
        "jets": jets_coll,
        "matched_jets": matched_jets,
        "genmet": batch.genmet.cpu(),
    }
    if batch.pythia is not None:
        outdict["pythia"] = batch.pythia.cpu()

    awkward.to_parquet(
        awkward.Array(outdict),
        outfile,
    )
    _logger.info(f"Saved predictions at {outfile}")


def predict_one_batch_args(args):
    predict_one_batch(*args)


@torch.no_grad()
def run_predictions(world_size, rank, model, loader, sample, outpath, jetdef, jet_ptcut=15.0, jet_match_dr=0.1, dir_name=""):
    """Runs inference on the given sample and stores the output as .parquet files."""
    if world_size > 1:
        conv_type = model.module.conv_type
    else:
        conv_type = model.conv_type

    model.eval()

    is_interactive = ((world_size <= 1) or (rank == 0)) and sys.stdout.isatty()
    iterator = enumerate(loader)
    if is_interactive:
        iterator = tqdm.tqdm(iterator, total=len(loader), desc=f"Running predictions on sample {sample} on rank={rank}")

    ti = time.time()
    for i, batch in iterator:
        predict_one_batch(conv_type, model, i, batch, rank, jetdef, jet_ptcut, jet_match_dr, outpath, dir_name, sample)
    tf = time.time()
    time_total_min = (tf - ti) / 60.0

    _logger.info(f"Time taken to make predictions on device {rank} is: {time_total_min:.2f} min")


def make_plots(outpath, sample, dataset, dir_name="", num_test_events=None):
    """Uses the predictions stored as .parquet files from run_predictions to make plots."""
    import matplotlib.pyplot as plt

    ret_dict = {}
    mplhep.style.use(mplhep.styles.CMS)
    # class_names = get_class_names(sample)
    os.system(f"mkdir -p {outpath}/plots{dir_name}/{sample}")

    plots_path = Path(f"{outpath}/plots{dir_name}/{sample}/")
    pred_path = Path(f"{outpath}/preds{dir_name}/{sample}/")

    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), num_test_events)
    _logger.info(f"Loaded data for plotting from {pred_path}")

    plot_num_elements(X, cp_dir=plots_path)
    _logger.info("Plotted number of elements")

    # plot_elements(X, yvals, cp_dir=plots_path, dataset=dataset, sample=sample)

    plot_jets(
        yvals,
        cp_dir=plots_path,
        dataset=dataset,
        sample=sample,
    )
    _logger.info("Plotted jets")
    ret_dict["jet_ratio"] = plot_jet_ratio(
        yvals,
        cp_dir=plots_path,
        bins=np.linspace(0, 5, 500),
        logy=True,
        dataset=dataset,
        sample=sample,
    )

    # commented out to save memory and make the validation runtime faster
    # _logger.info("Plotted jet ratio")
    # plot_jet_ratio(
    #     yvals,
    #     cp_dir=plots_path,
    #     bins=np.linspace(0.5, 1.5, 500),
    #     logy=True,
    #     file_modifier="_bins_0p5_1p5",
    #     dataset=dataset,
    #     sample=sample,
    # )
    # _logger.info("Plotted jet ratio with bins 0.5-1.5")
    # plot_jet_ratio(
    #     yvals,
    #     cp_dir=plots_path,
    #     bins=np.linspace(0, 2, 500),
    #     logy=True,
    #     file_modifier="_bins_0_2",
    #     dataset=dataset,
    #     sample=sample,
    # )
    # _logger.info("Plotted jet ratio with bins 0-2")
    # plot_jet_response_binned(yvals, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted binned jet response")
    # plot_jet_response_binned_vstarget(yvals, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted binned jet response vs target")
    # plot_jet_response_binned_eta(yvals, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted binned jet response vs eta")
    # plot_jet_response_binned_separate(yvals, cp_dir=plots_path, title=title)

    # met_data = compute_met_and_ratio(yvals)
    # _logger.info("Computed MET and ratio")
    # plot_met(met_data, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted MET")
    # plot_met_ratio(met_data, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted MET ratio")
    # plot_met_ratio(met_data, cp_dir=plots_path, bins=np.linspace(0, 20, 100), logy=True, dataset=dataset, sample=sample)
    # _logger.info("Plotted MET ratio with bins 0-20")
    # plot_met_ratio(
    #     met_data,
    #     cp_dir=plots_path,
    #     bins=np.linspace(0, 2, 500),
    #     logy=False,
    #     file_modifier="_bins_0_2",
    #     dataset=dataset,
    #     sample=sample,
    # )
    # _logger.info("Plotted MET ratio with bins 0-2")
    # plot_met_ratio(
    #     met_data,
    #     cp_dir=plots_path,
    #     bins=np.linspace(0, 5, 500),
    #     logy=False,
    #     file_modifier="_bins_0_5",
    #     dataset=dataset,
    #     sample=sample,
    # )
    # _logger.info("Plotted MET ratio with bins 0-5")
    # plot_met_response_binned(met_data, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted binned MET response")

    # plot_particles(yvals, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted particles")
    # plot_particle_ratio(yvals, class_names, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted particle ratio")
    # plot_particle_response(X, yvals, class_names, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted particle response")
    # plot_pu_fraction(yvals, cp_dir=plots_path, dataset=dataset, sample=sample)
    # _logger.info("Plotted PU fraction")

    del X, yvals
    plt.close("all")
    gc.collect()

    return ret_dict
