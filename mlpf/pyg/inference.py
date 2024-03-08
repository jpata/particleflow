import os
import time
from pathlib import Path

import awkward
import fastjet
import mplhep
import numpy as np
import torch
import tqdm
import vector
from jet_utils import build_dummy_array, match_two_jet_collections
from plotting.plot_utils import (
    compute_met_and_ratio,
    format_dataset_name,
    load_eval_data,
    plot_jets,
    plot_jet_ratio,
    plot_jet_response_binned,
    plot_jet_response_binned_eta,
    plot_jet_response_binned_separate,
    plot_met,
    plot_met_ratio,
    plot_met_response_binned,
    plot_num_elements,
    plot_particle_multiplicity,
    plot_particles,
    plot_sum_energy,
)
import torch_geometric
from torch_geometric.data import Batch

from .logger import _logger
from .utils import CLASS_NAMES, unpack_predictions, unpack_target


def predict_one_batch(conv_type, model, i, batch, rank, jetdef, jet_ptcut, jet_match_dr, outpath, dir_name, sample):
    if conv_type != "gravnet":
        X_pad, mask = torch_geometric.utils.to_dense_batch(batch.X, batch.batch)
        batch_pad = Batch(X=X_pad, mask=mask).to(rank)
        ypred = model(batch_pad.X, batch_pad.mask)
        ypred = ypred[0][mask], ypred[1][mask]
    else:
        _batch = batch.to(rank)
        ypred = model(_batch.X, _batch.batch)

    ygen = unpack_target(batch.ygen)
    ycand = unpack_target(batch.ycand)
    ypred = unpack_predictions(ypred)

    for k, v in ygen.items():
        ygen[k] = v.detach().cpu()
    for k, v in ycand.items():
        ycand[k] = v.detach().cpu()
    for k, v in ypred.items():
        ypred[k] = v.detach().cpu()

    # loop over the batch to disentangle the events
    batch_ids = batch.batch.cpu().numpy()

    jets_coll = {}

    cs = np.unique(batch_ids, return_counts=True)[1]
    Xs = awkward.unflatten(awkward.from_numpy(batch.X.numpy()), cs)

    for typ, ydata in zip(["gen", "cand"], [ygen, ycand]):
        clsid = awkward.unflatten(ydata["cls_id"], cs)
        msk = clsid != 0
        p4 = awkward.unflatten(ydata["p4"], cs)
        vec = vector.awk(
            awkward.zip(
                {
                    "pt": p4[msk][:, :, 0],
                    "eta": p4[msk][:, :, 1],
                    "phi": p4[msk][:, :, 2],
                    "e": p4[msk][:, :, 3],
                }
            )
        )
        cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
        jets_coll[typ] = cluster.inclusive_jets(min_pt=jet_ptcut)

    # in case of no predicted particles in the batch
    if torch.sum(ypred["cls_id"] != 0) == 0:
        vec = vector.awk(
            awkward.zip(
                {
                    "pt": build_dummy_array(len(vec), np.float64),
                    "eta": build_dummy_array(len(vec), np.float64),
                    "phi": build_dummy_array(len(vec), np.float64),
                    "e": build_dummy_array(len(vec), np.float64),
                }
            )
        )
    else:
        clsid = awkward.unflatten(ypred["cls_id"], cs)
        msk = clsid != 0
        p4 = awkward.unflatten(ypred["p4"], cs)

        vec = vector.awk(
            awkward.zip(
                {
                    "pt": p4[msk][:, :, 0],
                    "eta": p4[msk][:, :, 1],
                    "phi": p4[msk][:, :, 2],
                    "e": p4[msk][:, :, 3],
                }
            )
        )

    cluster = fastjet.ClusterSequence(vec.to_xyzt(), jetdef)
    jets_coll["pred"] = cluster.inclusive_jets(min_pt=jet_ptcut)

    gen_to_pred = match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr)
    gen_to_cand = match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr)

    matched_jets = awkward.Array({"gen_to_pred": gen_to_pred, "gen_to_cand": gen_to_cand})

    awkvals = {
        "gen": awkward.from_iter([{k: ygen[k][batch_ids == b] for k in ygen.keys()} for b in np.unique(batch_ids)]),
        "cand": awkward.from_iter([{k: ycand[k][batch_ids == b] for k in ycand.keys()} for b in np.unique(batch_ids)]),
        "pred": awkward.from_iter([{k: ypred[k][batch_ids == b] for k in ypred.keys()} for b in np.unique(batch_ids)]),
    }

    awkward.to_parquet(
        awkward.Array(
            {
                "inputs": Xs,
                "particles": awkvals,
                "jets": jets_coll,
                "matched_jets": matched_jets,
            }
        ),
        f"{outpath}/preds{dir_name}/{sample}/pred_{rank}_{i}.parquet",
    )
    _logger.info(f"Saved predictions at {outpath}/preds{dir_name}/{sample}/pred_{rank}_{i}.parquet")


@torch.no_grad()
def run_predictions(world_size, rank, model, loader, sample, outpath, jetdef, jet_ptcut=15.0, jet_match_dr=0.1, dir_name=""):
    """Runs inference on the given sample and stores the output as .parquet files."""
    if world_size > 1:
        conv_type = model.module.conv_type
    else:
        conv_type = model.conv_type

    model.eval()

    # only show progress bar on rank 0
    if (world_size > 1) and (rank != 0):
        iterator = enumerate(loader)
    else:
        iterator = tqdm.tqdm(enumerate(loader), total=len(loader))

    ti = time.time()
    for i, batch in iterator:
        predict_one_batch(conv_type, model, i, batch, rank, jetdef, jet_ptcut, jet_match_dr, outpath, dir_name, sample)

    _logger.info(f"Time taken to make predictions on device {rank} is: {((time.time() - ti) / 60):.2f} min")


def make_plots(outpath, sample, dataset, dir_name=""):
    """Uses the predictions stored as .parquet files (see above) to make plots."""

    mplhep.set_style(mplhep.styles.CMS)

    os.system(f"mkdir -p {outpath}/plots{dir_name}/{sample}")

    plots_path = Path(f"{outpath}/plots{dir_name}/{sample}/")
    pred_path = Path(f"{outpath}/preds{dir_name}/{sample}/")

    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), -1)

    title = format_dataset_name(sample)
    plot_num_elements(X, cp_dir=plots_path, title=title)
    plot_sum_energy(yvals, CLASS_NAMES[dataset], cp_dir=plots_path, title=title)
    plot_particle_multiplicity(X, yvals, CLASS_NAMES[dataset], cp_dir=plots_path, title=title)

    plot_jets(
        yvals,
        cp_dir=plots_path,
        title=title,
    )
    plot_jet_ratio(
        yvals,
        cp_dir=plots_path,
        title=title,
        bins=np.linspace(0, 5, 100),
        logy=True,
    )
    plot_jet_ratio(
        yvals,
        cp_dir=plots_path,
        title=title,
        bins=np.linspace(0.5, 1.5, 100),
        logy=False,
        file_modifier="_bins_0p5_1p5",
    )
    plot_jet_response_binned(yvals, cp_dir=plots_path, title=title)
    plot_jet_response_binned_eta(yvals, cp_dir=plots_path, title=title)
    plot_jet_response_binned_separate(yvals, cp_dir=plots_path, title=title)

    met_data = compute_met_and_ratio(yvals)
    plot_met(met_data, cp_dir=plots_path, title=title)
    plot_met_ratio(met_data, cp_dir=plots_path, title=title)
    plot_met_ratio(
        met_data,
        cp_dir=plots_path,
        title=title,
        bins=np.linspace(0, 20, 100),
        logy=True,
    )
    plot_met_ratio(
        met_data,
        cp_dir=plots_path,
        title=title,
        bins=np.linspace(0, 2, 100),
        logy=False,
        file_modifier="_bins_0_2",
    )
    plot_met_ratio(
        met_data,
        cp_dir=plots_path,
        title=title,
        bins=np.linspace(0, 5, 100),
        logy=False,
        file_modifier="_bins_0_5",
    )
    plot_met_response_binned(met_data, cp_dir=plots_path, title=title)

    plot_particles(yvals, cp_dir=plots_path, title=format_dataset_name(sample))
