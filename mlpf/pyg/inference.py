import os
import time
from pathlib import Path

import awkward
import fastjet
import mplhep
import numpy as np
import torch
import tqdm
import utils
import vector
from jet_utils import build_dummy_array, match_two_jet_collections
from plotting.plot_utils import (
    compute_met_and_ratio,
    format_dataset_name,
    load_eval_data,
    plot_jet_ratio,
    plot_met,
    plot_met_ratio,
    plot_num_elements,
    plot_particles,
    plot_sum_energy,
)

from .logger import _logger
from .utils import CLASS_NAMES

jet_pt = 5.0
jet_match_dr = 0.1


def particle_array_to_awkward(batch_ids, arr_id, arr_p4):
    ret = {
        "cls_id": arr_id,
        "pt": arr_p4["pt"],
        "eta": arr_p4["eta"],
        "sin_phi": arr_p4["sin_phi"],
        "cos_phi": arr_p4["cos_phi"],
        "energy": arr_p4["e"],
    }

    ret = awkward.from_iter([{k: ret[k][batch_ids == b] for k in ret.keys()} for b in np.unique(batch_ids)])
    return ret


@torch.no_grad()
def run_predictions(rank, model, loader, sample, outpath, jetdef):
    """Runs inference on the given sample and stores the output as .parquet files."""

    ti = time.time()

    for i, event in tqdm.tqdm(enumerate(loader), total=len(loader)):
        event.X = event.X.to(rank)

        ygen = utils.unpack_target(event.ygen)
        ycand = utils.unpack_target(event.ycand)

        ypred = utils.unpack_predictions(model(event))

        for k, v in ypred.items():
            ypred[k] = v.detach().cpu()

        awkvals = {
            "gen": ygen,
            "cand": ycand,
            "pred": ypred,
        }

        jets_coll = {}
        Xs, p4s = [], []
        # loop over each batch to disentangle the events
        batch_ids = event.batch.cpu().numpy()
        for _ibatch in np.unique(batch_ids):
            for typ in ["gen", "cand", "pred"]:
                msk_batch = batch_ids == _ibatch

                msk = (awkvals[typ]["ids"][msk_batch] != 0).numpy()

                Xs.append(event.X[msk_batch].cpu().numpy())
                p4s.append(awkvals[typ]["p4"][msk_batch][msk].numpy())

            Xs = awkward.from_iter(Xs)

            # in case of no predicted particles in the batch
            if torch.sum(awkvals[typ]["ids"] != 0) == 0:
                pt = build_dummy_array(len(p4s), np.float64)
                eta = build_dummy_array(len(p4s), np.float64)
                phi = build_dummy_array(len(p4s), np.float64)
                energy = build_dummy_array(len(p4s), np.float64)
            else:
                p4s = awkward.from_iter(p4s)
                pt = p4s[:, :, 0]
                eta = p4s[:, :, 1]
                phi = p4s[:, :, 2]
                energy = p4s[:, :, 3]

            awk_p4s = vector.awk(awkward.zip({"pt": pt, "eta": eta, "phi": phi, "e": energy}))

            cluster = fastjet.ClusterSequence(awkward.Array(awk_p4s.to_xyzt()), jetdef)
            jets_coll[typ] = cluster.inclusive_jets(min_pt=jet_pt)

        gen_to_pred = match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr)
        gen_to_cand = match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr)

        matched_jets = awkward.Array({"gen_to_pred": gen_to_pred, "gen_to_cand": gen_to_cand})

        awkvals = {
            "gen": awkward.from_iter([{k: ygen[k][batch_ids == b] for k in ygen.keys()} for b in np.unique(batch_ids)]),
            "cand": awkward.from_iter([{k: ycand[k][batch_ids == b] for k in ygen.keys()} for b in np.unique(batch_ids)]),
            "pred": awkward.from_iter([{k: ypred[k][batch_ids == b] for k in ygen.keys()} for b in np.unique(batch_ids)]),
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
            f"{outpath}/ypred/{sample}/pred_{rank}_{i}.parquet",
        )
        _logger.info(f"Saved predictions at {outpath}/ypred/{sample}/pred_{rank}_{i}.parquet")

    _logger.info(f"Time taken to make predictions on device {rank} is: {((time.time() - ti) / 60):.2f} min")


def make_plots(outpath, sample, dataset):
    """Uses the predictions stored as .parquet files (see above) to make plots."""

    mplhep.set_style(mplhep.styles.CMS)

    class_names = CLASS_NAMES[dataset]

    _title = format_dataset_name(sample)  # use the dataset names from the common nomenclature

    if not os.path.isdir(f"{outpath}/plots/"):
        os.makedirs(f"{outpath}/plots/")
    if not os.path.isdir(f"{outpath}/plots/{sample}"):
        os.makedirs(f"{outpath}/plots/{sample}")

    plots_path = Path(f"{outpath}/plots/{sample}/")
    pred_path = Path(f"{outpath}/ypred/{sample}/")

    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), -1)

    plot_num_elements(X, cp_dir=plots_path, title=_title)
    plot_sum_energy(yvals, class_names, cp_dir=plots_path, title=_title)

    plot_jet_ratio(yvals, cp_dir=plots_path, title=_title)

    met_data = compute_met_and_ratio(yvals)
    plot_met(met_data, cp_dir=plots_path, title=_title)
    plot_met_ratio(met_data, cp_dir=plots_path, title=_title)

    plot_particles(yvals, cp_dir=plots_path, title=_title)
