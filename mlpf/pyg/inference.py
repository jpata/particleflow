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
    plot_jet_ratio,
    plot_met,
    plot_met_ratio,
    plot_num_elements,
    plot_particles,
    plot_sum_energy,
)

from .logger import _logger
from .utils import CLASS_NAMES


@torch.no_grad()
def run_predictions(
    rank, world_size, is_distributed, model, loader, sample, outpath, jetdef, jet_ptcut=5.0, jet_match_dr=0.1
):
    """Runs inference on the given sample and stores the output as .parquet files."""

    model.eval()

    ti = time.time()
    for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader)):
        if (world_size > 1) and not is_distributed:  # torch_geometric.nn.data_parallel is given a list of Batch()
            X = batch
        else:
            X = batch.to(rank)

        ygen, ycand, ypred = model(X)

        for k, v in ygen.items():
            ygen[k] = v.detach().cpu()
        for k, v in ycand.items():
            ycand[k] = v.detach().cpu()
        for k, v in ypred.items():
            ypred[k] = v.detach().cpu()

        # loop over the batch to disentangle the events
        batch_ids = X.batch.cpu().numpy()

        jets_coll = {}
        Xs, p4s = [], {"gen": [], "cand": [], "pred": []}
        for _ibatch in np.unique(batch_ids):
            msk_batch = batch_ids == _ibatch

            Xs.append(X.X[msk_batch].cpu().numpy())

            # mask nulls for jet reconstruction
            msk = (ygen["cls_id"][msk_batch] != 0).numpy()
            p4s["gen"].append(ygen["p4"][msk_batch][msk].numpy())

            msk = (ycand["cls_id"][msk_batch] != 0).numpy()
            p4s["cand"].append(ycand["p4"][msk_batch][msk].numpy())

            msk = (ypred["cls_id"][msk_batch] != 0).numpy()
            p4s["pred"].append(ypred["p4"][msk_batch][msk].numpy())

        Xs = awkward.from_iter(Xs)

        for typ in ["gen", "cand"]:
            vec = vector.awk(
                awkward.zip(
                    {
                        "pt": awkward.from_iter(p4s[typ])[:, :, 0],
                        "eta": awkward.from_iter(p4s[typ])[:, :, 1],
                        "phi": awkward.from_iter(p4s[typ])[:, :, 2],
                        "e": awkward.from_iter(p4s[typ])[:, :, 3],
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
                        "pt": build_dummy_array(len(p4s["pred"]), np.float64),
                        "eta": build_dummy_array(len(p4s["pred"]), np.float64),
                        "phi": build_dummy_array(len(p4s["pred"]), np.float64),
                        "e": build_dummy_array(len(p4s["pred"]), np.float64),
                    }
                )
            )
        else:
            vec = vector.awk(
                awkward.zip(
                    {
                        "pt": awkward.from_iter(p4s["pred"])[:, :, 0],
                        "eta": awkward.from_iter(p4s["pred"])[:, :, 1],
                        "phi": awkward.from_iter(p4s["pred"])[:, :, 2],
                        "e": awkward.from_iter(p4s["pred"])[:, :, 3],
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
            f"{outpath}/preds/{sample}/pred_{rank}_{i}.parquet",
        )
        _logger.info(f"Saved predictions at {outpath}/preds/{sample}/pred_{rank}_{i}.parquet")

    _logger.info(f"Time taken to make predictions on device {rank} is: {((time.time() - ti) / 60):.2f} min")


def make_plots(outpath, sample, dataset):
    """Uses the predictions stored as .parquet files (see above) to make plots."""

    mplhep.set_style(mplhep.styles.CMS)

    os.system(f"mkdir -p {outpath}/plots/{sample}")

    plots_path = Path(f"{outpath}/plots/{sample}/")
    pred_path = Path(f"{outpath}/preds/{sample}/")

    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), -1)

    plot_num_elements(X, cp_dir=plots_path, title=format_dataset_name(sample))
    plot_sum_energy(yvals, CLASS_NAMES[dataset], cp_dir=plots_path, title=format_dataset_name(sample))

    plot_jet_ratio(yvals, cp_dir=plots_path, title=format_dataset_name(sample))

    met_data = compute_met_and_ratio(yvals)
    plot_met(met_data, cp_dir=plots_path, title=format_dataset_name(sample))
    plot_met_ratio(met_data, cp_dir=plots_path, title=format_dataset_name(sample))

    plot_particles(yvals, cp_dir=plots_path, title=format_dataset_name(sample))
