import os
import os.path as osp
import time

import awkward
import fastjet
import mplhep
import numpy as np
import torch
import tqdm
import vector
from jet_utils import build_dummy_array, match_two_jet_collections
from logger import _logger
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

jetdef = fastjet.JetDefinition(fastjet.ee_genkt_algorithm, 0.7, -1.0)
jet_pt = 5.0
jet_match_dr = 0.1


def particle_array_to_awkward(batch_ids, arr_id, arr_p4):
    ret = {
        "cls_id": arr_id,
        "pt": arr_p4[:, 1],
        "eta": arr_p4[:, 2],
        "sin_phi": arr_p4[:, 3],
        "cos_phi": arr_p4[:, 4],
        "energy": arr_p4[:, 5],
    }
    ret["phi"] = np.arctan2(ret["sin_phi"], ret["cos_phi"])
    ret = awkward.from_iter([{k: ret[k][batch_ids == b] for k in ret.keys()} for b in np.unique(batch_ids)])
    return ret


def make_predictions(rank, mlpf, loader, model_prefix, sample):
    if not osp.isdir(f"{model_prefix}/preds/{sample}"):
        os.makedirs(f"{model_prefix}/preds/{sample}")

    ti = time.time()

    for i, batch in tqdm.tqdm(enumerate(loader)):
        event = batch.to(rank)

        # recall target ~ ["PDG", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "jet_idx"]
        target_ids = event.ygen[:, 0].long()
        event.ygen = event.ygen[:, 1:]

        cand_ids = event.ycand[:, 0].long()
        event.ycand = event.ycand[:, 1:]

        # make mlpf forward pass
        pred_ids_one_hot, pred_momentum, pred_charge = mlpf(event.X, event.batch)

        pred_ids = torch.argmax(pred_ids_one_hot.detach(), axis=-1)
        pred_charge = torch.argmax(pred_charge.detach(), axis=1, keepdim=True) - 1
        pred_p4 = torch.cat([pred_charge, pred_momentum.detach()], axis=-1)

        batch_ids = event.batch.cpu().numpy()
        awkvals = {
            "gen": particle_array_to_awkward(batch_ids, target_ids.cpu().numpy(), event.ygen.cpu().numpy()),
            "cand": particle_array_to_awkward(batch_ids, cand_ids.cpu().numpy(), event.ycand.cpu().numpy()),
            "pred": particle_array_to_awkward(batch_ids, pred_ids.cpu().numpy(), pred_p4.cpu().numpy()),
        }

        gen_p4, cand_p4, pred_p4 = [], [], []
        gen_cls, cand_cls, pred_cls = [], [], []
        Xs = []
        for _ibatch in np.unique(event.batch.cpu().numpy()):
            msk_batch = event.batch == _ibatch
            msk_gen = target_ids[msk_batch] != 0
            msk_cand = cand_ids[msk_batch] != 0
            msk_pred = pred_ids[msk_batch] != 0

            Xs.append(event.X[msk_batch].cpu().numpy())

            gen_p4.append(event.ygen[msk_batch, 1:][msk_gen])
            gen_cls.append(target_ids[msk_batch][msk_gen])

            cand_p4.append(event.ycand[msk_batch, 1:][msk_cand])
            cand_cls.append(cand_ids[msk_batch][msk_cand])

            pred_p4.append(pred_momentum[msk_batch, :][msk_pred])
            pred_cls.append(pred_ids[msk_batch][msk_pred])

        Xs = awkward.from_iter(Xs)
        gen_p4 = awkward.from_iter(gen_p4)
        gen_cls = awkward.from_iter(gen_cls)
        gen_p4 = vector.awk(
            awkward.zip({"pt": gen_p4[:, :, 0], "eta": gen_p4[:, :, 1], "phi": gen_p4[:, :, 2], "e": gen_p4[:, :, 3]})
        )

        cand_p4 = awkward.from_iter(cand_p4)
        cand_cls = awkward.from_iter(cand_cls)
        cand_p4 = vector.awk(
            awkward.zip({"pt": cand_p4[:, :, 0], "eta": cand_p4[:, :, 1], "phi": cand_p4[:, :, 2], "e": cand_p4[:, :, 3]})
        )

        # in case of no predicted particles in the batch
        if torch.sum(pred_ids != 0) == 0:
            pt = build_dummy_array(len(pred_p4), np.float64)
            eta = build_dummy_array(len(pred_p4), np.float64)
            phi = build_dummy_array(len(pred_p4), np.float64)
            pred_cls = build_dummy_array(len(pred_p4), np.float64)
            energy = build_dummy_array(len(pred_p4), np.float64)
            pred_p4 = vector.awk(awkward.zip({"pt": pt, "eta": eta, "phi": phi, "e": energy}))
        else:
            pred_p4 = awkward.from_iter(pred_p4)
            pred_cls = awkward.from_iter(pred_cls)
            pred_p4 = vector.awk(
                awkward.zip(
                    {
                        "pt": pred_p4[:, :, 0],
                        "eta": pred_p4[:, :, 1],
                        "phi": pred_p4[:, :, 2],
                        "e": pred_p4[:, :, 3],
                    }
                )
            )

        jets_coll = {}

        cluster1 = fastjet.ClusterSequence(awkward.Array(gen_p4.to_xyzt()), jetdef)
        jets_coll["gen"] = cluster1.inclusive_jets(min_pt=jet_pt)
        cluster2 = fastjet.ClusterSequence(awkward.Array(cand_p4.to_xyzt()), jetdef)
        jets_coll["cand"] = cluster2.inclusive_jets(min_pt=jet_pt)
        cluster3 = fastjet.ClusterSequence(awkward.Array(pred_p4.to_xyzt()), jetdef)
        jets_coll["pred"] = cluster3.inclusive_jets(min_pt=jet_pt)

        gen_to_pred = match_two_jet_collections(jets_coll, "gen", "pred", jet_match_dr)
        gen_to_cand = match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr)
        matched_jets = awkward.Array({"gen_to_pred": gen_to_pred, "gen_to_cand": gen_to_cand})

        awkward.to_parquet(
            awkward.Array(
                {
                    "inputs": Xs,
                    "particles": awkvals,
                    "jets": jets_coll,
                    "matched_jets": matched_jets,
                }
            ),
            f"{model_prefix}/preds/{sample}/pred_{i}.parquet",
        )
        _logger.info(f"Saved predictions at {model_prefix}/preds/{sample}/pred_{i}.parquet")

        if i == 2:
            break

    _logger.info(f"Time taken to make predictions on rank {rank} is: {((time.time() - ti) / 60):.2f} min")


def make_plots(model_prefix, sample):
    mplhep.set_style(mplhep.styles.CMS)

    # Use the dataset names from the common nomenclature
    _title = format_dataset_name(sample)

    plots_path = f"{model_prefix}/plots/"

    if not os.path.isdir(plots_path):
        os.makedirs(plots_path)

    yvals, X, _ = load_eval_data(f"{model_prefix}/preds/sample/" + "*.parquet", -1)

    plot_num_elements(X, cp_dir=plots_path, title=_title)
    plot_sum_energy(yvals, cp_dir=plots_path, title=_title)

    plot_jet_ratio(yvals, cp_dir=plots_path, title=_title)

    met_data = compute_met_and_ratio(yvals)
    plot_met(met_data, cp_dir=plots_path, title=_title)
    plot_met_ratio(met_data, cp_dir=plots_path, title=_title)

    plot_particles(yvals, cp_dir=plots_path, title=_title)
