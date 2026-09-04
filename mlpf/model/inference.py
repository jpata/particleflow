import os
import time
from pathlib import Path
import sys

import awkward
import fastjet
import mplhep
import torch
import tqdm
import vector
import numpy as np
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
from mlpf.model.utils import unpack_target
from mlpf.conf import OutputMode


def predict_one_batch(conv_type, model, i, batch, rank, jetdef, jet_ptcut, jet_match_dr, outpath, dir_name, sample):

    # skip prediction if output exists
    outfile = f"{outpath}/preds{dir_name}/{sample}/pred_{rank}_{i}.parquet"
    if os.path.isfile(outfile):
        return

    # run model on batch
    batch = batch.to(rank)

    if hasattr(model, "module"):
        model_module = model.module
    else:
        model_module = model

    ypred = model_module.predict_particles(batch.X, batch.mask)

    if model_module.output_mode == OutputMode.SET:
        ytarget = unpack_target(batch.ytarget_set.to(torch.float32), model_module)
        ytarget["pt"] = torch.exp(ytarget["pt"])
        ytarget["energy"] = torch.exp(ytarget["energy"])
        ytarget["momentum"] = torch.stack([ytarget["pt"], ytarget["eta"], ytarget["sin_phi"], ytarget["cos_phi"], ytarget["energy"]], dim=-1)
        ytarget["p4"] = torch.stack([ytarget["pt"], ytarget["eta"], ytarget["phi"], ytarget["energy"]], dim=-1)
    else:
        batch.ytarget[..., 2] = batch.ytarget_pt_orig
        batch.ytarget[..., 6] = batch.ytarget_e_orig
        ytarget = unpack_target(batch.ytarget.to(torch.float32), model_module)
    ycand = unpack_target(batch.ycand.to(torch.float32), model_module)

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
    genjets = vector.awk(awkward.zip({"px": genjets.px, "py": genjets.py, "pz": genjets.pz, "E": genjets.E}))

    jets_coll = {}
    jets_coll["gen"] = genjets

    # Flatten each independently padded collection with its own mask.
    X = batch.X[batch.mask].cpu().float().contiguous().numpy()
    input_counts = torch.sum(batch.mask, axis=1).cpu().numpy()
    if model_module.output_mode == OutputMode.SET:
        target_mask = batch.target_mask.bool()
        prediction_mask = ypred["cls_id"] != 0
    else:
        target_mask = batch.mask.bool()
        prediction_mask = batch.mask.bool()

    collection_masks = {"target": target_mask, "cand": batch.mask.bool(), "pred": prediction_mask}
    awkvals = {}
    for flat_arr, typ in [(ytarget, "target"), (ycand, "cand"), (ypred, "pred")]:
        collection_mask = collection_masks[typ]
        counts = collection_mask.sum(dim=1).cpu().numpy()
        values = {key: value[collection_mask].detach().cpu().float().contiguous().numpy() for key, value in flat_arr.items()}
        awk_arr = awkward.Array(values)
        awkvals[typ] = awkward.unflatten(awk_arr, counts)
    Xs = awkward.unflatten(awkward.from_numpy(X), input_counts)

    # now cluster jets
    for typ, ydata in zip(
        ["cand", "target", "pred", "pred_nopu"],
        [awkvals["cand"], awkvals["target"], awkvals["pred"], awkvals["pred"]],
    ):
        msk = ydata["cls_id"] != 0
        # placeholder cut on the PU frac prediction
        if typ == "pred_nopu":
            msk1 = ydata["ispu"] < 0.8
            msk = msk & msk1

        pt = ydata["pt"][msk]
        eta = ydata["eta"][msk]
        phi = ydata["phi"][msk]
        energy = ydata["energy"][msk]

        vec = awkward.zip(
            {
                "px": pt * np.cos(phi),
                "py": pt * np.sin(phi),
                "pz": pt * np.sinh(eta),
                "E": energy,
            }
        )

        cluster = fastjet.ClusterSequence(vec, jetdef)
        jets = cluster.inclusive_jets(min_pt=jet_ptcut)
        jets_coll[typ] = vector.awk(awkward.zip({"px": jets.px, "py": jets.py, "pz": jets.pz, "E": jets.E}))

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


def make_plots(outpath, sample, dataset, dir_name="", num_test_events=None, baseline_yvals=None):
    """Uses the predictions stored as .parquet files from run_predictions to make plots."""
    ds_name = dataset.value

    ret_dict = {}
    mplhep.style.use(mplhep.styles.CMS)
    # class_names = get_class_names(sample)
    os.system(f"mkdir -p {outpath}/plots{dir_name}/{sample}")

    plots_path = Path(f"{outpath}/plots{dir_name}/{sample}/")
    pred_path = Path(f"{outpath}/preds{dir_name}/{sample}/")

    _logger.info(f"Loading data for plotting from {pred_path}")
    yvals, X, _ = load_eval_data(str(pred_path / "*.parquet"), num_test_events)
    _logger.info(f"Loaded data for plotting from {pred_path}")

    plot_num_elements(X, cp_dir=plots_path)
    _logger.info("Plotted number of elements")

    # plot_elements(X, yvals, cp_dir=plots_path, dataset=ds_name, sample=sample)

    plot_jets(
        yvals,
        cp_dir=plots_path,
        sample=sample,
        dataset=ds_name,
        baseline_yvals=baseline_yvals,
    )
    _logger.info("Plotted jets")

    ret_dict["jet_ratio"] = plot_jet_ratio(
        yvals,
        cp_dir=plots_path,
        sample=sample,
        dataset=ds_name,
        baseline_yvals=baseline_yvals,
    )
    _logger.info("Plotted jet ratio")

    return ret_dict


def load_pf_baseline(data_dir, sample, dataset, version="3.2.0", splits=("1",), num_events=None):
    """Load conventional PF candidates and compute their jet-level plotting values."""
    from mlpf.jet_utils import get_jet_config
    from mlpf.model.PFDataset import PFDataset

    jetdef, jet_ptcut, jet_match_dr = get_jet_config(dataset)
    samples_per_split = None if num_events is None else max(1, num_events // len(splits))
    datasets = [
        PFDataset(
            data_dir,
            f"{sample}/{split}:{version}",
            "test",
            num_samples=samples_per_split,
            pad_to_multiple=None,
        ).ds
        for split in splits
    ]
    source = torch.utils.data.ConcatDataset(datasets)
    nevents = len(source) if num_events is None else min(num_events, len(source))

    cand_components = {name: [] for name in ["px", "py", "pz", "E"]}
    target_components = {name: [] for name in ["px", "py", "pz", "E"]}
    gen_components = {name: [] for name in ["px", "py", "pz", "E"]}

    def append_particles(values, components, pt_override=None, energy_override=None):
        mask = values[:, 0] != 0
        pt = values[:, 2] if pt_override is None else pt_override
        energy = values[:, 6] if energy_override is None else energy_override
        eta = values[:, 3]
        phi = np.arctan2(values[:, 4], values[:, 5])
        components["px"].append((pt * np.cos(phi))[mask])
        components["py"].append((pt * np.sin(phi))[mask])
        components["pz"].append((pt * np.sinh(eta))[mask])
        components["E"].append(energy[mask])

    for index in range(nevents):
        event = source[index]
        append_particles(event["ycand"], cand_components)
        append_particles(
            event["ytarget"],
            target_components,
            pt_override=event["ytarget_pt_orig"],
            energy_override=event["ytarget_e_orig"],
        )

        genjets = event["genjets"]
        mask = genjets[:, 0] > jet_ptcut
        pt, eta, phi, energy = (genjets[:, i] for i in range(4))
        gen_components["px"].append((pt * np.cos(phi))[mask])
        gen_components["py"].append((pt * np.sin(phi))[mask])
        gen_components["pz"].append((pt * np.sinh(eta))[mask])
        gen_components["E"].append(energy[mask])

    def vectors(components):
        return awkward.zip({name: awkward.Array(values) for name, values in components.items()})

    jets_coll = {"gen": vector.awk(vectors(gen_components))}
    for name, components in [("cand", cand_components), ("target", target_components)]:
        clustered = fastjet.ClusterSequence(vectors(components), jetdef).inclusive_jets(min_pt=jet_ptcut)
        jets_coll[name] = vector.awk(awkward.zip({"px": clustered.px, "py": clustered.py, "pz": clustered.pz, "E": clustered.E}))

    matched = {
        "gen_to_cand": match_two_jet_collections(jets_coll, "gen", "cand", jet_match_dr),
        "target_to_cand": match_two_jet_collections(jets_coll, "target", "cand", jet_match_dr),
    }
    yvals = {}
    for name in ["gen", "target", "cand"]:
        for value in ["pt", "eta"]:
            yvals[f"jets_{name}_{value}"] = getattr(jets_coll[name], value)

    for left, right in [("gen", "cand"), ("target", "cand")]:
        indices = matched[f"{left}_to_{right}"]
        left_pt = awkward.flatten(jets_coll[left].pt[indices[left]], axis=1)
        right_pt = awkward.flatten(jets_coll[right].pt[indices[right]], axis=1)
        yvals[f"jet_ratio_{left}_to_{right}_pt"] = awkward.to_numpy(right_pt / left_pt)

    return yvals
