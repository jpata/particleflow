import os
import pickle as pkl
from pathlib import Path

import awkward
import fastjet
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.metrics
import torch
import torch_geometric
import tqdm
import vector
from jet_utils import build_dummy_array, match_two_jet_collections
from plotting.plot_utils import load_eval_data, plot_jet_ratio

from .utils import CLASS_NAMES_CLIC_LATEX, NUM_CLASSES, combine_PFelements, distinguish_PFelements

matplotlib.use("Agg")

# Ignore divide by 0 errors
np.seterr(divide="ignore", invalid="ignore")

CLASS_TO_ID = {
    "charged_hadron": 1,
    "neutral_hadron": 2,
    "photon": 3,
    "electron": 4,
    "muon": 5,
}


def particle_array_to_awkward(batch_ids, arr_id, arr_p4):
    ret = {
        "cls_id": arr_id,
        "pt": arr_p4[:, 1],
        "eta": arr_p4[:, 2],
        "phi": arr_p4[:, 3],
        "energy": arr_p4[:, 4],
    }
    ret["sin_phi"] = np.sin(ret["phi"])
    ret["cos_phi"] = np.cos(ret["phi"])
    ret = awkward.from_iter([{k: ret[k][batch_ids == b] for k in ret.keys()} for b in np.unique(batch_ids)])
    return ret


def evaluate(device, encoder, decoder, mlpf, batch_size_mlpf, mode, outpath, data_, save_as_):

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    jet_pt = 5.0
    jet_match_dr = 0.1

    npred_, ngen_, ncand_, = (
        {},
        {},
        {},
    )

    mlpf.eval()
    encoder.eval()
    decoder.eval()
    for j, data in enumerate(data_):
        print(f"Testing the {mode} model on the {save_as_[j]}")

        this_out_path = "{}/{}/{}".format(outpath, mode, save_as_[j])
        os.makedirs(this_out_path)
        test_loader = torch_geometric.loader.DataLoader(data, batch_size_mlpf)

        npred, ngen, ncand = {}, {}, {}
        for class_ in CLASS_TO_ID.keys():
            npred[class_], ngen[class_], ncand[class_] = [], [], []

        mlpf.eval()
        encoder.eval()
        decoder.eval()

        conf_matrix = np.zeros((6, 6))
        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
                print(f"making predictions: {i+1}/{len(test_loader)}")

                if mode == "ssl":
                    # make transformation
                    tracks, clusters = distinguish_PFelements(batch.to(device))

                    # ENCODE
                    embedding_tracks, embedding_clusters = encoder(tracks, clusters)

                    # use the learnt representation as your input as well as the global feature vector
                    tracks.x = embedding_tracks
                    clusters.x = embedding_clusters

                    event = combine_PFelements(tracks, clusters)

                elif mode == "native":
                    event = batch

                # make mlpf forward pass
                pred_ids_one_hot, pred_momentum, pred_charge = mlpf(event.to(device))

                pred_ids = torch.argmax(pred_ids_one_hot, axis=1)
                target_ids = event.ygen_id
                cand_ids = event.ycand_id

                batch_ids = event.batch.cpu().numpy()
                awkvals = {
                    "gen": particle_array_to_awkward(batch_ids, target_ids.cpu().numpy(), event.ygen.cpu().numpy()),
                    "cand": particle_array_to_awkward(batch_ids, cand_ids.cpu().numpy(), event.ycand.cpu().numpy()),
                    "pred": particle_array_to_awkward(
                        batch_ids, pred_ids.cpu().numpy(), torch.cat([pred_charge, pred_momentum], axis=-1).cpu().numpy()
                    ),
                }

                gen_p4 = []
                gen_cls = []
                cand_p4 = []
                cand_cls = []
                pred_p4 = []
                pred_cls = []
                Xs = []
                for ibatch in np.unique(event.batch.cpu().numpy()):
                    msk_batch = event.batch == ibatch
                    msk_gen = target_ids[msk_batch] != 0
                    msk_cand = cand_ids[msk_batch] != 0
                    msk_pred = pred_ids[msk_batch] != 0

                    Xs.append(event.x[msk_batch].cpu().numpy())

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
                    awkward.zip(
                        {"pt": gen_p4[:, :, 0], "eta": gen_p4[:, :, 1], "phi": gen_p4[:, :, 2], "e": gen_p4[:, :, 3]}
                    )
                )

                cand_p4 = awkward.from_iter(cand_p4)
                cand_cls = awkward.from_iter(cand_cls)
                cand_p4 = vector.awk(
                    awkward.zip(
                        {"pt": cand_p4[:, :, 0], "eta": cand_p4[:, :, 1], "phi": cand_p4[:, :, 2], "e": cand_p4[:, :, 3]}
                    )
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
                            {"pt": pred_p4[:, :, 0], "eta": pred_p4[:, :, 1], "phi": pred_p4[:, :, 2], "e": pred_p4[:, :, 3]}
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

                conf_matrix += sklearn.metrics.confusion_matrix(
                    target_ids.detach().cpu(),
                    pred_ids.detach().cpu(),
                    labels=range(NUM_CLASSES),
                )

                awkward.to_parquet(
                    awkward.Array(
                        {
                            "inputs": Xs,
                            "particles": awkvals,
                            "jets": jets_coll,
                            "matched_jets": matched_jets,
                        }
                    ),
                    "{}/pred_{}.parquet".format(this_out_path, i),
                )

                for batch_index in range(batch_size_mlpf):
                    # unpack the batch
                    pred = pred_ids[event.batch == batch_index]
                    target = target_ids[event.batch == batch_index]
                    cand = cand_ids[event.batch == batch_index]

                    for class_, id_ in CLASS_TO_ID.items():
                        npred[class_].append((pred == id_).sum().item())
                        ngen[class_].append((target == id_).sum().item())
                        ncand[class_].append((cand == id_).sum().item())

            make_conf_matrix(conf_matrix, outpath, mode, save_as_[j])
            npred_[save_as_[j]], ngen_[save_as_[j]], ncand_[save_as_[j]] = make_multiplicity_plots(
                npred, ngen, ncand, outpath, mode, save_as_[j]
            )
            yvals, _, _ = load_eval_data("{}/pred_*.parquet".format(this_out_path))
            plot_jet_ratio(yvals, cp_dir=Path(this_out_path), title=save_as_[j])

    return npred_, ngen_, ncand_


def make_conf_matrix(cm, outpath, mode, save_as):
    import itertools

    cmap = plt.get_cmap("Blues")
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm[np.isnan(cm)] = 0.0

    plt.figure(figsize=(8, 6))
    plt.axes()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:0.2f}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=15,
        )
    if mode == "ssl":
        plt.title(f"{mode} based MLPF", fontsize=25)
    else:
        plt.title(f"{mode} MLPF", fontsize=25)
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)

    plt.xticks(
        range(len(CLASS_NAMES_CLIC_LATEX)),
        CLASS_NAMES_CLIC_LATEX,
        rotation=45,
        fontsize=15,
    )
    plt.yticks(range(len(CLASS_NAMES_CLIC_LATEX)), CLASS_NAMES_CLIC_LATEX, fontsize=15)

    plt.tight_layout()

    plt.savefig(f"{outpath}/conf_matrix_{mode}_{save_as}.pdf")
    with open(f"{outpath}/conf_matrix_{mode}_{save_as}.pkl", "wb") as f:
        pkl.dump(cm, f)
    plt.close()


def make_multiplicity_plots(npred, ngen, ncand, outpath, mode, save_as):
    for class_ in ["charged_hadron", "neutral_hadron", "photon"]:
        # Plot the particle multiplicities
        plt.figure()
        plt.axes()
        plt.scatter(ngen[class_], ncand[class_], marker=".", alpha=0.4, label="PF")
        plt.scatter(ngen[class_], npred[class_], marker=".", alpha=0.4, label="MLPF")
        a = 0.5 * min(np.min(npred[class_]), np.min(ngen[class_]))
        b = 1.5 * max(np.max(npred[class_]), np.max(ngen[class_]))
        # plt.xlim(a, b)
        # plt.ylim(a, b)
        plt.plot([a, b], [a, b], color="black", ls="--")
        plt.title(class_)
        plt.xlabel("number of truth particles")
        plt.ylabel("number of reconstructed particles")
        plt.legend(loc=4)
        plt.savefig(f"{outpath}/multiplicity_plots_{CLASS_TO_ID[class_]}_{mode}_{save_as}.pdf")
        plt.close()

    return npred, ngen, ncand


def make_multiplicity_plots_both(ret_ssl, ret_native, outpath):

    npred_ssl, ngen_ssl, _ = ret_ssl
    npred_native, ngen_native, _ = ret_native

    for data_ in npred_ssl.keys():
        for class_ in ["charged_hadron", "neutral_hadron", "photon"]:
            # Plot the particle multiplicities
            plt.figure()
            plt.axes()
            plt.scatter(ngen_ssl[data_][class_], npred_ssl[data_][class_], marker=".", alpha=0.4, label="ssl-based MLPF")
            plt.scatter(ngen_native[data_][class_], npred_native[data_][class_], marker=".", alpha=0.4, label="native MLPF")
            a = 0.5 * min(np.min(npred_ssl[data_][class_]), np.min(ngen_ssl[data_][class_]))
            b = 1.5 * max(np.max(npred_ssl[data_][class_]), np.max(ngen_ssl[data_][class_]))
            # plt.xlim(a, b)
            # plt.ylim(a, b)
            plt.plot([a, b], [a, b], color="black", ls="--")
            plt.title(class_)
            plt.xlabel("number of truth particles")
            plt.ylabel("number of reconstructed particles")
            plt.legend(title=data_, loc=4)
            plt.savefig(f"{outpath}/multiplicity_plots_{CLASS_TO_ID[class_]}_{data_}.pdf")
            plt.close()
