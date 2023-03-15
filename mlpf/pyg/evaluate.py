import time

import numpy as np
import torch
import torch_geometric
from pyg.ssl.utils import combine_PFelements, distinguish_PFelements
import awkward
from jet_utils import build_dummy_array, match_two_jet_collections
import fastjet
import vector
import tqdm


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


def make_predictions_awk(rank, dataset, mlpf, file_loader, batch_size, PATH, ssl_encoder=None):

    ti = time.time()

    tf_0, tf_f = time.time(), 0
    for num, this_loader in enumerate(file_loader):
        if "utils" in str(type(file_loader)):  # it must be converted to a pyg DataLoader if it's not (only needed for CMS)
            print(f"Time to load file {num+1}/{len(file_loader)} on rank {rank} is {round(time.time() - tf_0, 3)}s")
            tf_f = tf_f + (time.time() - tf_0)
            this_loader = torch_geometric.loader.DataLoader([x for t in this_loader for x in t], batch_size=batch_size)

        tf = 0
        for i, batch in tqdm.tqdm(enumerate(this_loader), total=len(this_loader)):

            if ssl_encoder is not None:
                # seperate PF-elements
                tracks, clusters = distinguish_PFelements(batch.to(rank))
                # ENCODE
                embedding_tracks, embedding_clusters = ssl_encoder(tracks, clusters)
                # concat the inputs with embeddings
                tracks.x = torch.cat([batch.x[batch.x[:, 0] == 1], embedding_tracks], axis=1)
                clusters.x = torch.cat([batch.x[batch.x[:, 0] == 2], embedding_clusters], axis=1)
                # combine PF-elements
                event = combine_PFelements(tracks, clusters).to(rank)

            else:
                event = batch.to(rank)

            t0 = time.time()
            pred_ids_one_hot, pred_momentum, pred_charge = mlpf(event)
            tf = tf + (time.time() - t0)

            pred_ids = torch.argmax(pred_ids_one_hot.detach(), axis=-1)
            pred_charge = torch.argmax(pred_charge.detach(), axis=1, keepdim=True) - 1
            pred_p4 = torch.cat([pred_charge, pred_momentum.detach()], axis=-1)

            target_ids = event.ygen_id
            cand_ids = event.ycand_id

            batch_ids = event.batch.cpu().numpy()
            awkvals = {
                "gen": particle_array_to_awkward(batch_ids, target_ids.cpu().numpy(), event.ygen.cpu().numpy()),
                "cand": particle_array_to_awkward(batch_ids, cand_ids.cpu().numpy(), event.ycand.cpu().numpy()),
                "pred": particle_array_to_awkward(batch_ids, pred_ids.cpu().numpy(), pred_p4.cpu().numpy()),
            }

            gen_p4 = []
            gen_cls = []
            cand_p4 = []
            cand_cls = []
            pred_p4 = []
            pred_cls = []
            Xs = []
            for _ibatch in np.unique(event.batch.cpu().numpy()):
                msk_batch = event.batch == _ibatch
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
                awkward.zip({"pt": gen_p4[:, :, 0], "eta": gen_p4[:, :, 1], "phi": gen_p4[:, :, 2], "e": gen_p4[:, :, 3]})
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
                f"{PATH}/pred_{i}.parquet",
            )

        print(f"Average inference time per batch on rank {rank} is {(tf / len(this_loader)):.3f}s")
        t0 = time.time()
        print(f"Time taken to make predictions on rank {rank} is: {((time.time() - ti) / 60):.2f} min")
