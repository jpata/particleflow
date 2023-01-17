import awkward
import networkx as nx
import numpy as np

# 12,14,16 are neutrinos.
neutrinos = [12, 14, 16]

# this is what I can reconstruct
labels_ys_cand = [0, 211, 130, 22, 11, 13]
labels_ys_gen = [0, 211, 130, 22, 11, 13]


def map_pdgid_to_candid(pdgid, charge):
    if pdgid in [0, 22, 11, 13]:
        return pdgid

    # charged hadron
    if abs(charge) > 0:
        return 211

    # neutral hadron
    return 130


def track_pt(omega):
    a = 3 * 10**-4
    b = 5  # B-field in tesla

    return a * np.abs(b / omega)


# this defines the track features
def track_as_array(df_tr, itr):
    row = df_tr[itr]
    return np.array(
        [
            1,  # tracks are type 1
            row["px"],
            row["py"],
            row["pz"],
            row["nhits"],
            row["d0"],
            row["z0"],
            row["dedx"],
            row["radius_innermost_hit"],
            row["tan_lambda"],
            row["nhits"],
            row["chi2"],
        ]
    )


# this defines the cluster features
def cluster_as_array(df_cl, icl):
    row = df_cl[icl]
    return np.array(
        [
            2,
            row["x"],
            row["y"],
            row["z"],
            row["nhits_ecal"],
            row["nhits_hcal"],
            row["energy"],
        ]  # clusters are type 2
    )


# this defines the genparticle features
def gen_as_array(df_gen, igen):
    if igen:
        row = df_gen[igen]
        return np.array(
            [
                abs(row["pdgid"]),
                row["charge"],
                row["px"],
                row["py"],
                row["pz"],
                row["energy"],
            ]
        )
    else:
        return np.zeros(6)


# this defines the PF particle features
def pf_as_array(df_pfs, igen):
    if igen:
        row = df_pfs[igen]
        return np.array(
            [
                abs(row["type"]),
                row["charge"],
                row["px"],
                row["py"],
                row["pz"],
                row["energy"],
            ]
        )
    else:
        return np.zeros(6)


def filter_gp(df_gen, gp):
    row = df_gen[gp]
    # status 1 is stable particle in this case
    # energy cutoff 0.2 is arbitrary and might need to be tuned
    if row["status"] == 1 and row["energy"] > 0.2:
        return True
    return False


def flatten_event(df_tr, df_cl, df_gen, df_pfs, pairs):
    Xs_tracks = []
    Xs_clusters = []
    ys_gen = []
    ys_cand = []

    # find all track-associated particles
    for itr in range(len(df_tr)):

        k = ("tr", itr)
        gp = None
        rp = None
        if k in pairs:
            gp = pairs[k][0]
            rp = pairs[k][1]

        # normalize ysgen and yscand
        ys = gen_as_array(df_gen, gp)
        cand = pf_as_array(df_pfs, rp)

        # skip the neutrinos
        if (abs(ys[0]) in neutrinos) or (abs(cand[0]) in neutrinos):
            continue
        else:
            ys[0] = labels_ys_gen.index(map_pdgid_to_candid(abs(ys[0]), ys[1]))
            cand[0] = labels_ys_cand.index(map_pdgid_to_candid(abs(cand[0]), cand[1]))

        ys_gen.append(ys)
        ys_cand.append(cand)
        Xs_tracks.append(track_as_array(df_tr, itr))

    # find all cluster-associated particles
    for icl in range(len(df_cl)):

        k = ("cl", icl)
        gp = None
        rp = None
        if k in pairs:
            gp = pairs[k][0]
            rp = pairs[k][1]

        # normalize ysgen and yscand
        ys = gen_as_array(df_gen, gp)
        cand = pf_as_array(df_pfs, rp)
        # skip the neutrinos
        if (abs(ys[0]) in neutrinos) or (abs(cand[0]) in neutrinos):
            continue
        else:
            ys[0] = labels_ys_gen.index(map_pdgid_to_candid(abs(ys[0]), ys[1]))
            cand[0] = labels_ys_cand.index(map_pdgid_to_candid(abs(cand[0]), cand[1]))

        ys_gen.append(ys)
        ys_cand.append(cand)
        Xs_clusters.append(cluster_as_array(df_cl, icl))

    Xs_clusters = np.stack(Xs_clusters, axis=-1).T  # [Nclusters, Nfeat_cluster]
    Xs_tracks = np.stack(Xs_tracks, axis=-1).T  # [Ntracks, Nfeat_track]

    # Here we pad the tracks and clusters to the same shape along the feature dimension
    if Xs_tracks.shape[1] > Xs_clusters.shape[-1]:
        Xs_clusters = np.pad(
            Xs_clusters,
            [(0, 0), (0, Xs_tracks.shape[1] - Xs_clusters.shape[-1])],
        )
    elif Xs_tracks.shape[1] < Xs_clusters.shape[-1]:
        Xs_clusters = np.pad(
            Xs_clusters,
            [(0, 0), (0, Xs_clusters.shape[-1] - Xs_tracks.shape[1])],
        )

    Xs = np.concatenate([Xs_tracks, Xs_clusters], axis=0)  # [Ntracks+Nclusters, max(Nfeat_cluster, Nfeat_track)]
    ys_gen = np.stack(ys_gen, axis=-1).T
    ys_cand = np.stack(ys_cand, axis=-1).T

    return Xs, ys_gen, ys_cand


def prepare_data_clic(fn):
    """
    Processing function that takes as input a raw parquet file and processes it.

    Returns
        a list of events, each containing three arrays [Xs, ygen, ycand].

    """

    data = awkward.from_parquet(fn)

    ret = []
    # loop over the events in the dataset
    for iev in range(len(data)):
        df_gen = data[iev]["genparticles"]

        df_cl = data[iev]["clusters"]
        df_tr = data[iev]["tracks"]
        df_pfs = data[iev]["pfs"]
        # print("Clusters={}, tracks={}, PFs={}, Gen={}".format(len(df_cl), len(df_tr), len(df_pfs), len(df_gen)))

        # skip events that don't have enough activity from training
        if len(df_pfs) < 2 or len(df_gen) < 2 or len(df_tr) < 2 or len(df_cl) < 2:
            continue

        # compute pt, px,py,pz
        df_tr["pt"] = track_pt(df_tr["omega"])
        df_tr["px"] = np.cos(df_tr["phi"]) * df_tr["pt"]
        df_tr["py"] = np.sin(df_tr["phi"]) * df_tr["pt"]
        df_tr["pz"] = df_tr["tan_lambda"] * df_tr["pt"]

        # fill track/cluster to genparticle contributions
        matrix_tr_to_gp = np.zeros((len(df_tr), len(df_gen)))
        matrix_cl_to_gp = np.zeros((len(df_cl), len(df_gen)))

        for itr in range(len(df_tr)):
            gps = df_tr[itr]["gp_contributions"]
            for gp, val in zip(gps["0"], gps["1"]):
                matrix_tr_to_gp[itr, int(gp)] += val

        for icl in range(len(df_cl)):
            gps = df_cl[icl]["gp_contributions"]
            for gp, val in zip(gps["0"], gps["1"]):
                matrix_cl_to_gp[icl, int(gp)] += val

        # fill track/cluster to PF map
        reco_to_pf = {}
        for ipf in range(len(df_pfs)):
            row = df_pfs[ipf]
            if row["track_idx"] != -1:
                k = ("tr", int(row["track_idx"]))
                assert not (k in reco_to_pf)
                reco_to_pf[k] = ipf
            elif row["cluster_idx"] != -1:
                k = ("cl", int(row["cluster_idx"]))
                assert not (k in reco_to_pf)
                reco_to_pf[k] = ipf
            else:
                # PF should always have a track or a cluster associated
                assert False

        dg = nx.Graph()
        gps = set()

        # loop over clusters, get all genparticles associated to clusters
        for icl in range(len(df_cl)):
            dg.add_node(("cl", icl))
            gp_contrib = df_cl[icl]["gp_contributions"]
            for gp, weight in zip(gp_contrib["0"], gp_contrib["1"]):
                gp = int(gp)
                if filter_gp(df_gen, gp):
                    dg.add_node(("gp", gp))
                    gps.add(gp)
                    dg.add_edge(("gp", gp), ("cl", icl), weight=weight)

        # loop over tracks, get all genparticles associated to tracks
        for itr in range(len(df_tr)):
            dg.add_node(("tr", itr))
            gp_contrib = df_tr[itr]["gp_contributions"]
            for gp, weight in zip(gp_contrib["0"], gp_contrib["1"]):
                gp = int(gp)
                if filter_gp(df_gen, gp):
                    dg.add_node(("gp", gp))
                    gps.add(gp)

                    # the track is added to the genparticle with a very high weight
                    # because we always want to associate the genparticle to a track if it's possible
                    dg.add_edge(("gp", gp), ("tr", itr), weight=9999.0)

        # uniqe genparticles
        gps = set(gps)

        # now loop over all the genparticles
        pairs = {}
        for gp in gps:
            gp_node = ("gp", gp)

            # find the neighboring reco elements (clusters and tracks)
            neighbors = list(dg.neighbors(gp_node))
            weights = [dg.edges[gp_node, n]["weight"] for n in neighbors]
            nw = zip(neighbors, weights)

            # sort the neighbors by the edge weight (deposited energy)
            nw = sorted(nw, key=lambda x: x[1], reverse=True)
            reco_obj = None
            if len(nw) > 0:
                # choose the closest neighbor as the "key" reco element
                reco_obj = nw[0][0]

                # remove the reco element from the list, so it can't be associated to anything else
                dg.remove_node(reco_obj)

            # this genparticle had a unique reco element
            if reco_obj:
                pf_obj = None
                if reco_obj and reco_obj in reco_to_pf:
                    pf_obj = reco_to_pf[reco_obj]

                assert not (reco_obj in pairs)
                pairs[reco_obj] = (gp, pf_obj)

            # this is a case where a genparticle did not have a key reco element, but instead was smeared between others
            # else:
            # print("genparticle {} is merged and cannot be reconstructed".format(gp))
            # print(df_gen.loc[gp])

        Xs, ys_gen, ys_cand = flatten_event(df_tr, df_cl, df_gen, df_pfs, pairs)

        ret.append([Xs, ys_gen, ys_cand])

    return ret
