"""Data adapter: CMS Phase-2 Run3-style pkls -> TFDS tensors.

Reads the run3style postprocessing output (Xelem, ytarget, ycand, genjet, targetjet,
pythia) and produces X / ytarget / ycand in the canonical schema the MLPF training
expects (see mlpf/conf.py: X_FEATURES['cms_phase2'], Y_FEATURES, CLASS_LABELS).
"""
import pickle
import datetime
import numpy as np

from mlpf.conf import Dataset, X_FEATURES as _XF, Y_FEATURES, CLASS_LABELS

DS = Dataset.CMS_PHASE2.value
X_FEATURES = _XF[DS]
CLS = CLASS_LABELS[DS]                 # [0, 211, 130, 22, 11, 13]
TYP_MAP = {1: 1, 2: 1, 3: 2, 4: 2}     # nano_v2 Xelem typ -> typ_idx: track=1, gsf=1, trackster=2, EM-supercl=2
#                                        (cms_phase2 has 2 nonzero element types [1,2]; the old {1:1,2:3,4:2}
#                                        was for the run3style/v3 codes and sent nano_v2 tracksters (typ 3) to padding)
NUM_SPLITS = 10


def _cls_idx_target(pid):
    p = abs(int(pid))
    return CLS.index(p) if p in CLS else 0


def _cls_idx_cand(pid, charge):
    p = abs(int(pid))
    if p in (22, 11, 13):
        return CLS.index(p)
    return CLS.index(211) if abs(charge) > 0 else CLS.index(130)  # charged vs neutral hadron


def prepare_data_phase2(fn):
    Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias = [], [], [], [], [], [], []
    try:
        data = pickle.load(open(fn, "rb"))
    except Exception as e:
        print("Could not open {}: {}".format(fn, e))
        return Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias

    for ev in data:
        Xe, yt, yc = ev["Xelem"], ev["ytarget"], ev["ycand"]
        n = len(Xe)

        # ---- X ----
        phi = Xe["phi"]
        # Map Xelem -> X_FEATURES['cms_phase2']. The v2 postprocessing
        # (postprocessing_ticl_nopu_nano_v2.py) prefixes several fields (ts_*, track_*) vs this
        # builder's original names, and omits the trackster eigenvalues ev1/ev2/ev3 (it carries
        # shower_depth / n_clusters instead). g() tolerates absent fields by zero-filling.
        _names = set(Xe.dtype.names)
        def g(k):
            return Xe[k] if k in _names else np.zeros(n, np.float32)
        col = {
            "typ_idx": np.array([TYP_MAP.get(int(t), 0) for t in Xe["typ"]], np.float32),
            "pt": Xe["pt"], "eta": Xe["eta"], "sin_phi": np.sin(phi), "cos_phi": np.cos(phi),
            "e": Xe["energy"], "charge": Xe["charge"], "px": Xe["px"], "py": Xe["py"],
            "pz": Xe["pz"], "em_energy": Xe["em_energy"], "nhits": Xe["nhits"],
            "bary_z": Xe["bary_z"],
            "time": g("ts_time"), "timeerror": g("ts_time_err"),         # v2 rename ts_time(_err)
            "ev1": g("ev1"), "ev2": g("ev2"), "ev3": g("ev3"),           # not in v2 -> zero (TODO: shower_depth/n_clusters)
            "muon_type": g("track_muon_type"), "muon_dt_hits": g("track_muon_dt_hits"), "muon_csc_hits": g("track_muon_csc_hits"),
            "gsf_type": g("track_gsf_type"),
            "pterror": g("track_pt_err"), "etaerror": g("track_eta_err"), "phierror": g("track_phi_err"),
            "lambdaerror": g("track_lambda_err"), "qoverperror": g("track_qoverp_err"),
            "vx": g("track_vx"), "vy": g("track_vy"), "vz": g("track_vz"),
            "min_dR_track": Xe["min_dR_track"], "near_track_pt": Xe["near_track_pt"], "sum_pt_dR10": Xe["sum_pt_dR10"],
            "n_trk_dR01": Xe["n_trk_dR01"], "n_trk_dR02": Xe["n_trk_dR02"], "n_trk_dR03": Xe["n_trk_dR03"],
            "n_trk_dR04": Xe["n_trk_dR04"], "n_trk_dR05": Xe["n_trk_dR05"],
        }
        X = np.stack([col[k] for k in X_FEATURES], axis=-1).astype(np.float32)

        # ---- ytarget (canonical 14-col Y_FEATURES; col0 = class index) ----
        yg = np.zeros((n, len(Y_FEATURES)), np.float32)
        yg[:, 0] = [_cls_idx_target(p) for p in yt["pid"]]
        yg[:, 1] = yt["charge"]; yg[:, 2] = yt["pt"]; yg[:, 3] = yt["eta"]
        yg[:, 4] = yt["sin_phi"]; yg[:, 5] = yt["cos_phi"]; yg[:, 6] = yt["energy"]
        yg[:, 7] = yt["ispu"]; yg[:, 12] = yt["jet_idx"]
        # cols 8-11 (genStatus/simStatus/gp_to_track/gp_to_cluster) and 13 (particle_number) left 0

        # ---- ycand: must be per-element aligned with X (padded/sorted alongside it).
        # TICLCandidates are a separate reco list (not per-element), so we store zeros here;
        # the real TICL jet baseline lives in the pkl's `candjet` (used outside training).
        ycf = np.zeros((n, len(Y_FEATURES)), np.float32)

        # ---- event-level ----
        pyth = np.asarray(ev["pythia"], np.float32).reshape(-1, 5)
        if len(pyth):
            gm = float(np.hypot(np.sum(pyth[:, 1]*np.cos(pyth[:, 3])), np.sum(pyth[:, 1]*np.sin(pyth[:, 3]))))
        else:
            gm = 0.0

        Xs.append(X); ytargets.append(yg); ycands.append(ycf); genmets.append(np.float32(gm))
        genjets.append(np.asarray(ev["genjet"], np.float32).reshape(-1, 4))
        targetjets.append(np.asarray(ev["targetjet"], np.float32).reshape(-1, 4))
        ypythias.append(pyth)
    return Xs, ytargets, ycands, genmets, genjets, targetjets, ypythias


def split_list(lst, x):
    s = len(lst) // x
    r = [lst[i*s:(i+1)*s] for i in range(x - 1)]
    r.append(lst[(x-1)*s:])
    return r


def split_sample(path, builder_config, num_splits=NUM_SPLITS, train_frac=0.9):
    files = sorted(list(path.glob("*.pkl")))
    print("Found {} files in {}".format(len(files), path))
    assert len(files) > 0
    idx = int(train_frac * len(files))
    ftr, fte = files[:idx], files[idx:]
    si = int(builder_config.name) - 1
    return {"train": generate_examples(split_list(ftr, num_splits)[si]),
            "test": generate_examples(split_list(fte, num_splits)[si])}


def generate_examples(files):
    for fi in files:
        print(datetime.datetime.now(), "reading", fi)
        try:
            Xs, ytg, yc, gm, gj, tj, yp = prepare_data_phase2(str(fi))
        except Exception as e:
            # one malformed/stale-schema file must not kill an hours-long tfds build
            print("Skipping {}: {}".format(fi, e))
            continue
        for ii in range(len(Xs)):
            yield str(fi) + "_" + str(ii), {
                "X": Xs[ii], "ytarget": ytg[ii], "ycand": yc[ii],
                "genmet": gm[ii], "genjets": gj[ii], "targetjets": tj[ii], "pythia": yp[ii],
            }
