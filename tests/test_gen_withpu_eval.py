import awkward
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from mlpf.jet_utils import match_two_jet_collections  # noqa: E402
from mlpf.model.inference import gen_jet_vectors  # noqa: E402
from mlpf.plotting.plot_utils import load_eval_data, plot_jet_ratio, plot_jets  # noqa: E402

JET_PTCUT = 3.0

# two events, zero-padded [batch, jets, (pt, eta, phi, energy)]; the second event has a single jet
GENJETS_WITHPU = torch.tensor(
    [
        [[40.0, 0.5, 0.1, 50.0], [20.0, -1.0, 2.0, 35.0]],
        [[30.0, 1.5, -1.0, 80.0], [0.0, 0.0, 0.0, 0.0]],
    ]
)


def _particles(num_events):
    fields = {name: np.ones(2 * num_events, dtype=np.float32) for name in ("pt", "eta", "sin_phi", "cos_phi", "energy", "cls_id")}
    return awkward.unflatten(awkward.Array(fields), np.full(num_events, 2))


def _write_predictions(path, with_pileup_reference):
    # hard-scatter gen jets at half the pt/energy of the pileup-inclusive ones, same directions
    genjets = GENJETS_WITHPU * torch.tensor([0.5, 1.0, 1.0, 0.5])
    jets_coll = {"gen": gen_jet_vectors(genjets, JET_PTCUT)}
    for typ in ("cand", "pred", "pred_nopu", "target"):
        jets_coll[typ] = gen_jet_vectors(GENJETS_WITHPU, JET_PTCUT)
    if with_pileup_reference:
        jets_coll["gen_withpu"] = gen_jet_vectors(GENJETS_WITHPU, JET_PTCUT)

    matched_jets = {
        "target_to_cand": match_two_jet_collections(jets_coll, "target", "cand", 0.1),
        "target_to_pred": match_two_jet_collections(jets_coll, "target", "pred", 0.1),
    }
    for gen in [typ for typ in ("gen", "gen_withpu") if typ in jets_coll]:
        for typ in ("pred_nopu", "pred", "cand", "target"):
            matched_jets[f"{gen}_to_{typ}"] = match_two_jet_collections(jets_coll, gen, typ, 0.1)

    num_events = len(GENJETS_WITHPU)
    outdict = {
        "inputs": awkward.unflatten(np.ones((2 * num_events, 3), dtype=np.float32), np.full(num_events, 2)),
        "particles": {typ: _particles(num_events) for typ in ("target", "cand", "pred")},
        "jets": jets_coll,
        "matched_jets": awkward.Array(matched_jets),
        "genmet": torch.tensor([10.0, 20.0]),
    }
    if with_pileup_reference:
        outdict["genmet_withpu"] = torch.tensor([15.0, 30.0])
    awkward.to_parquet(awkward.Array(outdict), path / "pred_0_0.parquet")


def test_gen_jet_vectors_drops_padding_and_soft_jets():
    jets = gen_jet_vectors(GENJETS_WITHPU, JET_PTCUT)
    assert awkward.to_list(awkward.num(jets)) == [2, 1]
    np.testing.assert_allclose(awkward.to_numpy(awkward.flatten(jets.pt)), [40.0, 20.0, 30.0], rtol=1e-6)
    assert awkward.to_list(awkward.num(gen_jet_vectors(GENJETS_WITHPU, 25.0))) == [1, 1]


def test_eval_data_and_plots_use_the_pileup_inclusive_reference(tmp_path):
    _write_predictions(tmp_path, with_pileup_reference=True)
    yvals, _, _ = load_eval_data(str(tmp_path / "*.parquet"))

    np.testing.assert_allclose(awkward.to_numpy(yvals["gen_withpu_met"]), [15.0, 30.0])
    np.testing.assert_allclose(awkward.to_numpy(awkward.flatten(yvals["jets_gen_withpu_pt"])), [40.0, 20.0, 30.0], rtol=1e-6)
    # the predicted jets equal the pileup-inclusive gen jets and are twice the hard-scatter ones
    for typ in ("pred", "pred_nopu", "cand", "target"):
        np.testing.assert_allclose(yvals[f"jet_ratio_gen_withpu_to_{typ}_pt"], 1.0, rtol=1e-5)
        np.testing.assert_allclose(yvals[f"jet_ratio_gen_to_{typ}_pt"], 2.0, rtol=1e-5)

    plot_jets(yvals, cp_dir=tmp_path, sample="colliderml_ttbar_pu200_pf", dataset="colliderml", reference="gen_withpu", file_modifier="_withpu")
    ret = plot_jet_ratio(
        yvals, cp_dir=tmp_path, sample="colliderml_ttbar_pu200_pf", dataset="colliderml", reference="gen_withpu", file_modifier="_withpu"
    )
    for name in ("jet_pt_log", "jet_pt", "jet_eta", "jet_res"):
        assert (tmp_path / f"{name}_withpu.png").is_file()
    assert set(ret) == {f"jet_ratio_gen_withpu_to_{typ}_pt" for typ in ("target", "cand", "pred", "pred_nopu")}


def test_eval_data_without_pileup_reference_is_unchanged(tmp_path):
    _write_predictions(tmp_path, with_pileup_reference=False)
    yvals, _, _ = load_eval_data(str(tmp_path / "*.parquet"))
    assert not any("withpu" in key for key in yvals)
    np.testing.assert_allclose(yvals["jet_ratio_gen_to_pred_pt"], 2.0, rtol=1e-5)
