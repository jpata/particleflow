import pytest
import torch

from mlpf.conf import OutputMode
from mlpf.model.PFDataset import PFBatch
from mlpf.model.validation_metrics import (
    MATCH_COST_CAP,
    MATCH_DR,
    MATCH_LOG_PT,
    MATCH_REL_PT,
    _matching_cost,
    compute_validation_particle_metrics,
    validation_particle_collections,
)


def make_collection(cls_id, pt, eta, phi, energy):
    phi = torch.tensor([phi], dtype=torch.float32)
    return {
        "cls_id": torch.tensor([cls_id], dtype=torch.long),
        "pt": torch.tensor([pt], dtype=torch.float32),
        "eta": torch.tensor([eta], dtype=torch.float32),
        "sin_phi": torch.sin(phi),
        "cos_phi": torch.cos(phi),
        "energy": torch.tensor([energy], dtype=torch.float32),
    }


def finalized(metrics):
    return {name: total / count for name, (total, count) in metrics.items() if count}


def test_common_particle_metrics_match_permuted_particles_and_find_duplicate():
    targets = make_collection(
        cls_id=[1, 2],
        pt=[10.0, 20.0],
        eta=[0.0, 1.0],
        phi=[0.0, 0.5],
        energy=[12.0, 24.0],
    )
    predictions = make_collection(
        cls_id=[2, 3, 1],
        pt=[20.0, 10.0, 10.0],
        eta=[1.0, 0.0, 0.02],
        phi=[0.5, 0.0, 0.0],
        energy=[24.0, 12.0, 12.0],
    )
    target_mask = torch.ones(1, 2, dtype=torch.bool)
    prediction_mask = torch.ones(1, 3, dtype=torch.bool)

    values = finalized(
        compute_validation_particle_metrics(
            targets,
            target_mask,
            predictions,
            prediction_mask,
            num_classes=6,
        )
    )

    assert values["count/target_mean"] == 2
    assert values["count/prediction_mean"] == 3
    assert values["count/bias_mean"] == 1
    assert values["count/mae"] == 1
    assert values["matching/efficiency"] == 1
    assert values["matching/purity"] == pytest.approx(2 / 3)
    assert values["matching/f1"] == pytest.approx(0.8)
    assert values["matching/duplicate_fraction"] == pytest.approx(1 / 3)
    assert values["matched/pid_accuracy"] == 0.5
    assert values["matched/delta_r_mean"] == 0
    assert values["event/energy_response_mean"] == pytest.approx(4 / 3)


def test_common_particle_metrics_are_prediction_permutation_invariant():
    targets = make_collection(
        cls_id=[1, 2],
        pt=[10.0, 20.0],
        eta=[0.0, 1.0],
        phi=[0.0, 0.5],
        energy=[12.0, 24.0],
    )
    predictions = make_collection(
        cls_id=[2, 1],
        pt=[20.0, 10.0],
        eta=[1.0, 0.0],
        phi=[0.5, 0.0],
        energy=[24.0, 12.0],
    )
    masks = (torch.ones(1, 2, dtype=torch.bool),) * 2
    original = compute_validation_particle_metrics(targets, masks[0], predictions, masks[1], num_classes=6)

    permutation = torch.tensor([1, 0])
    permuted = {name: value[:, permutation] for name, value in predictions.items()}
    reordered = compute_validation_particle_metrics(targets, masks[0], permuted, masks[1], num_classes=6)

    assert original == reordered


def test_validation_collections_restore_same_physical_targets_for_both_modes():
    element_target = torch.zeros(1, 3, 14)
    element_target[0, :2, 0] = torch.tensor([1, 2])
    element_target[0, :2, 3] = torch.tensor([0.1, -0.2])
    element_target[0, :2, 4] = torch.sin(torch.tensor([0.3, -0.4]))
    element_target[0, :2, 5] = torch.cos(torch.tensor([0.3, -0.4]))
    element_batch = PFBatch(
        X=torch.ones(1, 3, 15),
        ytarget=element_target,
        ytarget_pt_orig=torch.tensor([[10.0, 20.0, 0.0]]),
        ytarget_e_orig=torch.tensor([[12.0, 24.0, 0.0]]),
    )

    set_target = element_target[:, :2].clone()
    set_target[..., 2] = torch.log(torch.tensor([[10.0, 20.0]]))
    set_target[..., 6] = torch.log(torch.tensor([[12.0, 24.0]]))
    set_batch = PFBatch(X=torch.ones(1, 3, 15), ytarget_set=set_target)
    predictions = make_collection(
        cls_id=[1, 2, 0],
        pt=[10.0, 20.0, 0.0],
        eta=[0.1, -0.2, 0.0],
        phi=[0.3, -0.4, 0.0],
        energy=[12.0, 24.0, 0.0],
    )

    element_collections = validation_particle_collections(element_batch, predictions, OutputMode.ELEMENTWISE)
    set_collections = validation_particle_collections(set_batch, predictions, OutputMode.SET)

    torch.testing.assert_close(element_collections[0]["pt"][:, :2], set_collections[0]["pt"])
    torch.testing.assert_close(element_collections[0]["energy"][:, :2], set_collections[0]["energy"])
    assert element_collections[1].sum() == set_collections[1].sum() == 2
    assert element_collections[3].sum() == set_collections[3].sum() == 2


def test_elementwise_validation_reconstructs_targets_without_cached_values():
    X = torch.ones(1, 2, 15)
    X[..., 1] = torch.tensor([[2.0, 4.0]])
    X[..., 5] = torch.tensor([[3.0, 6.0]])
    target = torch.zeros(1, 2, 14)
    target[..., 0] = torch.tensor([[1, 2]])
    target[..., 2] = torch.log(torch.tensor([[5.0, 2.0]]))
    target[..., 4] = 0.0
    target[..., 5] = 1.0
    target[..., 6] = torch.log(torch.tensor([[4.0, 3.0]]))
    batch = PFBatch(X=X, ytarget=target)
    predictions = make_collection(
        cls_id=[1, 2],
        pt=[10.0, 8.0],
        eta=[0.0, 0.0],
        phi=[0.0, 0.0],
        energy=[12.0, 18.0],
    )

    targets, _, _, _ = validation_particle_collections(batch, predictions, OutputMode.ELEMENTWISE)

    torch.testing.assert_close(targets["pt"], torch.tensor([[10.0, 8.0]]))
    torch.testing.assert_close(targets["energy"], torch.tensor([[12.0, 18.0]]))


def test_matching_keeps_acceptable_pair_instead_of_balancing_unacceptable_ones():
    # P1 sits on T1; P2 and T2 are 0.11 on either side, so only P1-T1 passes the delta-R cut.
    # Uncapped, keeping P1-T1 costs 0 + (0.22 / 0.1)^2 = 4.84, while swapping to P1-T2 + P2-T1
    # costs 1.21 + 1.21 = 2.42, so the assignment would drop the only acceptable match.
    targets = make_collection(cls_id=[1, 1], pt=[10.0, 10.0], eta=[0.0, 0.11], phi=[0.0, 0.0], energy=[10.0, 10.0])
    predictions = make_collection(cls_id=[1, 1], pt=[10.0, 10.0], eta=[0.0, -0.11], phi=[0.0, 0.0], energy=[10.0, 10.0])
    mask = torch.ones(1, 2, dtype=torch.bool)

    values = finalized(compute_validation_particle_metrics(targets, mask, predictions, mask, num_classes=6))

    assert values["matching/efficiency"] == 0.5
    assert values["matching/purity"] == 0.5
    assert values["matched/delta_r_mean"] == 0


def test_matching_cost_cap_never_changes_an_acceptable_pair():
    delta_r = torch.linspace(0.0, MATCH_DR * (1 - 1e-4), 101)
    pt_ratio = torch.linspace((1 - MATCH_REL_PT) * (1 + 1e-4), (1 + MATCH_REL_PT) * (1 - 1e-4), 101)
    delta_r, log_pt_ratio = torch.meshgrid(delta_r, torch.log(pt_ratio).abs(), indexing="ij")
    uncapped = (delta_r / MATCH_DR).square() + (log_pt_ratio / MATCH_LOG_PT).square()

    assert uncapped.max() < MATCH_COST_CAP
    torch.testing.assert_close(_matching_cost(delta_r, log_pt_ratio), uncapped)
    assert _matching_cost(torch.tensor([3.0]), torch.tensor([0.0])).item() == MATCH_COST_CAP
