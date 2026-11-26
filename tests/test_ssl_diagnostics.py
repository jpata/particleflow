import torch

from mlpf.ssl_diagnostics import distribution_metrics, replace_context, sample_weights


def test_replacement_preserves_queries_counts_and_detector_type():
    x = torch.tensor([[1., 10.], [2., 20.], [1., 30.], [2., 40.]])
    donor = torch.tensor([[1., 100.], [2., 200.], [1., 300.], [2., 400.]])
    query = torch.tensor([0, 3])
    changed, reused = replace_context(x, query, donor, torch.Generator().manual_seed(1))
    torch.testing.assert_close(changed[query], x[query])
    torch.testing.assert_close(changed[:, 0], x[:, 0])
    assert (changed[[1, 2], 1] >= 100).all()
    assert reused == 0


def test_identical_distributions_have_zero_divergence():
    p = torch.tensor([[.9, .1], [.2, .8]])
    metrics = distribution_metrics(p, p)
    torch.testing.assert_close(metrics["kl"], torch.zeros(2))
    torch.testing.assert_close(metrics["js"], torch.zeros(2))
    torch.testing.assert_close(metrics["agreement"], torch.ones(2))
    torch.testing.assert_close(metrics["cross_entropy"], metrics["target_entropy"])


def test_weighting_balances_events_then_types_not_hit_counts():
    cache = {"event": torch.tensor([0, 0, 0, 1, 1]), "x": torch.tensor([[1.], [1.], [2.], [2.], [2.]])}
    torch.testing.assert_close(sample_weights(cache), torch.tensor([.125, .125, .25, .25, .25]))
