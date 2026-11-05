import random
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from mlpf.model.training import _particle_metric_event_indices, _validation_event_count


@pytest.mark.parametrize("size,limit", [(1033, 128), (10, 3), (129, 128), (1, 1), (10, 0), (5, 128)])
def test_particle_event_sampling_is_bounded_stratified_and_reproducible(size, limit):
    state = random.getstate()
    selected = _particle_metric_event_indices(size, limit, 12345)
    assert random.getstate() == state
    assert selected == _particle_metric_event_indices(size, limit, 12345)
    if limit == 0 or size <= limit:
        assert selected is None
    else:
        assert len(selected) == limit
        for i, index in enumerate(sorted(selected)):
            assert i * size // limit <= index < (i + 1) * size // limit


def test_validation_event_count_handles_groups_and_partial_batches():
    first = DataLoader(torch.zeros(11), batch_size=4)
    second = DataLoader(torch.zeros(7), batch_size=3)
    assert _validation_event_count(SimpleNamespace(data_loaders=[first, second])) == 18
    assert _validation_event_count(DataLoader(torch.zeros(11), batch_size=4, drop_last=True)) == 8
    assert _validation_event_count([SimpleNamespace(X=torch.zeros(3, 2)), SimpleNamespace(X=torch.zeros(1, 2))]) == 4
