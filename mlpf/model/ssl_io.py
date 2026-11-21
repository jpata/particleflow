"""Deterministic bulk reads and worker prefetching for SSL."""

import bisect

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from mlpf.model.ssl import collate_inputs


def fetch_records(dataset, indices):
    """Bulk fetch through subsets/concatenations, restoring original batch order."""
    indices = [int(i) for i in indices]
    if isinstance(dataset, Subset):
        return fetch_records(dataset.dataset, [dataset.indices[i] for i in indices])
    if isinstance(dataset, ConcatDataset):
        grouped = {}
        for position, index in enumerate(indices):
            part = bisect.bisect_right(dataset.cumulative_sizes, index)
            offset = dataset.cumulative_sizes[part - 1] if part else 0
            grouped.setdefault(part, []).append((position, index - offset))
        result = [None] * len(indices)
        for part, requests in grouped.items():
            records = fetch_records(dataset.datasets[part], [index for _, index in requests])
            for (position, _), record in zip(requests, records):
                result[position] = record
        return result
    if hasattr(dataset, "__getitems__"):
        return dataset.__getitems__(indices)
    return [dataset[i] for i in indices]


class StepBatches(Dataset):
    def __init__(self, dataset, seed, batch_size, first, last):
        self.dataset, self.seed, self.batch_size = dataset, seed, batch_size
        self.first, self.last = first, last

    def __len__(self):
        return max(0, self.last - self.first + 1)

    def __getitem__(self, index):
        step = self.first + index
        generator = torch.Generator().manual_seed(self.seed + step)
        indices = torch.randint(len(self.dataset), (self.batch_size,), generator=generator)
        return collate_inputs(fetch_records(self.dataset, indices))


def training_loader(dataset, config, first, device):
    batches = StepBatches(dataset, config.seed, config.batch_size, first, config.steps)
    kwargs = {}
    if config.num_workers:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=config.prefetch_factor, persistent_workers=True)
    # DataLoader worker seeding must not advance the model's RNG on resume.
    return DataLoader(
        batches,
        batch_size=None,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
        **kwargs,
    )
