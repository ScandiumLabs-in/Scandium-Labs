from __future__ import annotations

import math
import os
from typing import Iterator

import torch
from torch.utils.data import Sampler


def precompute_graph_sizes(graph_dir: str, indices: list[int]) -> list[int]:
    """Load num_nodes from each cached .pt file (reads metadata only)."""
    sizes = []
    for idx in indices:
        path = os.path.join(graph_dir, f"{idx}.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=False)
            cg = data[0]
            lg = data[1]
            total = (cg.num_nodes if hasattr(cg, "num_nodes") else 0) + \
                    (lg.num_nodes if hasattr(lg, "num_nodes") else 0)
            sizes.append(total)
        else:
            sizes.append(0)
    return sizes


class SizeBucketedBatchSampler(Sampler[list[int]]):
    """Yields batches of indices grouped by graph size.

    Graphs sorted by (num_nodes + num_line_graph_nodes) are split into
    buckets, each bucket contains at most ``bucket_size_mult`` ×
    ``batch_size`` indices.  Within each bucket indices are shuffled
    every epoch, then drawn in contiguous chunks of ``batch_size``.
    """

    def __init__(
        self,
        indices: list[int],
        sizes: list[int] | None = None,
        graph_dir: str | None = None,
        batch_size: int = 16,
        bucket_size_mult: float = 2.0,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle_flag = shuffle
        self.drop_last = drop_last

        if sizes is None and graph_dir is not None:
            sizes = precompute_graph_sizes(graph_dir, indices)
        elif sizes is None:
            raise ValueError("Provide either sizes or graph_dir")

        sizes_arr = list(sizes)
        self._sorted_indices = [i for _, i in sorted(zip(sizes_arr, indices))]
        bucket_capacity = max(1, int(batch_size * bucket_size_mult))
        self._buckets: list[list[int]] = []
        for start in range(0, len(self._sorted_indices), bucket_capacity):
            end = min(start + bucket_capacity, len(self._sorted_indices))
            self._buckets.append(self._sorted_indices[start:end])

    def __iter__(self) -> Iterator[list[int]]:
        bucket_batches: list[list[int]] = []
        for bucket in self._buckets:
            perm = bucket[:]
            if self.shuffle_flag:
                import random
                random.shuffle(perm)
            for i in range(0, len(perm), self.batch_size):
                chunk = perm[i:i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                bucket_batches.append(chunk)
        if self.shuffle_flag:
            import random
            random.shuffle(bucket_batches)
        return iter(bucket_batches)

    def __len__(self) -> int:
        n = 0
        for bucket in self._buckets:
            n += len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size != 0:
                n += 1
        return n
