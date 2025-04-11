"""Test set-up and fixtures code."""

import importlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import DataLoader

from meds_torchdata import MEDSPytorchDataset, MEDSTorchBatch

@pytest.fixture(scope="session")
def sample_batch(sample_pytorch_dataset: MEDSPytorchDataset) -> MEDSTorchBatch:
    dataloader = DataLoader(sample_pytorch_dataset, batch_size=4, shuffle=False,
                            collate_fn=sample_pytorch_dataset.collate_fn)
    yield next(iter(dataloader))

@pytest.fixture(scope="session")
def sample_batch_with_task(sample_pytorch_dataset_with_task: MEDSPytorchDataset) -> MEDSTorchBatch:
    dataloader = DataLoader(sample_pytorch_dataset_with_task, batch_size=4, shuffle=False,
                            collate_fn=sample_pytorch_dataset_with_task.collate_fn)
    yield next(iter(dataloader))

@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    sample_pytorch_dataset: MEDSPytorchDataset,
    sample_pytorch_dataset_with_task: MEDSPytorchDataset,
    sample_batch: MEDSTorchBatch,
    sample_batch_with_task: MEDSTorchBatch,
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "sample_pytorch_dataset": sample_pytorch_dataset,
            "sample_pytorch_dataset_with_task": sample_pytorch_dataset_with_task,
            "sample_batch": sample_batch,
            "sample_batch_with_task": sample_batch_with_task,
        }
    )
