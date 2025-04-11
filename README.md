# MEDS TorchEncoders

Helpers to build PyTorch AI models over MEDS datasets.

# Installation

```bash
pip install meds-torch-encoders
```

# Model Components

We subdivide the modeling into the following steps:

```mermaid
flowchart LR;
    R[MEDS Dataset] --> P{`MEDSPytorchDataset`} --> B{MEDSTorchBatch}
    B -->|InputEncoder| E{ShallowEmbeddedBatch}
    E -->|TrajectoryEncoder| T{DeepEmbeddedBatch}
    B -->|CodesOnlyTrajectoryEncoder| T
```
