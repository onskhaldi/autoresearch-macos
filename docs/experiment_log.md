# AutoResearch on Apple Silicon — Research Log

**Seminar:** Auto-Research with Coding Agents  
**Student:** Ones Khaldi  
**Supervisor:** Prof. Paul Swoboda  


## 1. Research Objective

The goal is to study and reproduce the core autonomous experimentation workflow of Andrej Karpathy's *autoresearch* at a smaller scale.

The system should allow an AI coding agent to:

1. Propose modifications to model training code.
2. Execute training experiments.
3. Evaluate model performance.
4. Keep or reject changes based on experimental results.
5. Use previous observations to guide future experiments.

The reimplementation must be pedagogical, reproducible, and feasible on consumer hardware.

**Scope:** Proposed; awaiting supervisor approval.

## 2. Source Repository

- Original project: https://github.com/karpathy/autoresearch
- macOS adaptation: https://github.com/miolini/autoresearch-macos
- Working fork: https://github.com/onskhaldi/autoresearch-macos
- Development branch: `seminar-m2`

The macOS fork was selected because it includes modifications for Apple Silicon and retains a PyTorch-based workflow.

The repository will be used to study the original implementation and develop a simplified educational reproduction.

## 3. Hardware and Software Environment

### Hardware

| Component | Specification |
|---|---|
| Computer | MacBook Pro 13-inch, 2022 |
| Processor | Apple M2 |
| Unified memory | 8 GB |
| GPU backend | Apple Metal / MPS |
| Operating system | macOS Ventura 13 |

### Software

| Component | Version |
|---|---|
| Python (project environment) | 3.10.22 |
| PyTorch | 2.6.0 |
| uv | 0.13.4 |
| NumPy | 2.2.6 |
| Matplotlib | 3.10.8 |
| Development environment | VS Code |
| Version control | Git / GitHub |

### Environment Verification

The following command was executed:

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

Results:

```text
PyTorch: 2.6.0
MPS available: True
MPS built: True
```

**Conclusion:** PyTorch recognizes the Apple M2 GPU. Actual training compatibility and memory usage remain to be tested.

## 4. Original Training Configuration

The following parameters were inspected in `train.py` and `prepare.py`.

| Hyperparameter | Original value |
|---|---|
| Transformer layers | 4 |
| Embedding dimension | 256 |
| Attention heads | 2 |
| Head dimension | 128 |
| Sequence length | 2,048 |
| Device batch size | 16 |
| Effective batch size | 65,536 tokens |
| Training time budget | 300 seconds |
| Weight decay | 0.2 |
| Embedding learning rate | 0.6 |
| Unembedding learning rate | 0.004 |
| Matrix learning rate | 0.04 |
| Scalar learning rate | 0.5 |
| Adam betas | (0.8, 0.95) |

The actual embedding dimension is derived from the model depth, aspect ratio, and head dimension.

The training configuration specifies a four-layer Transformer with 256-dimensional embeddings and two attention heads.

The code selects the MPS device when CUDA is unavailable and MPS is supported.

## 5. Initial Memory Analysis

The original microbatch processes:

\[
16 \times 2048 = 32768\text{ tokens}
\]

This may require substantial memory during training due to activations, gradients, model parameters, and optimizer states.

Our MacBook has only 8 GB of unified memory.

### Proposed adaptation

Reduce:

```python
DEVICE_BATCH_SIZE = 16
```

to:

```python
DEVICE_BATCH_SIZE = 1
```

Keep the original effective batch size initially.

The corresponding gradient accumulation would become:

\[
\frac{65536}{1 \times 2048}=32
\]

**Hypothesis:** Reducing the microbatch size should reduce peak activation memory usage, potentially allowing training to run on the M2.

**Trade-off:** More gradient accumulation steps may increase runtime and reduce the number of optimizer updates completed within the five-minute training budget.

**Status:** Proposed, not yet experimentally validated.

## 6. Planned Baseline Experiment

**Experiment ID:** B0

**Objective:** Determine whether the smaller training configuration runs successfully on Apple Silicon and establish an initial validation-performance benchmark.

### Planned measurements

- Total number of trainable parameters
- Validation bits per byte (`val_bpb`)
- Training loss
- Optimizer steps completed
- Training duration
- Tokens processed per second
- Memory usage
- Errors or warnings

### Results

Not yet available. No baseline training has been executed.

## 7. Planned Autonomous Experiments

After establishing a working baseline, the coding agent will propose modifications to the training code and evaluate their effects.

For each experiment, record:

| ID | Hypothesis | Code modification | Validation BPB | Runtime | Outcome |
|---|---|---|---|---|---|
| B0 | Baseline | Initial configuration | Pending | Pending | Pending |
| E1 | Pending | Pending | Pending | Pending | Pending |
| E2 | Pending | Pending | Pending | Pending | Pending |
| E3 | Pending | Pending | Pending | Pending | Pending |

All experiments should use comparable evaluation conditions.

## 8. Open Questions

1. Does the current model fit in 8 GB of unified memory?
2. How many optimizer updates can be completed within five minutes?
3. Does reducing the batch size create unacceptable training overhead?
4. How should we constrain the coding agent's permitted modifications?
5. How many experiments can realistically be completed before the presentation?
6. How do we compare autonomous modifications fairly with a baseline?

## 9. Current Progress — 21 October 2026

**Completed:**

- Forked and cloned the macOS repository.
- Created and published the `seminar-m2` development branch.
- Installed uv and project dependencies.
- Verified PyTorch MPS availability.
- Inspected key model hyperparameters.
- Identified a possible memory-related adaptation.

**Pending:**

- Inspect the full training loop.
- Confirm that training data is prepared.
- Apply and test the reduced microbatch configuration.
- Run the first baseline experiment.
- Measure training performance and memory usage.
- Obtain supervisor feedback on reimplementation scope.

## 10. Preliminary Observations

The existing macOS implementation provides an appropriate starting point for studying autoresearch.

The primary anticipated constraint is memory capacity rather than lack of GPU support.

No claims about training performance or autonomous improvements can be made until experiments have been run.

---

*This log will be updated after each implementation change and experiment. Observed results will be clearly distinguished from hypotheses and planned experiments.*