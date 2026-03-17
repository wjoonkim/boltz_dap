# Boltz-DAP Distributed Prediction with Row-Sharded Parallelism

> **Note:** The current implementation supports Boltz-2 only.

This document describes how to run distributed structure prediction
using `boltz_dap_v2/run_boltz_dap_v2.py`, which provides a Click CLI for
row-sharded **DAP (Dynamic Axial Parallelism)** across multiple GPUs. No single
GPU holds the full pair tensor `z [B, N, N, D]`; activations are sharded along
the row dimension so that peak memory scales down with the number of GPUs.

---

## Entrypoint

The distributed prediction CLI is:

```
boltz_dap_v2/run_boltz_dap_v2.py <DATA> [options]
```

`DATA` is the path to an input YAML config file (or directory of configs), as in
serial Boltz. The script initializes PyTorch distributed (DAP), resolves the
model checkpoint and molecule directory from `--cache` if needed, then runs the
full pipeline: preprocessing (optional MSA server), model load, DAP injection,
scatter → trunk → gather → distogram, diffusion, structure, and confidence.

### Checking your environment (for beginners)

Before running a long job, it helps to confirm that (1) the right number of GPUs
are visible and (2) processes can talk to each other (NCCL / PyTorch distributed).

**If you are on an HPC cluster or similar:** You must **allocate GPU resources first**
(e.g. `srun --gres=gpu:4 --pty bash` or an interactive job with GPUs). The steps
below assume you are already on a node that has GPUs assigned. Also **activate
the same Python environment** (venv or conda) where PyTorch and Boltz are
installed before running the PyTorch or NCCL checks; otherwise you may see
`ModuleNotFoundError: No module named 'torch'` if the default `python` has no
PyTorch.

**1. How many GPUs do you have?**

```bash
nvidia-smi -L
```

You should see one line per GPU (e.g. `GPU 0: NVIDIA H800`). Use the same
number for `--nproc_per_node` when you launch DAP.

**2. Does PyTorch see them?**

From the same environment you will use for DAP (e.g. your venv):

```bash
python -c "import torch; print('GPUs visible to PyTorch:', torch.cuda.device_count())"
```

The count should match `nvidia-smi -L`. If it is 0, check that PyTorch was
installed with CUDA and that `CUDA_VISIBLE_DEVICES` is not set to hide GPUs.

**3. Can the processes talk (NCCL)?**

Save the following as `test_nccl.py` and run it with the same process count you
will use for DAP (e.g. 4):

```python
# test_nccl.py
import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    t = torch.ones(1, device=f"cuda:{local_rank}") * rank
    dist.all_reduce(t)
    print(f"Rank {rank}/{size} (cuda:{local_rank}), all_reduce(sum of ranks)={t.item():.0f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

```bash
torchrun --nnodes 1 --nproc_per_node 4 test_nccl.py
```

You should see four lines, one per rank, and the `all_reduce(sum of ranks)`
value should be `6` (0+1+2+3) on every rank. If this hangs or errors, fix
NCCL/network (e.g. firewall, CUDA driver, or `NCCL_DEBUG=WARN` for hints)
before running the full DAP script.

**4. Optional: NVLink**

On multi-GPU nodes, NVLink can speed up DAP communication. To see topology:

```bash
nvidia-smi nvlink --status
```

If you have NVLink between GPUs, no extra config is needed; PyTorch/NCCL will
use it when available.

### Launching with `torchrun` or `srun`

The script is designed to be launched with **either** `torchrun` (single-node  
or multi-node outside SLURM) **or** `srun` (SLURM). Each process binds to one
GPU via `LOCAL_RANK`; the total number of processes is the DAP size (no
separate data-parallel dimension).

- `**torchrun`** — set `--nproc_per_node` (and optionally `--nnodes`) so that
`world_size` equals the number of GPUs you want for DAP.
- `**srun**` — set `--ntasks` or `--ntasks-per-node` × `--nodes` so that the
total task count equals the DAP size.

Example with `torchrun` (single node, 4 GPUs):

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
  boltz_dap_v2/run_boltz_dap_v2.py \
  input.yaml \
  --out_dir ./output \
  --cache ~/.boltz \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 1
```

Example with `srun` (SLURM, 4 GPUs on one node):

```bash
srun torchrun --nproc_per_node=4 \
  boltz_dap_v2/run_boltz_dap_v2.py \
  input.yaml \
  --out_dir ./output \
  --cache ~/.boltz \
  --recycling_steps 3
```

For large complexes (e.g. hexamer, 25 diffusion samples), add
`--use_flex_attention_chunked` to avoid OOM. For MSA generation without local
MSA files, use `--use_msa_server`.

---

## Input Data

Input is the same as serial Boltz: **config file(s)** (YAML or FASTA).

- **YAML**: path to a single YAML or a directory containing YAML/FASTA files.
Each entry describes the target(s) and, if not using the MSA server, paths to
MSA/structure/template data.
- **MSA server**: if `--use_msa_server` is set, MSA can be generated remotely
(e.g. ColabFold); local MSA paths in the YAML are not required.

Rank 0 runs preprocessing (including optional MSA server calls), then all ranks
proceed with model load and DAP inference. Output is written under `--out_dir`
(including `predictions/`, `processed/`, etc.).

---

## CLI Options

All options are provided as Click flags on the script.

### Required Arguments


| Argument    | Description                                  |
| ----------- | -------------------------------------------- |
| `DATA`      | Path to input YAML config file or directory. |
| `--out_dir` | Output directory for predictions and logs.   |


### Common Options


| Option    | Type | Default    | Description                             |
| --------- | ---- | ---------- | --------------------------------------- |
| `--cache` | path | `~/.boltz` | Cache for checkpoint and CCD molecules. |


### Diffusion and Recycling


| Option                | Type | Default | Description                              |
| --------------------- | ---- | ------- | ---------------------------------------- |
| `--recycling_steps`   | int  | `3`     | Number of trunk recycling iterations.    |
| `--sampling_steps`    | int  | `200`   | Number of diffusion denoising steps.     |
| `--diffusion_samples` | int  | `1`     | Number of independent diffusion samples. |


### MSA and Input


| Option             | Type | Default | Description                             |
| ------------------ | ---- | ------- | --------------------------------------- |
| `--use_msa_server` | flag | `False` | Use remote MSA server (e.g. ColabFold). |


### Attention and Kernels


| Option                         | Type | Default | Description                                                    |
| ------------------------------ | ---- | ------- | -------------------------------------------------------------- |
| `--no_kernels`                 | flag | `False` | Disable cuequivariance CUDA kernels (use PyTorch-native).      |
| `--use_flex_attention`         | flag | `False` | Use FlexAttention for triangle attention (single-GPU style).   |
| `--use_flex_attention_chunked` | flag | `False` | Chunked FlexAttention for DAP (avoids OOM on large N/samples). |


### Optional Features


| Option             | Type | Default | Description                               |
| ------------------ | ---- | ------- | ----------------------------------------- |
| `--use_potentials` | flag | `False` | Enable FK steering and physical guidance. |
| `--seed`           | int  | `None`  | Random seed for reproducibility.          |


### Output and Behavior

- **Confidence** (pLDDT, pTM, iPTM, PAE, PDE) is computed when the model
supports it; no separate flag.
- GPU memory usage is logged to `<out_dir>/gpu_memory.log` (rank 0).

---

## Inference Pipeline Stages

The `run_boltz_dap_v2.py` flow runs the following stages:

### 1. Distributed Setup

- Initializes PyTorch process group (NCCL for CUDA). `world_size` is the DAP
size; there is no separate data-parallel dimension.
- Each rank binds to `cuda:{LOCAL_RANK}`.

### 2. Preprocessing (Rank 0)

- Rank 0 loads the config, runs Boltz preprocessing (including optional MSA
server calls), and writes preprocessed data under `out_dir`. Other ranks
wait at a barrier.

### 3. Model Loading

- All ranks load the same Boltz-2 checkpoint from `--cache` (or default) into
CPU. Optional: apply FlexAttention patch (or chunked FlexAttention patch)
before moving modules to GPU.

### 4. Selective GPU Placement

- **GPU 0**: Full model (trunk + distogram, diffusion, structure, confidence).
- **GPU 1, 2, …**: Trunk-only modules (input embedder, MSA, pairformer stack,
confidence pairformer) so that all ranks participate in the DAP trunk;
post-trunk modules stay on CPU and are used only on GPU 0 after gather.

### 5. DAP Injection

- Wraps trunk (and confidence pairformer) layers with DAP-aware versions:
scatter/gather, broadcast-chunked triangle multiplication, bias-only gather
for triangle and sequence attention.

### 6. Prediction Loop

- **Scatter**: Full `z` is split along the row dimension; each rank holds
`z_shard [B, N/P, N, D]`.
- **Trunk**: 48 Pairformer layers (and template/MSA modules) run on shards;
triangle multiplications use broadcast-chunked; triangle/seq attention gather
only bias.
- **Gather**: Rank 0 gathers full `z` on GPU 0.
- **Post-trunk**: Distogram, diffusion, structure, and confidence run on GPU 0
(confidence pairformer is already DAP-wrapped and runs across all ranks, then
results are merged on rank 0).

### 7. Output

- Rank 0 writes structures (CIF/PDB), confidence outputs, and any requested
artifacts under `--out_dir`.

---

## DAP-Specific Notes

- **No `size_dp` / `size_cp`**: The number of GPUs is the DAP size. Use
`torchrun --nproc_per_node=N` (and optionally `--nnodes`) or SLURM
`--ntasks` so that the total process count equals the desired DAP size.
- **Memory**: Peak memory per GPU decreases roughly with the number of GPUs
(row sharding). For very large N or many diffusion samples, use
`--use_flex_attention_chunked` to avoid OOM.
- **Numerical**: Results may differ slightly from single-GPU Boltz 2 due to
distributed reduction order; structure quality (e.g. LDDT, TM-score) is
statistically equivalent.

---

## Example Log

A full example run that produced 25 CIF files (hexamer, 4 GPUs,
`--use_flex_attention_chunked`, AF3-style defaults) is provided in the repo
root: [example_hexamer_25cif_full.log](../example_hexamer_25cif_full.log)
(~8.8 MB).