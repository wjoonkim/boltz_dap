# Getting started: run Boltz-2 with DAP (first-time users)

This guide walks you through using this repo to run **Boltz-2** structure prediction with **DAP** (multiple GPUs, no single-GPU OOM on large complexes). You do not modify Boltz-2; this repo **wraps** it at runtime.

---

## Step 1: Prerequisites

- **Hardware**: 2 or more GPUs on one machine (e.g. 2, 4, or 8). Same node required for DAP.
- **Software**: Python 3.10+, PyTorch 2.x with CUDA, and **Boltz-2** installed.

If you are on an **HPC cluster**, you will need to allocate GPUs (e.g. `srun --gres=gpu:4 --pty bash`) and activate a Python environment that has PyTorch and Boltz before running anything below. See [Checking your environment](boltz2_dap_prediction.md#checking-your-environment-for-beginners) in the prediction guide.

---

## Step 2: Install Boltz-2 (base model)

This repo depends on the official **Boltz-2** package. Install it first:

```bash
pip install boltz
```

That will pull in Boltz-2 and its dependencies. The first time you run the DAP script, it will download the model checkpoint and molecule data into `~/.boltz` (or `--cache` if you set it).

---

## Step 3: Clone this repo and use it (no extra install)

You do **not** need to `pip install` this repo. Clone it and run the script from the repo root so that `boltz_dap_v2` and `boltz_distributed` are on the Python path:

```bash
git clone https://github.com/coqylight/boltz_dap.git
cd boltz_dap
# Use the same Python/env where you installed boltz and PyTorch
```

All commands below assume you are inside `boltz_dap/`.

---

## Step 4: Prepare input (YAML config)

You need a **YAML config file** that describes your target(s), same format as serial Boltz-2.

**Option A — Use an MSA server (easiest):**  
Your YAML can list only the sequence(s). MSA will be fetched from a remote server when you pass `--use_msa_server` (e.g. ColabFold). No need to have huge MSA files locally.

**Option B — Local MSA:**  
Your YAML points to local paths for MSA (and optionally structures, templates). See [Boltz-2](https://github.com/jwohlwend/boltz) docs for the exact YAML format.

Example minimal YAML (one protein, sequence only; use with `--use_msa_server`):

```yaml
# my_target.yaml
metadata:
  id: my_target
targets:
  - id: my_target
    sequence: MKLV...  # your protein sequence
```

Save this as e.g. `input.yaml` in the repo root or anywhere you like; you will pass its path to the script.

---

## Step 5: Run DAP

From the **repo root** (`boltz_dap/`), run:

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
  boltz_dap_v2/run_boltz_dap_v2.py \
  input.yaml \
  --out_dir ./output \
  --cache ~/.boltz
```

- Replace `4` with your number of GPUs (2, 4, 8, etc.).
- Replace `input.yaml` with your YAML path.
- Replace `./output` with the directory where you want predictions and logs.

If you use an MSA server:

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
  boltz_dap_v2/run_boltz_dap_v2.py \
  input.yaml \
  --out_dir ./output \
  --cache ~/.boltz \
  --use_msa_server
```

**Large complexes (e.g. hexamer) or many samples (e.g. 25)?** Add `--use_flex_attention_chunked` to reduce OOM risk:

```bash
torchrun --nnodes 1 --nproc_per_node 4 \
  boltz_dap_v2/run_boltz_dap_v2.py \
  input.yaml \
  --out_dir ./output \
  --cache ~/.boltz \
  --use_msa_server \
  --use_flex_attention_chunked \
  --recycling_steps 10 \
  --diffusion_samples 25
```

---

## Step 6: Where to find the results

After a successful run, under `--out_dir` you will see:

- **`predictions/`** — CIF (or PDB) structure files and confidence outputs (pLDDT, pTM, PAE, etc.).
- **`processed/`** — Preprocessed inputs and manifest.
- **`gpu_memory.log`** — GPU memory usage (rank 0).

Open the CIF files with PyMOL, ChimeraX, or any structure viewer. Confidence JSON/PAE are in the same prediction folder.

---

## Summary checklist (first-time run)

1. Install Boltz-2: `pip install boltz`
2. Clone this repo: `git clone https://github.com/coqylight/boltz_dap.git && cd boltz_dap`
3. (Optional) Check GPUs and NCCL: see [Checking your environment](boltz2_dap_prediction.md#checking-your-environment-for-beginners)
4. Prepare `input.yaml` (sequence + optional MSA paths, or use `--use_msa_server`)
5. Run: `torchrun --nproc_per_node <N> boltz_dap_v2/run_boltz_dap_v2.py input.yaml --out_dir ./output [--use_msa_server] [other options]`
6. Check `./output/predictions/` for structures and confidence.

For all CLI options and pipeline details, see [boltz2_dap_prediction.md](boltz2_dap_prediction.md).
