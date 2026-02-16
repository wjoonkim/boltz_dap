# Boltz DAP v2

Distributed Attention Parallelism for [Boltz 2](https://github.com/jwohlwend/boltz) protein structure prediction.

Enables inference on larger protein complexes by distributing the pair representation `z [B, N, N, D]` across multiple GPUs using row-sharding, with all-to-all communication for operations that need full columns.

## What's included

| File | Description |
|---|---|
| `boltz_dap_v2/run_boltz_dap_v2.py` | Entry point — drop-in replacement for `boltz predict` |
| `boltz_dap_v2/dap_trunk.py` | Main DAP forward pass (scatter z → trunk → gather → post-trunk) |
| `boltz_dap_v2/dap_pairformer.py` | DAP-wrapped PairformerLayer |
| `boltz_dap_v2/dap_pairformer_noseq.py` | DAP-wrapped PairformerNoSeqLayer (template & confidence) |
| `boltz_dap_v2/dap_tri_att.py` | SDPA-based triangle attention (replaces manual Q@K^T) |
| `boltz_dap_v2/dap_trimul.py` | Broadcast-chunked triangle multiplication |
| `boltz_dap_v2/dap_msa.py` | DAP-wrapped MSA layers |
| `boltz_dap_v2/dap_confidence.py` | DAP-wrapped confidence module |
| `boltz_distributed/` | Communication primitives (scatter, gather, row↔col) |

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- `boltz` (pip install boltz)
- `flash-attn` (optional, for FlashAttention backend in SDPA)

## Usage

```bash
# 2 GPUs
torchrun --nproc_per_node=2 boltz_dap_v2/run_boltz_dap_v2.py input.yaml --cache ~/.boltz

# 4 GPUs (H800 superpod)
torchrun --nproc_per_node=4 boltz_dap_v2/run_boltz_dap_v2.py input.yaml --cache ~/.boltz
```

## Benchmarks (2× RTX 5880, 48 GB each)

| Input | N | Original Boltz | DAP v2 |
|---|---|---|---|
| Pentamer (5 × 519 aa) | 2,595 | ❌ OOM (38 GB) | ✅ 36.5 GB, 16 min |

## Key optimizations

- **Row-sharded z**: Each GPU holds N/dap_size rows during trunk
- **SDPA triangle attention**: Fused kernel, −85% transient memory
- **Broadcast-chunked tri_mul**: Stream partner shards, never materialize full tensor
- **Chunked transitions**: `chunk_size=128` for all Transition layers
- **CPU offloading**: Each module offloaded after use in post-trunk
