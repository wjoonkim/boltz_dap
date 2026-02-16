"""
Boltz DAP v2 - Proper FastFold-style Dynamic Axial Parallelism for Boltz 2.

Shards the pair representation z across GPUs to reduce peak activation memory.
No model duplication — only activations are distributed.
"""
