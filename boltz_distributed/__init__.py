"""
Boltz Distributed - DAP (Dynamic Axial Parallelism) for Boltz

Ported from FastFold's DAP implementation for multi-GPU inference.
"""

from .core import init_dap, get_dap_size, get_dap_rank, is_dap_initialized
from .comm import (
    scatter, gather, reduce, copy,
    row_to_col, col_to_row,
    _split, _gather, _reduce
)
from .wrappers import (
    DAPMSALayer, DAPPairformerLayer,
    wrap_msa_module_with_dap, wrap_pairformer_with_dap,
    inject_dap
)

__all__ = [
    # Initialization
    'init_dap', 'get_dap_size', 'get_dap_rank', 'is_dap_initialized',
    # Communication primitives
    'scatter', 'gather', 'reduce', 'copy',
    'row_to_col', 'col_to_row',
    '_split', '_gather', '_reduce',
    # Wrappers
    'DAPMSALayer', 'DAPPairformerLayer',
    'wrap_msa_module_with_dap', 'wrap_pairformer_with_dap',
    'inject_dap'
]

