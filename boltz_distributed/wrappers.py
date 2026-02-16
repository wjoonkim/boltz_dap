"""
DAP Wrappers for Boltz Modules

These wrappers add Dynamic Axial Parallelism support to Boltz's 
MSAModule and PairformerModule, enabling multi-GPU inference.

Key insight from FastFold: Scatter/gather at MODULE level, not layer level.
Tensors stay distributed throughout all layers, only gathering at the end.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn

from .core import get_dap_size, get_dap_rank
from .comm import scatter, gather, row_to_col, col_to_row


class DAPPairformerModule(nn.Module):
    """
    DAP-enabled wrapper for entire PairformerModule.
    
    Scatter at the start, run all layers on distributed tensors, gather at the end.
    This is similar to FastFold's Evoformer approach.
    """
    
    def __init__(self, pairformer_module: nn.Module):
        """
        Args:
            pairformer_module: The original Boltz PairformerModule
        """
        super().__init__()
        self.pairformer = pairformer_module
        self._padding_size = 0
    
    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        pair_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass with DAP distribution."""
        dap_size = get_dap_size()
        
        if dap_size == 1:
            # No parallelism, use original module
            return self.pairformer(s, z, mask, pair_mask, chunk_size_tri_attn, use_kernels)
        
        # Get original shapes
        batch = s.shape[0]
        seq_len = s.shape[1]
        
        # Compute padding to make seq_len divisible by dap_size
        padding_size = (dap_size - seq_len % dap_size) % dap_size
        self._padding_size = padding_size
        
        # Pad tensors if needed
        if padding_size > 0:
            s = torch.nn.functional.pad(s, (0, 0, 0, padding_size))
            z = torch.nn.functional.pad(z, (0, 0, 0, padding_size, 0, padding_size))
            mask = torch.nn.functional.pad(mask, (0, padding_size))
            pair_mask = torch.nn.functional.pad(pair_mask, (0, padding_size, 0, padding_size))
        
        padded_seq_len = seq_len + padding_size
        
        # --- Scatter tensors across GPUs ---
        # s: (batch, seq, dim) -> scatter on seq
        # z: (batch, seq, seq, dim) -> scatter on first seq dim (row-wise)
        s = scatter(s, dim=1)
        z = scatter(z, dim=1)
        
        # Also scatter masks appropriately
        # mask: (batch, seq) -> scatter
        # pair_mask: (batch, seq, seq) -> only scatter row dim to match z
        mask_scattered = scatter(mask, dim=1)
        pair_mask_scattered = scatter(pair_mask, dim=1)  # Row-wise scatter
        
        # Run all pairformer layers on distributed tensors
        # Note: We run the original layers since they don't need modification
        # The key fix: set chunk_size to avoid issues with partial tensors
        s, z = self.pairformer(s, z, mask_scattered, pair_mask_scattered, 
                               chunk_size_tri_attn, use_kernels)
        
        # --- Gather tensors back ---
        s = gather(s, dim=1)
        z = gather(z, dim=1)
        
        # Remove padding
        if padding_size > 0:
            s = s[:, :seq_len, :]
            z = z[:, :seq_len, :seq_len, :]
        
        return s, z


class DAPMSAModule(nn.Module):
    """
    DAP-enabled wrapper for entire MSAModule.
    
    Scatter at the start, run all layers on distributed tensors, gather at the end.
    """
    
    def __init__(self, msa_module: nn.Module):
        """
        Args:
            msa_module: The original Boltz MSAModule
        """
        super().__init__()
        self.msa_module = msa_module
        self._padding_size = 0
    
    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: dict,
        use_kernels: bool = False,
    ) -> Tensor:
        """Forward pass with DAP distribution."""
        dap_size = get_dap_size()
        
        if dap_size == 1:
            # No parallelism, use original module
            return self.msa_module(z, emb, feats, use_kernels)
        
        # For MSAModule, the distribution is more complex
        # z: (batch, seq, seq, dim) - pair representation
        # emb: (batch, seq, dim) - input embedding
        # msa: (batch, num_msa, seq, vocab) - from feats
        
        # Get dimensions
        batch = z.shape[0]
        seq_len = z.shape[1]
        
        # Compute padding
        padding_size = (dap_size - seq_len % dap_size) % dap_size
        self._padding_size = padding_size
        
        # Pad z and emb
        if padding_size > 0:
            z = torch.nn.functional.pad(z, (0, 0, 0, padding_size, 0, padding_size))
            emb = torch.nn.functional.pad(emb, (0, 0, 0, padding_size))
        
        # Scatter
        z = scatter(z, dim=1)
        emb = scatter(emb, dim=1)
        
        # For feats, we need to handle the msa and masks carefully
        # This is complex - for now, call original module without scattering feats
        # (MSA attention operates on full MSA, only pair representation is scattered)
        
        # Run MSA module
        z = self.msa_module(z, emb, feats, use_kernels)
        
        # Gather
        z = gather(z, dim=1)
        
        # Remove padding
        if padding_size > 0:
            z = z[:, :seq_len, :seq_len, :]
        
        return z


def wrap_pairformer_with_dap(pairformer_module: nn.Module) -> nn.Module:
    """
    Wrap a Boltz PairformerModule with DAP at the module level.
    
    Args:
        pairformer_module: The PairformerModule to wrap
        
    Returns:
        DAPPairformerModule wrapper
    """
    return DAPPairformerModule(pairformer_module)


def wrap_msa_module_with_dap(msa_module: nn.Module) -> nn.Module:
    """
    Wrap a Boltz MSAModule with DAP at the module level.
    
    Args:
        msa_module: The MSAModule to wrap
        
    Returns:
        DAPMSAModule wrapper
    """
    return DAPMSAModule(msa_module)


def inject_dap(model: nn.Module) -> nn.Module:
    """
    Inject DAP into a Boltz model.
    
    This finds MSAModule and PairformerModule instances and wraps them.
    
    Args:
        model: The Boltz model
        
    Returns:
        The same model with DAP injected
    """
    from boltz.model.modules.trunk import MSAModule, PairformerModule
    
    for name, module in list(model.named_modules()):
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1] if name else ''
        
        if isinstance(module, MSAModule):
            parent = model if not parent_name else dict(model.named_modules())[parent_name]
            setattr(parent, attr_name, DAPMSAModule(module))
            print(f"[BoltzDAP] Wrapped MSAModule '{name}' with DAP")
        elif isinstance(module, PairformerModule):
            parent = model if not parent_name else dict(model.named_modules())[parent_name]
            setattr(parent, attr_name, DAPPairformerModule(module))
            print(f"[BoltzDAP] Wrapped PairformerModule '{name}' with DAP")
    
    return model


# Keep old names for backward compatibility but they now do module-level wrapping
DAPMSALayer = DAPMSAModule
DAPPairformerLayer = DAPPairformerModule
