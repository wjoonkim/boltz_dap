"""
DAP Trunk Wrapper for Boltz 2.

This module monkey-patches the Boltz 2 model to use DAP-aware layers
in the trunk loop. The approach:
1. ALL GPUs: Run input embedding, z_init, recycling (small ops)
2. ALL GPUs: Scatter z, run template/MSA/pairformer with DAP layers
3. ALL GPUs: Gather z back to full
4. GPU 0 ONLY: Run post-trunk (distogram, diffusion, structure, confidence)
   Non-primary GPUs skip post-trunk entirely — those modules
   aren't even loaded on their GPUs.

No model duplication — non-primary GPUs only have trunk weights.
Activations (z) are sharded across GPUs during the trunk.
"""

import torch
from torch import Tensor, nn
from typing import Optional
import time
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from boltz_distributed.comm import scatter, gather
from boltz_distributed.core import get_dap_size, get_dap_rank

from dap_msa import DAPMSALayer
from dap_pairformer import DAPPairformerLayer
from dap_pairformer_noseq import DAPPairformerNoSeqLayer
from dap_confidence import inject_dap_into_confidence, run_confidence_dap


def inject_dap_into_model(model):
    """Inject DAP wrappers into a Boltz 2 model in-place.

    Replaces:
    - msa_module.layers[i] → DAPMSALayer (wraps MSALayer)
    - pairformer_module.layers[i] → DAPPairformerLayer (wraps PairformerLayer)
    - template_module.pairformer.layers[i] → DAPPairformerNoSeqLayer (wraps PairformerNoSeqLayer)

    Returns the model with a modified forward function.
    """
    dap_rank = get_dap_rank()
    dap_size = get_dap_size()

    if dap_size <= 1:
        print("[DAP] DAP size <= 1, no wrapping needed")
        return model

    # 1. Wrap MSA module layers
    if hasattr(model, 'msa_module'):
        msa = model.msa_module
        if hasattr(msa, '_orig_mod'):
            msa = msa._orig_mod
        for i in range(len(msa.layers)):
            original_layer = msa.layers[i]
            msa.layers[i] = DAPMSALayer(original_layer)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped {len(msa.layers)} MSA layers")

    # 2. Wrap main pairformer layers
    if hasattr(model, 'pairformer_module'):
        pf = model.pairformer_module
        if hasattr(pf, '_orig_mod'):
            pf = pf._orig_mod
        for i in range(len(pf.layers)):
            original_layer = pf.layers[i]
            pf.layers[i] = DAPPairformerLayer(original_layer)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped {len(pf.layers)} pairformer layers")

    # 3. Wrap template module's pairformer layers with DAP
    if hasattr(model, 'template_module') and model.use_templates:
        tmpl = model.template_module
        if hasattr(tmpl, '_orig_mod'):
            tmpl = tmpl._orig_mod
        pf_noseq = tmpl.pairformer
        for i in range(len(pf_noseq.layers)):
            original_layer = pf_noseq.layers[i]
            pf_noseq.layers[i] = DAPPairformerNoSeqLayer(original_layer)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped {len(pf_noseq.layers)} template pairformer layers")

    # 4. Wrap confidence module's pairformer layers
    if hasattr(model, 'confidence_module') and model.confidence_prediction:
        inject_dap_into_confidence(model.confidence_module)
        if dap_rank == 0:
            print(f"  [DAP] Wrapped confidence pairformer layers")

    # 5. Store original forward and install DAP-aware forward
    model._original_forward = model.forward
    model.forward = _make_dap_forward(model)

    if dap_rank == 0:
        print(f"  [DAP] Installed DAP forward pass (scatter z before trunk, gather after)")

    return model


def _make_dap_forward(model):
    """Create a DAP-aware forward function that wraps the original.

    Key modifications:
    - After z_init is computed, scatter it across GPUs
    - After the trunk loop (template + MSA + pairformer), gather z back
    - Everything else (distogram, diffusion, structure, confidence) uses full z
    """
    original_forward = model._original_forward

    def dap_forward(
        feats,
        recycling_steps=0,
        num_sampling_steps=None,
        multiplicity_diffusion_train=1,
        diffusion_samples=1,
        max_parallel_samples=None,
        run_confidence_sequentially=False,
    ):
        dap_size = get_dap_size()
        dap_rank = get_dap_rank()
        _t0 = time.time()

        # ── Peak tracker: records every checkpoint and detects peak changes ──
        _timeline = []  # list of (elapsed, alloc, peak, label)
        _peak_changes = []  # list of (label, old_peak, new_peak) — when peak increases

        def _mem_log(label):
            """Log memory checkpoint and detect if this checkpoint set a new peak."""
            if dap_rank != 0:
                return
            elapsed = time.time() - _t0
            alloc = torch.cuda.memory_allocated() // 1024**2
            peak = torch.cuda.max_memory_allocated() // 1024**2
            prev_peak = _timeline[-1][2] if _timeline else 0
            _timeline.append((elapsed, alloc, peak, label))
            marker = ""
            if peak > prev_peak:
                _peak_changes.append((label, prev_peak, peak))
                marker = f"  ◀◀ NEW PEAK (+{peak - prev_peak}MB)"
            print(f"    [TIMELINE] {elapsed:7.1f}s | alloc={alloc:>6d}MB | peak={peak:>6d}MB | {label}{marker}")

        def _print_peak_summary():
            """Print definitive summary of where the peak was set."""
            if dap_rank != 0 or not _timeline:
                return
            final_peak = _timeline[-1][2]
            print(f"\n{'='*72}")
            print(f"  PEAK SUMMARY  (final peak = {final_peak} MB)")
            print(f"{'='*72}")
            if _peak_changes:
                for label, old, new in _peak_changes:
                    print(f"    {old:>6d}MB → {new:>6d}MB  (+{new-old:>5d}MB)  at: {label}")
                # The last peak change is the one that set the final peak
                winner = _peak_changes[-1]
                print(f"\n  ▶▶ DEFINITIVE PEAK SET BY: \"{winner[0]}\"")
                print(f"     {winner[1]}MB → {winner[2]}MB (+{winner[2]-winner[1]}MB)")
            else:
                print("    No peak changes detected (peak was 0 throughout)")
            print(f"{'='*72}\n")

        if dap_size <= 1:
            return original_forward(
                feats, recycling_steps, num_sampling_steps,
                multiplicity_diffusion_train, diffusion_samples,
                max_parallel_samples, run_confidence_sequentially,
            )
         # ── Input embedding ──────────────────────────────────────────────
        # s_inputs [B, N, token_s] is small, computed on all GPUs.
        # z_init [B, N, N, C] is large. We compute it on all GPUs because
        # scatter() is an all-to-all collective requiring all ranks to
        # participate. We immediately scatter and delete the full tensor
        # so the transient peak is brief.
        with torch.set_grad_enabled(
            model.training and model.structure_prediction_training
        ):
            if dap_rank == 0:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{'='*72}")
                print(f"  MEMORY TIMELINE  (run started {ts})")
                print(f"{'='*72}")
                torch.cuda.reset_peak_memory_stats()
            _mem_log("start")

            s_inputs = model.input_embedder(feats)
            s_init = model.s_init(s_inputs)
            _mem_log("after input_embedder + s_init")

            # Compute masks (small, all GPUs)
            mask = feats["token_pad_mask"].float()
            pair_mask = mask[:, :, None] * mask[:, None, :]

            # Compute z_init (transient full-size, immediately scattered)
            z_init_full = (
                model.z_init_1(s_inputs)[:, :, None]
                + model.z_init_2(s_inputs)[:, None, :]
            )
            relative_position_encoding = model.rel_pos(feats)
            z_init_full = z_init_full + relative_position_encoding
            z_init_full = z_init_full + model.token_bonds(feats["token_bonds"].float())
            if model.bond_type_feature:
                z_init_full = z_init_full + model.token_bonds_type(feats["type_bonds"].long())
            z_init_full = z_init_full + model.contact_conditioning(feats)
            _mem_log("after z_init_full (before scatter)")
            original_N = z_init_full.shape[1]

            # Track padded size (scatter will pad dim 1 for divisibility by dap_size,
            # and row_to_col/col_to_row will pad dim 2 during all-to-all)
            N_padded = ((original_N + dap_size - 1) // dap_size) * dap_size

            # SCATTER z_init immediately — no GPU holds full z after this
            z_init_scattered = scatter(z_init_full, dim=1)
            del z_init_full  # Free the full tensor right away
            _mem_log("after scatter z_init (full z freed)")

            # Initialize recycling tensors as SCATTERED
            s = torch.zeros_like(s_init)
            z_scattered = torch.zeros_like(z_init_scattered)
            pair_mask_scattered = scatter(pair_mask, dim=1)

            if model.run_trunk_and_structure:
                for i in range(recycling_steps + 1):
                    with torch.set_grad_enabled(
                        model.training
                        and model.structure_prediction_training
                        and (i == recycling_steps)
                    ):
                        if (
                            model.training
                            and (i == recycling_steps)
                            and torch.is_autocast_enabled()
                        ):
                            torch.clear_autocast_cache()

                        # Recycling: s is full (small), z stays scattered
                        s = s_init + model.s_recycle(model.s_norm(s))
                        z_scattered = z_init_scattered + model.z_recycle(
                            model.z_norm(z_scattered)
                        )

                        # Template module — all GPUs participate since its
                        # internal pairformer layers are DAP-wrapped (need
                        # all-to-all comms). Temporarily gathers z for the
                        # template's non-DAP parts, then scatters output.
                        if model.use_templates:
                            tmpl = model.template_module
                            if hasattr(tmpl, '_orig_mod') and not model.training:
                                tmpl = tmpl._orig_mod
                            z_scattered = _run_template_dap(
                                tmpl, z_scattered, feats, pair_mask,
                                model.use_kernels, original_N, _mem_log
                            )
                            _mem_log("after template_module")

                        # MSA module — DAP-wrapped layers expect scattered z
                        msa = model.msa_module
                        if hasattr(msa, '_orig_mod') and not model.training:
                            msa = msa._orig_mod

                        z_scattered = _run_msa_dap(
                            msa, z_scattered, s_inputs, feats,
                            pair_mask, model.use_kernels,
                            mem_log=_mem_log,
                            _msa_diag=(i == 0),
                        )
                        _mem_log("after msa_module")

                        # Pairformer module — DAP-wrapped layers
                        pf = model.pairformer_module
                        if hasattr(pf, '_orig_mod') and not model.training:
                            pf = pf._orig_mod

                        s, z_scattered = _run_pairformer_dap(
                            pf, s, z_scattered, mask, pair_mask,
                            model.use_kernels,
                            mem_log=_mem_log
                        )
                        _mem_log("after pairformer_module")

                # ── GATHER z back to full and TRIM to original_N ──
                z = gather(z_scattered.contiguous(), dim=1, original_size=N_padded)
                del z_scattered
                _mem_log("after gather z (full z restored)")
                # Trim padding back to original sequence length
                if N_padded != original_N:
                    z = z[:, :original_N, :original_N, :]

            # ── OFFLOAD TRUNK WEIGHTS TO CPU ──────────────────────────────
            # Trunk is done — free GPU memory for post-trunk modules.
            trunk_module_names = [
                "input_embedder", "s_init", "z_init_1", "z_init_2",
                "rel_pos", "token_bonds", "contact_conditioning",
                "s_recycle", "z_recycle", "s_norm", "z_norm",
                "msa_module", "pairformer_module", "template_module",
            ]
            if model.bond_type_feature:
                trunk_module_names.append("token_bonds_type")
            for name in trunk_module_names:
                if hasattr(model, name):
                    getattr(model, name).cpu()
            torch.cuda.empty_cache()
            _mem_log("after trunk offload to CPU")

            # ── Post-trunk ────────────────────────────────────────────────
            pdistogram = model.distogram_module(z)
            dict_out = {"pdistogram": pdistogram, "s": s, "z": z}

            # GPU 0: runs distogram, diffusion, structure (GPU 1 waits)
            # Both GPUs: participate in confidence pairformer DAP

            if dap_rank == 0:
                # Offload distogram after use
                model.distogram_module.cpu()
                torch.cuda.empty_cache()
                _mem_log("after distogram (offloaded)")

            if (
                model.run_trunk_and_structure
                and ((not model.training) or model.confidence_prediction)
                and (not model.skip_run_structure)
            ):
                if dap_rank == 0:
                    # ── Inlined diffusion_conditioning with chunked Transitions ──
                    _mem_log("before diffusion_conditioning (inlined)")
                    dc = model.diffusion_conditioning

                    # ① PairwiseConditioning — with chunked Transitions
                    _mem_log("  dc: before pairwise_conditioner")
                    pw = dc.pairwise_conditioner
                    z_cond = torch.cat((z, relative_position_encoding), dim=-1)
                    z_cond = pw.dim_pairwise_init_proj(z_cond)
                    del relative_position_encoding  # Free 1.6 GB early
                    torch.cuda.empty_cache()
                    _mem_log("  dc: after pairwise proj (rel_pos_enc freed)")

                    for t_idx, transition in enumerate(pw.transitions):
                        z_cond = transition(z_cond, chunk_size=128) + z_cond
                        _mem_log(f"  dc: after pairwise transition[{t_idx}]")

                    # ② AtomEncoder
                    _mem_log("  dc: before atom_encoder")
                    q, c, p, to_keys = dc.atom_encoder(
                        feats=feats, s_trunk=s, z=z_cond,
                    )
                    _mem_log("  dc: after atom_encoder")

                    # ③ Atom encoder/decoder biases (small projections of p)
                    atom_enc_bias = torch.cat([layer(p) for layer in dc.atom_enc_proj_z], dim=-1)
                    atom_dec_bias = torch.cat([layer(p) for layer in dc.atom_dec_proj_z], dim=-1)
                    del p  # Free atom-pair features
                    _mem_log("  dc: after atom biases (p freed)")

                    # ④ Token transformer biases — 24 projections of z_cond [B,N,N,128]→[B,N,N,8]
                    # Accumulate incrementally to avoid holding 24 copies
                    token_trans_bias = torch.cat(
                        [layer(z_cond) for layer in dc.token_trans_proj_z], dim=-1
                    )
                    _mem_log("  dc: after token_trans_bias")

                    # Free z_cond — no longer needed
                    del z_cond
                    torch.cuda.empty_cache()
                    _mem_log("  dc: z_cond freed")

                    # Offload diffusion_conditioning weights
                    dc.cpu()
                    torch.cuda.empty_cache()
                    _mem_log("after diffusion_conditioning (offloaded)")

                    diffusion_conditioning = {
                        "q": q, "c": c, "to_keys": to_keys,
                        "atom_enc_bias": atom_enc_bias,
                        "atom_dec_bias": atom_dec_bias,
                        "token_trans_bias": token_trans_bias,
                    }

                    # Can we free z before structure_module? It's still in dict_out
                    _z_mb = z.nelement() * z.element_size() / (1024**2)
                    print(f"    [DIAG] z tensor size: {_z_mb:.0f} MB, s size: {s.nelement() * s.element_size() / (1024**2):.0f} MB")
                    _mem_log("before structure_module.sample")

                    # Structure module (rank 0 only, then offload)
                    with torch.autocast("cuda", enabled=False):
                        structure_output = model.structure_module.sample(
                            s_trunk=s.float(),
                            s_inputs=s_inputs.float(),
                            feats=feats,
                            num_sampling_steps=num_sampling_steps,
                            atom_mask=feats["atom_pad_mask"].float(),
                            multiplicity=diffusion_samples,
                            max_parallel_samples=max_parallel_samples,
                            steering_args=getattr(model, 'steering_args', None),
                            diffusion_conditioning=diffusion_conditioning,
                        )
                        dict_out.update(structure_output)

                    _mem_log("after structure_module.sample")
                    model.structure_module.cpu()
                    del diffusion_conditioning
                    torch.cuda.empty_cache()
                    _mem_log("after structure_module (offloaded)")

                    if model.predict_bfactor:
                        dict_out["bfactors_logits"] = model.bfactor_module(s)
                        model.bfactor_module.cpu()
                        torch.cuda.empty_cache()

                # Sync before confidence DAP (GPU 1 was waiting)
                if dap_size > 1:
                    import torch.distributed as tdist
                    tdist.barrier()

                # Confidence module with DAP (ALL GPUs participate)
                if model.confidence_prediction:
                    if dap_size > 1:
                        # Grab conf-specific data before offloading dict_out
                        if dap_rank == 0:
                            conf_x_pred = dict_out.get("sample_atom_coords", feats["coords"])
                            # .contiguous() on [B,N,N] slice (9 MB) so full pdistogram
                            # [B,N,N,64] (590 MB) can be freed by CPU offload
                            conf_pdist = dict_out["pdistogram"][:, :, :, 0].contiguous()
                        else:
                            conf_x_pred = torch.empty(0)
                            conf_pdist = torch.empty(0)

                        # ── Offload ALL dict_out to CPU ──
                        # z/s/pdistogram/structure_outputs all go to CPU.
                        # The local vars z, s, s_inputs still hold GPU refs.
                        if dap_rank == 0:
                            for key in list(dict_out.keys()):
                                if isinstance(dict_out[key], torch.Tensor) and dict_out[key].is_cuda:
                                    dict_out[key] = dict_out[key].cpu()
                            torch.cuda.empty_cache()

                        _mem_log("before confidence (dict_out offloaded)")

                        # Pass z via mutable list — run_confidence_dap clears
                        # z_holder[0] after scatter, breaking our last GPU ref
                        # to the full z (~1.2 GB freed mid-confidence).
                        z_holder = [z]
                        del z  # our local ref gone; z_holder[0] is the last one
                        confidence_output = run_confidence_dap(
                            model,
                            s_inputs=s_inputs,
                            s=s,
                            z_holder=z_holder,
                            x_pred=conf_x_pred,
                            feats=feats,
                            pred_distogram_logits=conf_pdist,
                            multiplicity=diffusion_samples,
                            run_sequentially=run_confidence_sequentially,
                            use_kernels=model.use_kernels,
                        )
                        # Release remaining local GPU refs
                        del s, s_inputs, conf_x_pred, conf_pdist, z_holder

                        _mem_log("after confidence_module")
                    else:
                        confidence_output = model.confidence_module(
                            s_inputs=s_inputs.detach(),
                            s=s.detach(),
                            z=z.detach(),
                            x_pred=(
                                dict_out["sample_atom_coords"].detach()
                                if not model.skip_run_structure
                                else feats["coords"].repeat_interleave(diffusion_samples, 0)
                            ),
                            feats=feats,
                            pred_distogram_logits=dict_out["pdistogram"][:, :, :, 0].detach(),
                            multiplicity=diffusion_samples,
                            run_sequentially=run_confidence_sequentially,
                            use_kernels=model.use_kernels,
                        )
                    if dap_rank == 0:
                        dict_out.update(confidence_output)
                        # Move any CPU-offloaded tensors back to GPU for writer
                        for key in list(dict_out.keys()):
                            if isinstance(dict_out[key], torch.Tensor) and not dict_out[key].is_cuda:
                                dict_out[key] = dict_out[key].cuda(0)
                    model.confidence_module.cpu()
                    torch.cuda.empty_cache()

        _print_peak_summary()
        return dict_out

    return dap_forward


def _run_template_dap(tmpl_module, z_scattered, feats, pair_mask, use_kernels, original_N, mem_log=None):
    """Run template module with DAP-scattered z.

    Instead of gathering z → running template → scattering output,
    we keep z scattered, compute pre-PF features on full N
    (small template coords), scatter a_tij, merge with z_proj(z_scattered),
    run DAP-wrapped PF layers, and return scattered output.

    z_scattered: [B, N/dap, N_padded, D] — row-scattered
    Returns: z_scattered + template_output (still row-scattered)
    """
    from boltz.data import const
    from torch.nn.functional import one_hot

    dap_size = get_dap_size()

    # Load template features (all on full N — these are small)
    asym_id = feats["asym_id"]                     # [B, N]
    res_type = feats["template_restype"]            # [B, T, N, C]
    frame_rot = feats["template_frame_rot"]         # [B, T, N, 3, 3]
    frame_t = feats["template_frame_t"]             # [B, T, N, 3]
    frame_mask = feats["template_mask_frame"]       # [B, T, N]
    cb_coords = feats["template_cb"]                # [B, T, N, 3]
    ca_coords = feats["template_ca"]                # [B, T, N, 3]
    cb_mask = feats["template_mask_cb"]             # [B, T, N]
    template_mask = feats["template_mask"].any(dim=2).float()  # [B, T]
    num_templates = template_mask.sum(dim=1).clamp(min=1)      # [B]

    B, T = res_type.shape[:2]
    N = original_N

    # Compute pairwise masks [B, T, N, N, 1]
    b_cb_mask = (cb_mask[:, :, :, None] * cb_mask[:, :, None, :])[..., None]
    b_frame_mask = (frame_mask[:, :, :, None] * frame_mask[:, :, None, :])[..., None]

    # Asym mask [B, T, N, N]
    asym_mask = (asym_id[:, :, None] == asym_id[:, None, :]).float()
    asym_mask = asym_mask[:, None].expand(-1, T, -1, -1)

    # Compute template features on full N (these are from coords, small)
    with torch.autocast(device_type="cuda", enabled=False):
        # Distogram [B, T, N, N, num_bins]
        cb_dists = torch.cdist(cb_coords, cb_coords)
        boundaries = torch.linspace(tmpl_module.min_dist, tmpl_module.max_dist,
                                     tmpl_module.num_bins - 1).to(cb_dists.device)
        distogram = (cb_dists[..., None] > boundaries).sum(dim=-1).long()
        distogram = one_hot(distogram, num_classes=tmpl_module.num_bins)

        # Unit vector [B, T, N, N, 3]
        frame_rot_t = frame_rot.unsqueeze(2).transpose(-1, -2)
        frame_t_exp = frame_t.unsqueeze(2).unsqueeze(-1)
        ca_exp = ca_coords.unsqueeze(3).unsqueeze(-1)
        vector = torch.matmul(frame_rot_t, (ca_exp - frame_t_exp))
        norm = torch.norm(vector, dim=-1, keepdim=True)
        unit_vector = torch.where(norm > 0, vector / norm, torch.zeros_like(vector))
        unit_vector = unit_vector.squeeze(-1)

        # Concatenate and project: a_tij [B, T, N, N, template_dim]
        a_tij = torch.cat([distogram, b_cb_mask, unit_vector, b_frame_mask], dim=-1)
        a_tij = a_tij * asym_mask.unsqueeze(-1)

        res_type_i = res_type[:, :, :, None].expand(-1, -1, -1, N, -1)
        res_type_j = res_type[:, :, None, :].expand(-1, -1, N, -1, -1)
        a_tij = torch.cat([a_tij, res_type_i, res_type_j], dim=-1)
        a_tij = tmpl_module.a_proj(a_tij)  # [B, T, N, N, template_dim]

    if mem_log:
        mem_log("  template: pre-PF features computed (full N)")

    # Scatter a_tij on dim 2 (the first N = rows) to match z_scattered
    # a_tij: [B, T, N, N, D] → scatter on dim 2 → [B, T, N/dap, N, D]
    if dap_size > 1:
        # Reshape to [B*T, N, N, D] for scatter, then back
        a_shape = a_tij.shape
        a_tij = a_tij.view(B * T, N, N, -1)
        a_tij = scatter(a_tij, dim=1)  # [B*T, N/dap, N, D]
        a_tij = a_tij.view(B, T, a_tij.shape[1], a_tij.shape[2], -1)

    # z_scattered is [B, N/dap, N_padded, D]. Trim to original_N for template ops
    # and project: z_proj(z_norm(z_scattered[:,None])) → [B, 1, N/dap, N, template_dim]
    z_for_tmpl = z_scattered[:, :, :original_N, :]  # [B, N/dap, N, D]
    v = tmpl_module.z_proj(tmpl_module.z_norm(z_for_tmpl[:, None])) + a_tij
    del a_tij

    # Prepare pair_mask for scattered template PF
    # pair_mask is [B, N, N]. We need it scattered: [B, N/dap, N]
    pair_mask_tmpl = pair_mask[:, :, :original_N]  # ensure original_N
    if dap_size > 1:
        pair_mask_tmpl = scatter(pair_mask_tmpl, dim=1)  # [B, N/dap, N]

    # Expand mask for T templates: [B*T, N/dap, N]
    pair_mask_tmpl = pair_mask_tmpl[:, None].expand(-1, T, -1, -1)
    pair_mask_tmpl = pair_mask_tmpl.reshape(B * T, *pair_mask_tmpl.shape[2:])

    # v: [B, T, N/dap, N, template_dim] → [B*T, N/dap, N, template_dim]
    v = v.view(B * T, *v.shape[2:])

    if mem_log:
        mem_log("  template: v scattered, entering PF")

    # Run DAP-wrapped pairformer (2 layers)
    # Set chunk size
    if not tmpl_module.pairformer.training:
        if original_N > const.chunk_size_threshold:
            chunk_size_tri_attn = 128
        else:
            chunk_size_tri_attn = 512
    else:
        chunk_size_tri_attn = None

    pf_input = v
    for i, layer in enumerate(tmpl_module.pairformer.layers):
        pf_input = layer(
            pf_input, pair_mask_tmpl,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )
        if mem_log:
            mem_log(f"  template: PF layer[{i}]")

    # v = v + pf_output (residual)
    v = v + pf_input
    del pf_input

    # Post-PF: norm, reshape, aggregate over templates
    v = tmpl_module.v_norm(v)
    v = v.view(B, T, *v.shape[1:])  # [B, T, N/dap, N, template_dim]

    # Aggregate templates
    tmask = template_mask[:, :, None, None, None]
    ntemplates = num_templates[:, None, None, None]
    u = (v * tmask).sum(dim=1) / ntemplates.to(v)  # [B, N/dap, N, template_dim]
    del v

    # Project back to z dim
    u = tmpl_module.u_proj(tmpl_module.relu(u))  # [B, N/dap, N, token_z]

    # Pad dim 2 to match z_scattered's padded N if needed
    if u.shape[2] < z_scattered.shape[2]:
        pad_n = z_scattered.shape[2] - u.shape[2]
        u = torch.nn.functional.pad(u, (0, 0, 0, pad_n))

    # Add to z_scattered
    z_scattered = z_scattered + u
    del u

    return z_scattered


def _run_msa_dap(msa_module, z_scattered, s_inputs, feats, full_pair_mask, use_kernels, mem_log=None, _msa_diag=False):
    """Run MSA module with DAP-scattered z.

    z_scattered: [B, N/dap, N, D]
    """
    # Set chunk sizes (same logic as original)
    N = z_scattered.shape[2]  # full N
    if not msa_module.training:
        from boltz.data import const
        if N > const.chunk_size_threshold:
            chunk_heads_pwa = True
            chunk_size_transition_z = 64
            chunk_size_transition_msa = 32
            chunk_size_outer_product = 4
            chunk_size_tri_attn = 128
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = 512
    else:
        chunk_heads_pwa = False
        chunk_size_transition_z = None
        chunk_size_transition_msa = None
        chunk_size_outer_product = None
        chunk_size_tri_attn = None

    # Prepare MSA features
    from boltz.data import const
    msa = feats["msa"]
    msa = torch.nn.functional.one_hot(msa, num_classes=const.num_tokens)
    has_deletion = feats["has_deletion"].unsqueeze(-1)
    deletion_value = feats["deletion_value"].unsqueeze(-1)
    is_paired = feats["msa_paired"].unsqueeze(-1)
    msa_mask = feats["msa_mask"]
    token_mask = feats["token_pad_mask"].float()
    token_mask_2d = token_mask[:, :, None] * token_mask[:, None, :]

    if msa_module.use_paired_feature:
        m = torch.cat([msa, has_deletion, deletion_value, is_paired], dim=-1)
    else:
        m = torch.cat([msa, has_deletion, deletion_value], dim=-1)

    if msa_module.subsample_msa:
        msa_indices = torch.randperm(msa.shape[1])[:msa_module.num_subsampled_msa]
        m = m[:, msa_indices]
        msa_mask = msa_mask[:, msa_indices]

    m = msa_module.msa_proj(m)
    m = m + msa_module.s_proj(s_inputs).unsqueeze(1)

    # Run MSA blocks with DAP layers
    for i in range(msa_module.msa_blocks):
        # Enable fine-grained diagnostics for layers 0,1 on first trunk cycle
        layer = msa_module.layers[i]
        layer._diag_enabled = _msa_diag and (i <= 1)
        z_scattered, m = layer(
            z_scattered, m, token_mask_2d, msa_mask,
            chunk_heads_pwa,
            chunk_size_transition_z,
            chunk_size_transition_msa,
            chunk_size_outer_product,
            chunk_size_tri_attn,
            use_kernels,
        )
        if mem_log:
            mem_log(f"  msa_module.layer[{i}]")

    return z_scattered


def _run_pairformer_dap(pf_module, s, z_scattered, mask, full_pair_mask, use_kernels, mem_log=None):
    """Run pairformer module with DAP-scattered z.

    z_scattered: [B, N/dap, N, D]
    s: [B, N, 384] — replicated
    """
    dap_size = get_dap_size()
    pair_mask_scattered = scatter(full_pair_mask, dim=1) if dap_size > 1 else full_pair_mask

    # Set chunk sizes for large N (same logic as MSA module)
    N = z_scattered.shape[2]
    if not pf_module.training:
        from boltz.data import const
        if N > const.chunk_size_threshold:
            chunk_size_tri_attn = 128
        else:
            chunk_size_tri_attn = 512
    else:
        chunk_size_tri_attn = None

    for i, layer in enumerate(pf_module.layers):
        s, z_scattered = layer(
            s, z_scattered, mask, pair_mask_scattered,
            chunk_size_tri_attn=chunk_size_tri_attn,
            use_kernels=use_kernels,
        )
        # Log every 8th layer to avoid spam (48 layers total)
        if mem_log and (i % 8 == 0 or i == len(pf_module.layers) - 1):
            mem_log(f"  pairformer.layer[{i}]")

    return s, z_scattered
