# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in MiniMax-H3 RMSNorm/AdaLN fusion using the shared Comfy Kitchen op.

The narrow block forward follows ComfyUI's comfy.ldm.minimax.model.DiTBlock;
ComfyUI's upstream GNU GPLv3 license and notices are retained separately.
Affine folding follows native moss_image modulation.
"""

from contextlib import nullcontext

import torch


def _fused_segments(x, weight, shift, scale, segments, eps, rms_adaln):
    """Fold affine RMSNorm weight into AdaLN scale, without a full token table.

    This intentionally changes low-precision rounding. Multiple segments need
    an output allocation plus one temporary segment; there is no claim of a
    speed/peak improvement until the target GPU and layout are benchmarked.
    """
    out = None
    cursor = 0
    for a, b, row in segments:
        if a != cursor or b < a or b > x.shape[0]:
            raise ValueError("Modulation segments must cover the token stream contiguously")
        cursor = b
        if a == b:
            continue
        selected_scale = scale[row].to(x.dtype)
        selected_shift = shift[row].to(x.dtype)
        if isinstance(row, torch.Tensor) and row.ndim != 0 and row.shape != (b - a,):
            raise ValueError("Per-token modulation row indices must match the segment length")
        folded_scale = selected_scale if weight is None else weight * (1.0 + selected_scale) - 1.0
        part = rms_adaln(x[a:b], folded_scale, selected_shift, eps)
        if a == 0 and b == x.shape[0] and len(segments) == 1:
            return part
        if out is None:
            out = torch.empty_like(x)
        out[a:b].copy_(part)
        del part
    if cursor != x.shape[0]:
        raise ValueError("Modulation segments do not cover the token stream")
    return torch.empty_like(x) if out is None else out


def fused_norm_modulation(x, norm, shift, scale, segments):
    """Use native casting/offload lifetime, epsilon, and row-selection semantics."""
    import comfy.model_management
    import comfy.ops
    import comfy.quant_ops
    from comfy.ldm.minimax.model import _mod_scale_shift

    if torch.is_grad_enabled() or comfy.model_management.in_training:
        return _mod_scale_shift(norm(x), shift, scale, segments)
    cast_weights = norm.comfy_cast_weights or bool(norm.weight_function) or bool(norm.bias_function)
    if not cast_weights and norm.weight is not None and norm.weight.dtype != x.dtype:
        # Native RMSNorm can promote through its affine weight. The shared fused
        # op is an input-dtype inference op, so leave this uncommon path native.
        return _mod_scale_shift(norm(x), shift, scale, segments)
    comfy.ops.run_every_op()
    context = norm.cast_weight(x) if cast_weights else nullcontext((norm.weight, None))
    with context as (weight, _):
        if weight is not None and weight.dtype != x.dtype:
            # A weight_function is allowed to promote the cast weight as well.
            h = torch.nn.functional.rms_norm(x, norm.normalized_shape, weight, norm.eps)
            return _mod_scale_shift(h, shift, scale, segments)
        eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
        return _fused_segments(x, weight, shift, scale, segments, eps, comfy.quant_ops.ck.rms_adaln)


class ModulationPolicy:
    def __init__(self):
        self.saved = []
        self.root = None

    def inject(self, patcher):
        import comfy.ops
        import comfy.quant_ops
        from comfy.ldm.minimax.model import DiTBlock, MiniMaxH3Model

        root = patcher.get_model_object("diffusion_model")
        if type(root) is not MiniMaxH3Model:
            raise TypeError("Fused modulation supports MiniMaxH3Model only")
        if self.saved:
            if self.root is not root:
                raise RuntimeError("Eject fused modulation before changing model")
            return
        if not comfy.quant_ops._CK_AVAILABLE:
            raise RuntimeError("Comfy Kitchen is required for fused modulation")
        blocks = list(root.blocks)
        if not blocks:
            raise RuntimeError("No MiniMax-H3 DiT blocks found")
        for block in blocks:
            if type(block) is not DiTBlock or getattr(block.forward, "__func__", None) is not DiTBlock.forward:
                raise RuntimeError("DiT block already wrapped; do not stack fused modulation")
            for norm in (block.norm1, block.norm2):
                if (type(norm) not in (comfy.ops.disable_weight_init.RMSNorm, comfy.ops.manual_cast.RMSNorm)
                        or len(norm.normalized_shape) != 1 or "forward" in norm.__dict__
                        or norm.comfy_force_forward or norm._forward_hooks or norm._forward_pre_hooks
                        or norm._backward_hooks):
                    raise RuntimeError("Fused modulation requires an unwrapped native RMSNorm")
        self.root = root
        try:
            for block in blocks:
                wrapped = self._wrap(block)
                self.saved.append((block, "forward" in block.__dict__, block.__dict__.get("forward"), wrapped))
                block.forward = wrapped
        except Exception:
            self._restore()
            raise

    @staticmethod
    def _wrap(block):
        from comfy.ldm.minimax.model import _mod_gate

        def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
            attention = block.attn if attention is None else attention
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
            h = fused_norm_modulation(x, block.norm1, shift_msa, scale_msa, mod_segments)
            x = _mod_gate(x, gate_msa, attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
            h = fused_norm_modulation(x, block.norm2, shift_mlp, scale_mlp, mod_segments)
            return _mod_gate(x, gate_mlp, block.mlp(h), mod_segments)
        return forward

    def _restore(self):
        for block, had_forward, previous, _ in reversed(self.saved):
            if had_forward:
                block.forward = previous
            elif "forward" in block.__dict__:
                del block.forward
        self.saved.clear()
        self.root = None

    def eject(self, patcher):
        for block, _, _, wrapped in self.saved:
            if block.forward is not wrapped:
                raise RuntimeError("DiT block changed after injection; eject later patch first")
        self._restore()


def enable_fused_modulation(model):
    """Return an opt-in ModelPatcher clone; no default behavior is changed."""
    from comfy.patcher_extension import PatcherInjection

    key = "pika_h3_4090_modulation"
    if key in model.injections:
        raise ValueError("Fused modulation is already attached")
    policy = ModulationPolicy()
    result = model.clone()
    result.set_injections(key, [PatcherInjection(policy.inject, policy.eject)])
    return result
