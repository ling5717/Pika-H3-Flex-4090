# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in, model-scoped SwiGLU; never changes comfy.ops or linear modules.

The Triton candidate preserves the SiLU -> half cast -> multiply boundary, not
bitwise transcendental equivalence. GPU acceptance is required before promotion.
"""
from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

_LOG = logging.getLogger(__name__)


def swiglu_eager(x: torch.Tensor) -> torch.Tensor:
    """Native Comfy gate-first convention, with an independently owned float h."""
    if x.ndim == 0 or x.shape[-1] % 2:
        raise ValueError("SwiGLU requires an even last dimension")
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


# Lazy import: importing this extension on CPU must not require Triton or CUDA.
def _load_kernel():
    global tl  # Triton resolves kernel symbols from this module's globals.
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(X, Y, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
        # Cast before multiplying: raw [gate|up] offsets may exceed 2**31.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        row, col = i // WIDTH, i % WIDTH
        gate = tl.load(X + row * (2 * WIDTH) + col, i < N, 0).to(tl.float32)
        up = tl.load(X + row * (2 * WIDTH) + WIDTH + col, i < N, 0).to(tl.float32)
        # Explicit low-precision rounding is essential: do not fuse silu*up in FP32.
        half_silu = (gate / (1.0 + tl.exp(-gate))).to(Y.dtype.element_ty)
        result = half_silu.to(tl.float32) * up
        tl.store(Y + i, result, i < N)

    return kernel


class SwiGLUActivation:
    """Callable activation(input), modes eager/auto/triton (default eager).

    auto falls back only on missing Triton imports. Compilation/launch errors
    propagate instead of retrying a possibly damaged CUDA context. Strict triton
    rejects unsupported CPU/layout/dtype/autograd; auto falls back to eager.
    No input/output tensors or device allocations are cached by this object.
    """
    def __init__(self, mode: str = "eager"):
        if mode not in {"eager", "auto", "triton"}:
            raise ValueError("mode must be eager, auto, or triton")
        self.mode = mode
        self._kernel = None
        self._failed = False
        self.last_backend = "eager"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0 or x.shape[-1] % 2:
            raise ValueError("SwiGLU requires an even last dimension")
        eligible = (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16)
                    and x.is_contiguous() and x.numel() > 0
                    and not (torch.is_grad_enabled() and x.requires_grad))
        if self.mode == "triton" and not eligible:
            raise ValueError("triton SwiGLU requires nonempty contiguous CUDA BF16/FP16 inference input")
        if self.mode == "eager" or not eligible or self._failed:
            self.last_backend = "eager"
            return swiglu_eager(x)
        try:
            if self._kernel is None:
                self._kernel = _load_kernel()
            out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
            n = out.numel()
            self._kernel[((n + 255) // 256,)](
                x, out, n, out.shape[-1], 256, enable_fp_fusion=False)
            self.last_backend = "triton"
            return out
        except Exception as exc:
            if self.mode == "triton" or not isinstance(exc, ImportError):
                raise
            self._failed = True
            self.last_backend = "eager"
            _LOG.warning("SwiGLU Triton unavailable; using native eager", exc_info=True)
            return swiglu_eager(x)

    def clear(self):
        self._kernel = None
        self._failed = False
        self.last_backend = "eager"
