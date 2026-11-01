# SPDX-License-Identifier: GPL-3.0-only
"""Optional compilation of caller-owned pure tensor functions, never models.

The owner (e.g. a ModelPatcher scoped FFN policy) must clear/eject or close this
wrapper. No process-global Dynamo reset, model attributes, input tensors,
weights, quantization buffers or outputs are cached here. torch.compile's own
compiler caches are PyTorch-owned and are not claimed to be bounded by us.
"""
from __future__ import annotations

from collections import OrderedDict
import inspect
import logging

import torch

_LOG = logging.getLogger(__name__)
_FAILED = object()


def _signature(args):
    # Metadata only: no references to the supplied tensors survive a call.
    return tuple((tuple(x.shape), tuple(x.stride()), x.dtype, x.device.type,
                  x.device.index, x.requires_grad) for x in args) + (
                      torch.is_grad_enabled(), torch.is_inference_mode_enabled(),
                      torch.is_autocast_enabled("cuda"), torch.is_autocast_enabled("cpu"),
                      torch.get_autocast_dtype("cuda"), torch.get_autocast_dtype("cpu"))


class TensorSubgraph:
    """Compile fn(*tensors), opt-in and bounded at the wrapper level.

    fn must be a pure free tensor function: not nn.Module, a bound method, or
    a closure over model/weight state. This deliberately excludes Comfy module
    forward, cast contexts, load/offload, and patcher Python. Arbitrary callable
    purity cannot be proven; the owner remains responsible for that contract.

    strict=True propagates failures; strict=False permits safe CPU fallback.
    Accelerator execution failures always propagate to avoid a damaged context.
    No whole-model compile,
    disable_dynamic=True, CUDA probing, or implicit dependency installation.
    """
    def __init__(self, fn, *, enabled=False, max_entries=4, strict=True,
                 backend="inductor"):
        if not inspect.isfunction(fn) or inspect.ismethod(fn):
            raise TypeError("fn must be a free pure tensor function, not a module/method")
        captured = inspect.getclosurevars(fn)
        if captured.nonlocals:
            raise TypeError("fn must not close over model or tensor state")
        retained = tuple(captured.globals.values()) + tuple(fn.__defaults__ or ()) + tuple((fn.__kwdefaults__ or {}).values())
        if any(isinstance(value, (torch.Tensor, torch.nn.Module)) for value in retained):
            raise TypeError("fn must not retain global/default model or tensor state")
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        if backend != "inductor":
            raise ValueError("only backend='inductor' is supported")
        self.fn = fn
        self.enabled = bool(enabled)
        self.strict = bool(strict)
        self.max_entries = max_entries
        self.backend = backend
        self._cache = OrderedDict()
        self._closed = False
        self.last_backend = "eager"

    @property
    def cache_size(self):
        return len(self._cache)

    def __call__(self, *args):
        if self._closed:
            raise RuntimeError("tensor subgraph is closed")
        if not args or any(type(x) is not torch.Tensor for x in args):
            raise TypeError("tensor subgraph accepts only plain tensor positional arguments")
        if not self.enabled:
            self.last_backend = "eager"
            return self.fn(*args)
        key = _signature(args)
        compiled = self._cache.pop(key, None)
        if compiled is None:
            # Evict BEFORE compilation: even a failed new signature cannot
            # grow the local cache beyond the lifecycle owner's budget.
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)
            try:
                compiled = torch.compile(self.fn, backend=self.backend,
                                         fullgraph=True, dynamic=True)
            except Exception:
                if self.strict:
                    raise
                _LOG.warning("Tensor compile failed; eager fallback", exc_info=True)
                compiled = _FAILED
        self._cache[key] = compiled
        if compiled is _FAILED:
            self.last_backend = "eager"
            return self.fn(*args)
        try:
            out = compiled(*args)
        except Exception:
            if self.strict or any(x.device.type != "cpu" for x in args):
                # Do not retry a failed accelerator operation in the same context.
                self._cache.pop(key, None)
                raise
            _LOG.warning("Compiled tensor execution failed; eager fallback", exc_info=True)
            self._cache[key] = _FAILED
            self.last_backend = "eager"
            return self.fn(*args)
        self.last_backend = "inductor"
        return out

    def clear(self):
        """Release our compiled callable references; safe to reuse after eject."""
        self._cache.clear()
        self.last_backend = "eager"

    def close(self):
        """Permanent teardown, including any reference held to fn."""
        self.clear()
        self.fn = None
        self._closed = True
