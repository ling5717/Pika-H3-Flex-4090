# SPDX-License-Identifier: GPL-3.0-only
"""Execution-scoped MiniMax-H3 FFN chunking; no attention or weight changes."""


def chunked_call(forward, x, chunk_size):
    if x.ndim != 2 or x.shape[0] <= chunk_size:
        return forward(x)
    first = forward(x[:chunk_size])
    out = first.new_empty((x.shape[0], first.shape[1]))
    out[:chunk_size].copy_(first)
    del first
    for start in range(chunk_size, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        out[start:stop].copy_(forward(x[start:stop]))
    return out


class FFNPolicy:
    def __init__(self, chunk_size=16384, activation="native"):
        if chunk_size not in (8192, 16384, 32768):
            raise ValueError("FFN chunk size must be 8192, 16384, or 32768")
        if activation not in ("native", "triton", "compiled"):
            raise ValueError("Activation must be native, triton, or compiled")
        self.chunk_size = chunk_size
        self.activation_mode = activation
        self.saved = []
        self.root = None
        self.activation = None

    def inject(self, patcher):
        from comfy.ldm.minimax.model import MiniMaxH3Model, MLP
        root = patcher.get_model_object("diffusion_model")
        if type(root) is not MiniMaxH3Model:
            raise TypeError("PikaH3Optimize supports MiniMaxH3Model only")
        if self.saved:
            if self.root is not root:
                raise RuntimeError("Eject the FFN policy before changing model")
            return
        modules = [m for m in root.modules() if type(m) is MLP]
        if not modules:
            raise RuntimeError("No MiniMax-H3 FFN modules found")
        for module in modules:
            if getattr(module.forward, "__func__", None) is not MLP.forward:
                raise RuntimeError("FFN already wrapped; do not stack FFN optimization nodes")
        if self.activation_mode == "triton":
            from .activation import SwiGLUActivation
            self.activation = SwiGLUActivation(mode="triton")
        elif self.activation_mode == "compiled":
            from .compilation import TensorSubgraph
            from .activation import swiglu_eager
            self.activation = TensorSubgraph(swiglu_eager, enabled=True, strict=True)
        self.root = root
        for module in modules:
            original = module.forward
            had_forward = "forward" in module.__dict__
            previous = module.__dict__.get("forward")
            forward = self._wrap(module, original)
            self.saved.append((module, had_forward, previous, forward))
            module.forward = forward

    def _wrap(self, module, original):
        def forward(x):
            if self.activation is None:
                inner = original
            else:
                def inner(tokens):
                    return module.fc2(self.activation(module.fc1(tokens)))
            return chunked_call(inner, x, self.chunk_size)
        return forward

    def eject(self, patcher):
        for module, _, _, forward in self.saved:
            if module.forward is not forward:
                raise RuntimeError("FFN changed after injection; eject later patch first")
        for module, had_forward, previous, _ in reversed(self.saved):
            if had_forward:
                module.forward = previous
            else:
                del module.forward
        self.saved.clear()
        self.root = None
        if self.activation is not None:
            self.activation.clear()
            self.activation = None


def optimize_model(model, chunk_size=16384, activation="native"):
    from comfy.patcher_extension import PatcherInjection
    result = model.clone()
    key = "pika_h3_4090_ffn"
    result.remove_injections(key)
    policy = FFNPolicy(chunk_size, activation)
    result.set_injections(key, [PatcherInjection(policy.inject, policy.eject)])
    return result
