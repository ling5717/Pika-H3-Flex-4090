# SPDX-License-Identifier: GPL-3.0-only
"""Low-peak ordinary linear LoRA using ComfyUI's mapping and injection lifecycle.

ComfyUI integration follows comfy.sd.load_bypass_lora_for_models and the native
BypassForwardHook API. ComfyUI's upstream GNU GPLv3 license and notices
are retained separately and are not replaced by this extension.
"""

import math

import torch
import torch.nn.functional as F


def plain_lora_forward(x, base_out, up, down, alpha=None, strength=1.0, mode="low_peak"):
    """Match native linear LoRA, reusing only the freshly allocated LoRA output.

    Never mutate base_out: even a no-grad base forward can return x or a view
    retained elsewhere. With autograd enabled low_peak uses native out-of-place
    math; experimental addmm is differentiable but changes rounding.
    """
    if mode not in ("low_peak", "addmm"):
        raise ValueError(f"Unknown LoRA mode: {mode}")
    scale = (alpha / down.shape[0] if alpha is not None else 1.0) * strength
    hidden = F.linear(x, down.to(dtype=x.dtype))
    up = up.to(dtype=x.dtype)
    if mode == "addmm" and base_out.dtype == x.dtype:
        # Experimental: the second GEMM and residual share one final rounding,
        # unlike native BF16/FP16 h() followed by a separate multiply and add.
        return torch.addmm(base_out.reshape(-1, up.shape[0]),
                           hidden.reshape(-1, down.shape[0]), up.t(),
                           alpha=scale).reshape(base_out.shape)
    out = F.linear(hidden, up)
    if torch.is_grad_enabled() or out.dtype != base_out.dtype:
        return base_out + out * scale
    if scale != 1.0:
        out.mul_(scale)
    return out.add_(base_out)


def _validate_adapter(key, adapter, adapter_type):
    if type(adapter) is not adapter_type:
        raise ValueError(f"Unsupported adapter for {key}: {type(adapter).__name__}")
    up, down, alpha, mid, dora, reshape = adapter.weights
    if mid is not None or dora is not None or reshape is not None:
        raise ValueError(f"Only plain LoRA without mid/DoRA/reshape is supported: {key}")
    if up.ndim != 2 or down.ndim != 2:
        raise ValueError(f"Only 2D linear LoRA is supported: {key}")
    if down.shape[0] == 0 or up.shape[1] != down.shape[0]:
        raise ValueError(f"Invalid LoRA rank: {key}")
    if alpha is not None and not math.isfinite(float(alpha)):
        raise ValueError(f"Nonfinite LoRA alpha: {key}")


def load_low_peak_lora(model, lora_state, strength=1.0, *, mode="low_peak"):
    """Return a cloned ModelPatcher with reversible low-peak LoRA injections.

    Only ordinary linear inference LoRA is accepted. Unknown state keys,
    unsupported adapters and incomplete mappings fail before attaching hooks.
    mode='addmm' is experimental: it fuses the second GEMM and residual and
    changes low-precision rounding; mode='low_peak' retains native rounding.
    No CUDA/ComfyUI initialization happens when importing this module alone.
    """
    # Lazy imports keep the numerical core usable without a ComfyUI installation.
    import comfy.lora as native_lora
    import comfy.lora_convert as lora_convert
    from comfy.patcher_extension import PatcherInjection
    from comfy.weight_adapter.bypass import BypassForwardHook, get_module_type_info
    from comfy.weight_adapter.lora import LoRAAdapter

    if mode not in ("low_peak", "addmm"):
        raise ValueError(f"Unknown LoRA mode: {mode}")
    strength = float(strength)
    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    if not lora_state:
        raise ValueError("LoRA state is empty")
    if any(k == "bypass_lora" or k.startswith("pika_low_peak_lora") for k in model.injections):
        raise ValueError("Stacked bypass LoRAs are not supported; load from an unpatched model")

    key_map = native_lora.model_lora_keys_unet(model.model, {})
    converted = lora_convert.convert_lora(dict(lora_state))
    # Native loaded_keys is shared between adapters; do not use its union as a
    # completeness check (an orphan alpha could otherwise be marked consumed).
    suffix_pairs = (
        (".lora_up.weight", ".lora_down.weight"),
        ("_lora.up.weight", "_lora.down.weight"),
        (".lora_B.weight", ".lora_A.weight"),
        (".lora.up.weight", ".lora.down.weight"),
        (".lora_B", ".lora_A"),
        (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
        (".lora_B.default.weight", ".lora_A.default.weight"),
    )
    consumed = set()
    targets = set()
    for source in converted:
        for up_suffix, down_suffix in suffix_pairs:
            if not source.endswith(up_suffix):
                continue
            stem = source[:-len(up_suffix)]
            down_key = stem + down_suffix
            if down_key not in converted:
                raise ValueError(f"LoRA down key is missing: {down_key}")
            target = key_map.get(stem)
            if not isinstance(target, str) or not target.endswith(".weight"):
                raise ValueError(f"Unsupported or missing LoRA target/offset: {stem}")
            if target in targets:
                raise ValueError(f"Duplicate LoRA target: {target}")
            targets.add(target)
            consumed.update((source, down_key))
            if stem + ".alpha" in converted:
                consumed.add(stem + ".alpha")
            break
    missing = set(converted) - consumed
    if missing:
        raise ValueError(f"LoRA keys not loaded: {', '.join(sorted(missing))}")
    loaded = native_lora.load_lora(converted, key_map, log_missing=False)
    if set(loaded) != targets:
        raise ValueError("Native LoRA mapping did not load every target")
    modules = []
    state_keys = set(model.model.state_dict())
    for key, adapter in loaded.items():
        if not isinstance(key, str) or not key.endswith(".weight"):
            raise ValueError(f"Unsupported LoRA target/offset: {key!r}")
        _validate_adapter(key, adapter, LoRAAdapter)
        if key not in state_keys:
            raise ValueError(f"LoRA target is missing from model: {key}")
        module = model.model.get_submodule(key[:-7])
        if get_module_type_info(module)["is_conv"]:
            raise ValueError(f"Convolutional LoRA is unsupported: {key}")
        up, down = adapter.weights[:2]
        if module.weight.shape[0] != up.shape[0]:
            raise ValueError(f"LoRA output shape does not match model: {key}")
        # Quantized storage may not expose the logical second dimension.
        in_features = getattr(module, "in_features", None)
        if in_features is not None and in_features != down.shape[1]:
            raise ValueError(f"LoRA input shape does not match model: {key}")
        modules.append((module, adapter))
    if not modules:
        raise ValueError("No supported LoRA targets found")
    if len({id(m) for m, _ in modules}) != len(modules):
        raise ValueError("Multiple LoRA targets alias the same module")

    class LowPeakHook(BypassForwardHook):
        def _bypass_forward(self, x, *args, **kwargs):
            base_out = self.original_forward(x, *args, **kwargs)
            up, down, alpha = self.adapter.weights[:3]
            return plain_lora_forward(x, base_out, up, down, alpha, self.multiplier, mode=mode)

        def inject(self):
            if self.original_forward is not None:
                return
            self.saved_weights = self.adapter.weights
            self.had_force_forward = "comfy_force_forward" in self.module.__dict__
            self.had_forward = "forward" in self.module.__dict__
            self.saved_forward = self.module.__dict__.get("forward")
            try:
                super().inject()
            except Exception:
                if self.original_forward is not None:
                    if self.module.forward == self.original_forward:
                        # Assignment failed before our wrapper was installed.
                        super().eject()
                        self._restore_fields()
                    else:
                        self.eject()
                self.adapter.weights = self.saved_weights
                raise

        def _restore_fields(self):
            if not self.had_force_forward:
                del self.module.comfy_force_forward
            if self.had_forward:
                self.module.forward = self.saved_forward
            elif "forward" in self.module.__dict__:
                del self.module.forward
            self.adapter.weights = self.saved_weights

        def eject(self):
            if self.original_forward is None:
                return
            current = self.module.forward
            if (getattr(current, "__self__", None) is not self
                    or getattr(current, "__func__", None) is not type(self)._bypass_forward):
                raise RuntimeError("LoRA forward was wrapped later; unload the later adapter before restoring this one")
            super().eject()
            self._restore_fields()

    hooks = [LowPeakHook(module, adapter, multiplier=strength) for module, adapter in modules]

    def inject_all(patcher):
        if "bypass_lora" in patcher.injections:
            raise ValueError("Stacked bypass LoRAs are not supported; use an unpatched base model")
        from comfy.model_management import get_torch_device
        compute = get_torch_device()
        requested = torch.device(patcher.load_device)
        if (requested.type != compute.type
                or (requested.index is not None and compute.index is not None
                    and requested.index != compute.index)):
            raise RuntimeError("LoRA bypass requires the patcher's load device to match ComfyUI's global compute device")
        injected = []
        try:
            for hook in hooks:
                hook.inject()
                injected.append(hook)
        except Exception:
            for hook in reversed(injected):
                hook.eject()
            raise

    def eject_all(patcher):
        for hook in reversed(hooks):
            hook.eject()

    result = model.clone()
    result.set_injections("pika_low_peak_lora", [PatcherInjection(inject_all, eject_all)])
    return result
