# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only LoRA regression tests; native integration runs in a --cpu subprocess.

Set COMFYUI_PATH to test a different existing ComfyUI checkout. No GPU calls,
package installation, model downloads, or modifications to ComfyUI are needed.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("low_peak_lora", ROOT / "comfy_extension" / "lora.py")
lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora)


def tensors(dtype, grad=False):
    gen = torch.Generator(device="cpu").manual_seed(89)
    return [torch.randn(shape, generator=gen, dtype=dtype, requires_grad=grad)
            for shape in ((2, 7, 8), (8, 3), (3, 8))]


class NumericalTests(unittest.TestCase):
    def test_native_rounding_all_dtypes_and_scales(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for alpha in (None, 3.0, 1.75):
                for strength in (0.0, 1.0, -0.7, 0.35):
                    with self.subTest(dtype=dtype, alpha=alpha, strength=strength), torch.no_grad():
                        x, up, down = tensors(dtype)
                        base = x.sin()
                        original = base.clone()
                        scale = (alpha / 3 if alpha is not None else 1.0) * strength
                        expected = base + F.linear(F.linear(x, down), up) * scale
                        actual = lora.plain_lora_forward(x, base, up, down, alpha, strength)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(base, original, rtol=0, atol=0)
                        self.assertEqual(actual.dtype, dtype)

    def test_identity_and_view_base_alias_not_modified(self):
        with torch.no_grad():
            x, up, down = tensors(torch.float32)
            before = x.clone()
            for base in (x, x.view_as(x)):
                result = lora.plain_lora_forward(x, base, up, down)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
                self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_no_scale_or_residual_large_copy(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        class Ops(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.names = []

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.names.append(str(func))
                return func(*args, **(kwargs or {}))

        with torch.no_grad():
            x, up, down = tensors(torch.bfloat16)
            for strength in (1.0, 0.4):
                ops = Ops()
                with ops:
                    lora.plain_lora_forward(x, x, up, down, strength=strength)
                self.assertNotIn("aten.mul.Tensor", ops.names)
                self.assertNotIn("aten.add.Tensor", ops.names)
                self.assertIn("aten.add_.Tensor", ops.names)
                self.assertEqual("aten.mul_.Tensor" in ops.names, strength != 1.0)

    def test_autograd_fallback_matches_gradients(self):
        x, up, down = tensors(torch.float32, grad=True)
        base = x.sigmoid()
        before = base.clone()
        actual = lora.plain_lora_forward(x, base, up, down, alpha=2.0, strength=0.6)
        expected = base + F.linear(F.linear(x, down), up) * 0.4
        grads_actual = torch.autograd.grad(actual.square().sum(), (x, up, down), retain_graph=True)
        grads_expected = torch.autograd.grad(expected.square().sum(), (x, up, down))
        torch.testing.assert_close(base, before, rtol=0, atol=0)
        for a, b in zip(grads_actual, grads_expected):
            torch.testing.assert_close(a, b)

    def test_mixed_output_dtype_preserves_promotion(self):
        with torch.no_grad():
            x, up, down = tensors(torch.float16)
            base = x.float()
            actual = lora.plain_lora_forward(x, base, up, down, strength=0.7)
            expected = base + F.linear(F.linear(x, down), up) * 0.7
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_experimental_addmm_metrics_and_alias(self):
        for dtype, max_nrmse in ((torch.bfloat16, 0.006), (torch.float16, 0.001),
                                 (torch.float32, 1e-6)):
            with self.subTest(dtype=dtype), torch.no_grad():
                x, up, down = tensors(dtype)
                before = x.clone()
                expected = x + F.linear(F.linear(x, down), up) * (1.75 / 3 * 0.7)
                actual = lora.plain_lora_forward(x, x, up, down, alpha=1.75,
                                                strength=0.7, mode="addmm")
                a, b = actual.float().flatten(), expected.float().flatten()
                nrmse = float((a - b).square().mean().sqrt() / b.square().mean().sqrt())
                cosine = float(F.cosine_similarity(a, b, dim=0))
                self.assertLess(nrmse, max_nrmse)
                self.assertGreater(cosine, 0.9999)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(x, before, rtol=0, atol=0)
                print(f"addmm {dtype}: NRMSE={nrmse:.8g}, cosine={cosine:.9g}")

    def test_addmm_gradients_and_mode_validation(self):
        x, up, down = tensors(torch.float32, grad=True)
        actual = lora.plain_lora_forward(x, x, up, down, strength=0.7, mode="addmm")
        expected = x + F.linear(F.linear(x, down), up) * 0.7
        ga = torch.autograd.grad(actual.sum(), (x, up, down), retain_graph=True)
        gb = torch.autograd.grad(expected.sum(), (x, up, down))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b)
        with self.assertRaisesRegex(ValueError, "Unknown LoRA mode"):
            lora.plain_lora_forward(x, x, up, down, mode="typo")

    def test_unknown_adapter_rejected(self):
        class Plain:
            pass

        class Unknown(Plain):
            pass

        with self.assertRaisesRegex(ValueError, "Unsupported adapter"):
            lora._validate_adapter("layer.weight", Unknown(), Plain)


class NativeIntegrationTests(unittest.TestCase):
    def test_native_cpu_subprocess(self):
        native = Path(os.environ.get("COMFYUI_PATH", ""))
        if not os.environ.get("COMFYUI_PATH") or not (native / "comfy" / "lora.py").is_file():
            self.skipTest("Set COMFYUI_PATH for native integration")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--native-child", str(native)],
                              capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("native CPU lifecycle and strict mapping passed", proc.stdout)


def native_child(native):
    sys.path.insert(0, native)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    # Comfy's optional backend registration probes availability even in CPU mode.
    # Only this availability probe is stubbed; CUDA initialization itself must not run.
    with patch.object(torch.cuda, "is_available", return_value=False):
        import comfy.lora as native_lora
        from comfy.model_patcher import ModelPatcher
        from comfy.weight_adapter.lora import LoRAAdapter

        class CPUModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.diffusion_model = torch.nn.Module()
                self.diffusion_model.proj = torch.nn.Linear(8, 8, bias=False, device="cpu")
                from types import SimpleNamespace
                self.model_config = SimpleNamespace(unet_config={})

        model = CPUModel()
        layer = model.diffusion_model.proj
        original_forward = layer.forward
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        x, up, down = tensors(torch.float32)
        state = {"diffusion_model.proj.lora_up.weight": up,
                 "diffusion_model.proj.lora_down.weight": down}
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            layer.to(dtype=dtype)
            tx = x.to(dtype)
            result = lora.load_low_peak_lora(patcher, state, strength=0.7)
            assert isinstance(result, ModelPatcher) and result is not patcher
            assert not patcher.injections
            with torch.no_grad():
                base = original_forward(tx)
                adapter = LoRAAdapter(set(state), (up.to(dtype), down.to(dtype), None, None, None, None))
                adapter.multiplier = 0.7
                expected = base + adapter.h(tx, base)
                for _ in range(2):
                    result.inject_model()
                    torch.testing.assert_close(layer(tx), expected, rtol=0, atol=0)
                    result.eject_model()
                    assert layer.forward == original_forward
                    assert "comfy_force_forward" not in layer.__dict__
                    torch.testing.assert_close(layer(tx), base, rtol=0, atol=0)
            fused = lora.load_low_peak_lora(patcher, state, strength=0.7, mode="addmm")
            with torch.no_grad():
                fused.inject_model()
                actual = layer(tx)
                fused.eject_model()
                error = (actual.float() - expected.float()).square().mean().sqrt()
                norm = expected.float().square().mean().sqrt()
                assert error / norm < 0.01
                assert actual.dtype == dtype and layer.forward == original_forward
            try:
                lora.load_low_peak_lora(result, state)
            except ValueError:
                pass
            else:
                raise AssertionError("stacked bypass accepted")
        for extra in ({"unknown.weight": torch.ones(1)},
                      {"diffusion_model.proj.dora_scale": torch.ones(8)},
                      {"orphan.alpha": torch.tensor(1.0)},
                      {"diffusion_model.proj.diff": torch.zeros(8, 8)}):
            try:
                lora.load_low_peak_lora(patcher, state | extra)
            except ValueError:
                pass
            else:
                raise AssertionError(f"unconsumed keys silently accepted: {extra.keys()}")
        # Native loader returning an unknown adapter cannot silently lose a target.
        with patch.object(native_lora, "load_lora", return_value={"diffusion_model.proj.weight": object()}):
            try:
                lora.load_low_peak_lora(patcher, state)
            except ValueError as exc:
                assert "Unsupported adapter" in str(exc)
            else:
                raise AssertionError("unknown native adapter accepted")
        for invalid in ({}, {"diffusion_model.proj.lora_up.weight": up}):
            try:
                lora.load_low_peak_lora(patcher, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("empty or incomplete state accepted")
        model.diffusion_model.alias = layer
        alias_state = {"diffusion_model.alias.lora_up.weight": up,
                       "diffusion_model.alias.lora_down.weight": down}
        try:
            lora.load_low_peak_lora(patcher, state | alias_state)
        except ValueError as exc:
            assert "alias" in str(exc)
        else:
            raise AssertionError("aliased modules accepted")
        del model.diffusion_model.alias

        class RejectForwardOnce(torch.nn.Linear):
            def __setattr__(self, name, value):
                if name == "forward" and self.__dict__.get("reject_next", False):
                    self.reject_next = False
                    raise RuntimeError("injected assignment failure")
                super().__setattr__(name, value)

        model.diffusion_model.failing = RejectForwardOnce(8, 8, bias=False, device="cpu")
        failing = model.diffusion_model.failing
        failing.reject_next = True
        original_failing = failing.forward
        failed_state = state | {"diffusion_model.failing.lora_up.weight": up,
                                "diffusion_model.failing.lora_down.weight": down}
        result = lora.load_low_peak_lora(patcher, failed_state)
        try:
            result.inject_model()
        except RuntimeError as exc:
            assert "injected assignment failure" in str(exc)
        else:
            raise AssertionError("fault injection did not trigger")
        assert layer.forward == original_forward
        assert failing.forward == original_failing
        assert "comfy_force_forward" not in layer.__dict__
        assert "comfy_force_forward" not in failing.__dict__
        del model.diffusion_model.failing

        # Exercise 208 independently mapped groups without loading real weights.
        model.diffusion_model.blocks = torch.nn.ModuleList([
            torch.nn.Linear(8, 8, bias=False, device="cpu") for _ in range(208)])
        fixture = {}
        for index in range(208):
            fixture[f"diffusion_model.blocks.{index}.lora_up.weight"] = up
            fixture[f"diffusion_model.blocks.{index}.lora_down.weight"] = down
        result = lora.load_low_peak_lora(patcher, fixture)
        result.inject_model()
        assert all(m.comfy_force_forward for m in model.diffusion_model.blocks)
        result.eject_model()
        assert all("comfy_force_forward" not in m.__dict__ for m in model.diffusion_model.blocks)
        assert not torch.cuda.is_initialized(), "CPU tests initialized CUDA"
        print("native CPU lifecycle and strict mapping passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--native-child":
        native_child(sys.argv[2])
    else:
        unittest.main()
