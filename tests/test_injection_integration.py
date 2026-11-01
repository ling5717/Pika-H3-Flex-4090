# SPDX-License-Identifier: GPL-3.0-only
"""Real Comfy loader lifecycle regressions, isolated in a CPU subprocess.

Set COMFYUI_PATH to an existing ComfyUI checkout. Never imports Comfy in the
parent test runner; never installs dependencies or loads production weights.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class InjectionIntegrationTests(unittest.TestCase):
    def test_native_cpu_loader_and_stack_lifecycle(self):
        native = os.environ.get("COMFYUI_PATH")
        if not native:
            self.skipTest("Set COMFYUI_PATH for native CPU integration")
        if not (Path(native) / "comfy" / "model_management.py").is_file():
            self.skipTest("COMFYUI_PATH does not contain ComfyUI")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--native-child", native],
            capture_output=True, text=True, timeout=180, env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("native CPU active-clone and stack regressions passed", proc.stdout)


def native_child(native):
    from types import SimpleNamespace
    from unittest.mock import patch
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, native)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()

    # Optional backend discovery can probe availability even with --cpu. Fail
    # immediately if anything attempts CUDA initialization rather than silently
    # allowing a GPU test to run alongside production inference.
    with patch.object(torch.cuda, "is_available", return_value=False), \
         patch.object(torch.cuda, "init", side_effect=AssertionError("CPU test initialized CUDA")), \
         patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CPU test initialized CUDA")):
        import comfy.model_management as management
        import comfy.sd
        from comfy.ldm.minimax.model import MiniMaxH3Model, MLP
        from comfy.model_patcher import ModelPatcher
        from comfy_extension.ffn import optimize_model
        from comfy_extension.lora import load_low_peak_lora

        torch.manual_seed(4090)
        torch.set_num_threads(1)
        root = MiniMaxH3Model.__new__(MiniMaxH3Model)
        torch.nn.Module.__init__(root)
        root.blocks = torch.nn.ModuleList([MLP(8, 12, operations=torch.nn)])
        base = torch.nn.Module()
        base.diffusion_model = root
        base.model_config = SimpleNamespace(unet_config={})
        cpu = torch.device("cpu")
        patcher = ModelPatcher(base, cpu, cpu)
        mlp = root.blocks[0]
        original_mlp = mlp.forward
        original_fc1 = mlp.fc1.forward
        original_fc2 = mlp.fc2.forward
        x = torch.randn(8195, 8)
        # Plain native modules give deterministic, independently checkable
        # expected values without materializing a real H3 checkpoint.
        up = torch.randn(24, 2) * 0.01
        down = torch.randn(2, 8) * 0.01
        state = {
            "diffusion_model.blocks.0.fc1.lora_up.weight": up,
            "diffusion_model.blocks.0.fc1.lora_down.weight": down,
        }

        def check_original(expected):
            assert mlp.forward == original_mlp
            assert mlp.fc1.forward == original_fc1
            assert mlp.fc2.forward == original_fc2
            assert not getattr(mlp.fc1, "comfy_force_forward", False)
            torch.testing.assert_close(mlp(x), expected)

        def expected_lora(strength):
            projected = original_fc1(x) + F.linear(F.linear(x, down), up) * strength
            gate, value = projected.chunk(2, -1)
            return original_fc2(F.silu(gate).mul_(value))

        with torch.no_grad():
            baseline = original_mlp(x)
            try:
                # Replacement of an already-loaded FFN policy must use native
                # load_models_gpu. Direct inject on a shared active clone is
                # deliberately NOT asserted as a supported lifecycle protocol.
                first = optimize_model(patcher, 8192)
                management.load_models_gpu([first])
                assert first.is_injected
                first_forward = mlp.forward
                second = optimize_model(first, 32768)
                assert second.is_injected  # Real clone copies this flag.
                management.load_models_gpu([second])
                assert not first.is_injected and second.is_injected
                assert mlp.forward is not first_forward
                torch.testing.assert_close(mlp(x), baseline)

                # Add a LoRA to an active FFN model. The host must eject old
                # wrappers, move/inject adapter state, then preserve both math
                # and the newly requested FFN policy.
                adapted = load_low_peak_lora(second, state, strength=0.7)
                management.load_models_gpu([adapted])
                assert not second.is_injected and adapted.is_injected
                assert mlp.fc1.comfy_force_forward
                torch.testing.assert_close(mlp(x), expected_lora(0.7))
                adapted_forward = mlp.forward
                replacement = optimize_model(adapted, 8192)
                management.load_models_gpu([replacement])
                assert not adapted.is_injected and replacement.is_injected
                assert mlp.forward is not adapted_forward
                torch.testing.assert_close(mlp(x), expected_lora(0.7))
                management.unload_all_models()
                check_original(baseline)

                # A later stock bypass loader cannot be prevented at node
                # construction time. The low-peak injection must reject its
                # presence BEFORE wrapping any linear, not corrupt on eject.
                low_peak = load_low_peak_lora(patcher, state, strength=0.7)
                stacked, _ = comfy.sd.load_bypass_lora_for_models(
                    low_peak, None, state, 1.0, 0.0)
                assert "pika_low_peak_lora" in stacked.injections
                assert "bypass_lora" in stacked.injections
                try:
                    management.load_models_gpu([stacked])
                except ValueError as exc:
                    assert "Stacked bypass LoRAs" in str(exc), str(exc)
                else:
                    raise AssertionError("LowPeak -> stock BypassLoRA stack was accepted")
                assert not stacked.is_injected
                assert "comfy_force_forward" not in mlp.fc1.__dict__
                check_original(baseline)
                stacked.eject_model()
                management.unload_all_models()
                check_original(baseline)

                # A rejected node must not poison subsequent normal loading.
                valid = load_low_peak_lora(patcher, state, strength=-0.4)
                management.load_models_gpu([valid])
                torch.testing.assert_close(mlp(x), expected_lora(-0.4))
                # An unknown later patch must never be clobbered by eject.
                owned_forward = mlp.fc1.forward
                def later_forward(tokens):
                    return owned_forward(tokens)
                mlp.fc1.forward = later_forward
                try:
                    try:
                        valid.eject_model()
                    except RuntimeError as exc:
                        assert "wrapped later" in str(exc), str(exc)
                    else:
                        raise AssertionError("Eject clobbered an unknown later forward")
                    assert mlp.fc1.forward is later_forward
                finally:
                    mlp.fc1.forward = owned_forward
                management.unload_all_models()
                check_original(baseline)
                management.load_models_gpu([patcher])
                check_original(baseline)
            finally:
                management.unload_all_models()
            assert not torch.cuda.is_initialized(), "CPU tests initialized CUDA"
            print("native CPU active-clone and stack regressions passed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--native-child":
        native_child(sys.argv[2])
    else:
        unittest.main()
