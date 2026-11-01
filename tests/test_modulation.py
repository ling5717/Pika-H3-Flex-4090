# SPDX-License-Identifier: GPL-3.0-only
"""CPU regression and native lifecycle checks for optional H3 RMS/AdaLN fusion."""

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
spec = importlib.util.spec_from_file_location("h3_modulation", ROOT / "comfy_extension" / "modulation.py")
modulation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulation)


def reference(x, weight, shift, scale, segments, eps):
    out = F.rms_norm(x, (x.shape[-1],), weight, eps)
    for a, b, row in segments:
        out[a:b].mul_(1.0 + scale[row].to(out.dtype)).add_(shift[row].to(out.dtype))
    return out


def fixture(dtype, width=5376):
    generator = torch.Generator(device="cpu").manual_seed(18)
    x = torch.randn((11, width), generator=generator, dtype=dtype)
    weight = (torch.randn((width,), generator=generator) * 0.1 + 1.0).to(dtype)
    scale = torch.randn((9, width), generator=generator) * 0.2
    shift = torch.randn((9, width), generator=generator) * 0.1
    segments = [(0, 2, 1), (2, 5, torch.tensor([6, 0, 3])), (5, 5, 3), (5, 9, 4), (9, 11, 8)]
    return x, weight, shift, scale, segments


def metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    nrmse = float((a - b).square().mean().sqrt() / b.square().mean().sqrt())
    cosine = float(F.cosine_similarity(a, b, dim=0))
    return nrmse, cosine


class SegmentTests(unittest.TestCase):
    def test_real_width_dtype_row_semantics(self):
        # This callback is a CPU reference to the shared operation's contract,
        # not a replacement production RMS kernel. Native CK is tested below.
        def shared_contract(x, scale, shift, eps):
            return F.rms_norm(x, (x.shape[-1],), eps=eps) * (1 + scale) + shift

        for dtype, tolerance in ((torch.bfloat16, 0.007), (torch.float16, 0.001), (torch.float32, 1e-6)):
            with self.subTest(dtype=dtype), torch.no_grad():
                x, weight, shift, scale, segments = fixture(dtype)
                original = x.clone()
                expected = reference(x, weight, shift, scale, segments, 1e-5)
                actual = modulation._fused_segments(x, weight, shift, scale, segments, 1e-5, shared_contract)
                nrmse, cosine = metrics(actual, expected)
                self.assertLess(nrmse, tolerance)
                self.assertGreater(cosine, 0.9999)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
                print(f"RMS/AdaLN {dtype}, D=5376: NRMSE={nrmse:.8g}, cosine={cosine:.9g}")

    def test_eps_single_segment_noncontiguous_and_no_affine(self):
        calls = []

        def shared(x, scale, shift, eps):
            calls.append((tuple(x.shape), tuple(scale.shape), eps))
            return F.rms_norm(x, (x.shape[-1],), eps=eps) * (1 + scale) + shift

        with torch.no_grad():
            x, _, shift, scale, _ = fixture(torch.float32, width=24)
            x = x[:, ::2]
            shift, scale = shift[:, ::2], scale[:, ::2]
            for eps in (1e-5, 0.3):
                expected = reference(x, None, shift, scale, [(0, 11, 3)], eps)
                actual = modulation._fused_segments(x, None, shift, scale, [(0, 11, 3)], eps, shared)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(calls[-1], ((11, 12), (12,), eps))
            self.assertEqual(len(calls), 2)

    def test_invalid_boundaries_and_per_token_rows(self):
        x, weight, shift, scale, _ = fixture(torch.float32, width=8)
        for segments in ([(1, 11, 1)], [(0, 12, 1)], [(0, 4, 1), (5, 11, 1)],
                         [(0, 11, torch.tensor([0, 1]))], [(0, 4, 1)]):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                modulation._fused_segments(x, weight, shift, scale, segments, 1e-5,
                                           lambda a, b, c, d: a.clone())


class NativeIntegrationTests(unittest.TestCase):
    def test_native_cpu_process(self):
        native = Path(os.environ.get("COMFYUI_PATH", ""))
        if not os.environ.get("COMFYUI_PATH") or not (native / "comfy" / "ldm" / "minimax" / "model.py").is_file():
            self.skipTest("Set COMFYUI_PATH to test native H3 integration")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--native-child", str(native)],
                              env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("native CK casting, segments, and lifecycle passed", proc.stdout)


def native_child(native):
    sys.path.insert(0, native)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    with patch.object(torch.cuda, "is_available", return_value=False):
        import comfy.ops
        import comfy.quant_ops
        from comfy.ldm.minimax.model import MiniMaxH3Model, _mod_scale_shift
        from comfy.model_patcher import ModelPatcher

        for dtype, tolerance in ((torch.bfloat16, 0.007), (torch.float16, 0.001), (torch.float32, 1e-6)):
            x, weight, shift, scale, segments = fixture(dtype)
            norm = comfy.ops.manual_cast.RMSNorm(5376, eps=1e-5, dtype=dtype, device="cpu")
            with torch.no_grad():
                norm.weight.copy_(weight)
                expected = _mod_scale_shift(norm(x), shift, scale, segments)
                actual = modulation.fused_norm_modulation(x, norm, shift, scale, segments)
                assert metrics(actual, expected)[0] < tolerance
                assert actual.dtype == dtype
                # Keep the native patch/cast path, including registered weight transforms.
                norm.weight_function = [lambda w: w * 0.9]
                expected = _mod_scale_shift(norm(x), shift, scale, segments)
                actual = modulation.fused_norm_modulation(x, norm, shift, scale, segments)
                assert metrics(actual, expected)[0] < tolerance
                # A native weight transform can promote output: retain that contract.
                norm.weight_function = [lambda w: w.float() * 0.9]
                expected = _mod_scale_shift(norm(x), shift, scale, segments)
                actual = modulation.fused_norm_modulation(x, norm, shift, scale, segments)
                assert actual.dtype == expected.dtype
                if dtype != torch.float32:
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                else:
                    assert metrics(actual, expected)[0] < tolerance
                norm.weight_function = []
                norm.eps = None
                expected = _mod_scale_shift(norm(x), shift, scale, segments)
                actual = modulation.fused_norm_modulation(x, norm, shift, scale, segments)
                assert metrics(actual, expected)[0] < tolerance
            # Grad-enabled path is native, not the inference-only custom op.
            actual = modulation.fused_norm_modulation(x, norm, shift, scale, segments)
            expected = _mod_scale_shift(norm(x), shift, scale, segments)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Cast/offload context must be released even when segment validation fails.
        from contextlib import contextmanager
        original_cast = norm.cast_weight
        events = []

        @contextmanager
        def tracked_cast(tokens):
            events.append("enter")
            try:
                with original_cast(tokens) as state:
                    yield state
            finally:
                events.append("exit")

        norm.cast_weight = tracked_cast
        with torch.no_grad():
            try:
                modulation.fused_norm_modulation(x, norm, shift, scale, [(1, 11, 0)])
            except ValueError:
                pass
            else:
                raise AssertionError("invalid segment was accepted")
        assert events == ["enter", "exit"]
        del norm.cast_weight

        # No-cast RMSNorm with a higher-precision affine weight must also stay native.
        with torch.no_grad():
            x, _, shift, scale, segments = fixture(torch.bfloat16, width=16)
            norm = comfy.ops.disable_weight_init.RMSNorm(16, eps=1e-5, dtype=torch.float32, device="cpu")
            norm.weight.fill_(1.0)
            expected = _mod_scale_shift(norm(x), shift, scale, segments)
            actual = modulation.fused_norm_modulation(x, norm, shift, scale, segments)
            assert actual.dtype == expected.dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        root = MiniMaxH3Model(hidden_size=16, num_layers=2, token_refiner_num_layers=1,
                             num_attention_heads=2, attention_head_dim=8, ffn_hidden_size=24,
                             timestep_input_dim=8, time_embed_hidden_size=16, time_embed_dim=8,
                             text_dim=12, dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(919)
            for parameter in root.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
            for block in root.blocks:
                block.norm1.weight.add_(1.0)
                block.norm2.weight.add_(1.0)
        model = torch.nn.Module()
        model.diffusion_model = root
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        result = modulation.enable_fused_modulation(patcher)
        assert result is not patcher and not patcher.injections
        originals = [block.forward for block in root.blocks]
        mlp_originals = [block.mlp.forward for block in root.blocks]
        attn_originals = [block.attn.forward for block in root.blocks]
        x = torch.randn((7, 16), generator=generator)
        t_emb = torch.randn((3, 8), generator=generator)
        segments = [(0, 2, 1), (2, 5, torch.tensor([6, 0, 3])), (5, 7, 8)]
        rope, options = object(), {"caller": "unchanged"}
        attention_calls = []

        def attention(h, rope_freqs, transformer_options):
            assert rope_freqs is rope and transformer_options is options
            attention_calls.append(tuple(h.shape))
            return h * 0.1

        with torch.no_grad():
            expected = [block(x.clone(), t_emb, segments, rope, transformer_options=options, attention=attention)
                        for block in root.blocks]
            for _ in range(2):
                result.inject_model()
                for index, block in enumerate(root.blocks):
                    actual = block(x.clone(), t_emb, segments, rope, transformer_options=options, attention=attention)
                    torch.testing.assert_close(actual, expected[index], rtol=1e-5, atol=1e-6)
                    assert block.mlp.forward == mlp_originals[index]
                    assert block.attn.forward == attn_originals[index]
                result.eject_model()
                for block, original in zip(root.blocks, originals):
                    assert block.forward == original and "forward" not in block.__dict__
            assert len(attention_calls) == 6
        # Never overwrite a wrapper attached after us during ejection.
        result.inject_model()
        wrapped = root.blocks[0].forward
        root.blocks[0].forward = lambda *args, **kwargs: None
        try:
            result.eject_model()
        except RuntimeError:
            pass
        else:
            raise AssertionError("a later forward wrapper was overwritten")
        root.blocks[0].forward = wrapped
        result.eject_model()
        assert "forward" not in root.blocks[0].__dict__
        try:
            modulation.enable_fused_modulation(result)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate modulation injection accepted")
        # Reject an already wrapped block before changing any block.
        root.blocks[1].forward = lambda *args, **kwargs: None
        try:
            result.inject_model()
        except RuntimeError:
            pass
        else:
            raise AssertionError("prewrapped DiT block accepted")
        assert "forward" not in root.blocks[0].__dict__
        del root.blocks[1].forward
        assert not torch.cuda.is_initialized(), "CPU tests initialized CUDA"
        print("native CK casting, segments, and lifecycle passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--native-child":
        native_child(sys.argv[2])
    else:
        unittest.main()
