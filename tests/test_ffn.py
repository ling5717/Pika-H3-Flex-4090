# SPDX-License-Identifier: GPL-3.0-only
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy_extension.ffn import chunked_call, FFNPolicy


class FFNTests(unittest.TestCase):
    def test_chunked_token_math(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = torch.randn(32771, 12, dtype=dtype)
            a = torch.randn(24, 12, dtype=dtype)
            b = torch.randn(12, 12, dtype=dtype)
            def f(t):
                gate, up = F.linear(t, a).chunk(2, -1)
                return F.linear(F.silu(gate) * up, b)
            ref = f(x)
            for size in (8192, 16384, 32768):
                with self.subTest(dtype=dtype, size=size):
                    out = chunked_call(f, x, size)
                    self.assertEqual(out.shape, ref.shape)
                    self.assertEqual(out.dtype, ref.dtype)
                    torch.testing.assert_close(out, ref)

    def test_passthrough_layout_and_empty(self):
        for shape in ((0, 12), (3, 12), (2, 3, 12)):
            x = torch.ones(shape)
            self.assertTrue(torch.equal(chunked_call(lambda t: t + 1, x, 8192), x + 1))

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            FFNPolicy(4096)
        with self.assertRaises(ValueError):
            FFNPolicy(16384, "unknown")

    def test_native_lifecycle(self):
        native = os.environ.get("COMFYUI_PATH")
        if not native:
            self.skipTest("Set COMFYUI_PATH for native integration")
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--native", native],
                                capture_output=True, text=True, timeout=120,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FFN native lifecycle passed", result.stdout)


def native_check(native):
    from unittest.mock import patch
    sys.path.insert(0, native)
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    with patch.object(torch.cuda, "is_available", return_value=False):
        from comfy.ldm.minimax.model import MiniMaxH3Model, MLP
        root = MiniMaxH3Model.__new__(MiniMaxH3Model)
        torch.nn.Module.__init__(root)
        root.blocks = torch.nn.ModuleList([MLP(8, 12, operations=torch.nn)])
        class Patcher:
            def get_model_object(self, name):
                assert name == "diffusion_model"
                return root
        m = root.blocks[0]
        x = torch.randn(8195, 8)
        original = m.forward
        ref = original(x)
        p = FFNPolicy(8192)
        p.inject(Patcher())
        p.inject(Patcher())
        torch.testing.assert_close(m(x), ref)
        p.eject(Patcher())
        p.eject(Patcher())
        assert "forward" not in m.__dict__
        torch.testing.assert_close(m(x), ref)
        assert not torch.cuda.is_initialized()
        print("FFN native lifecycle passed")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--native":
        native_check(sys.argv[2])
    else:
        unittest.main()
