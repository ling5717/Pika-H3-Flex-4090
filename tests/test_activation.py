# SPDX-License-Identifier: GPL-3.0-only
"""CPU-only tests; the Triton GPU candidate is intentionally not exercised."""
import unittest
from unittest.mock import PropertyMock, patch

import torch
import torch.nn.functional as F

from comfy_extension.activation import SwiGLUActivation, swiglu_eager


class ActivationTests(unittest.TestCase):
    def test_native_rounding_gate_order_and_float_output(self):
        for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.linspace(-5, 6, 264, dtype=dtype).reshape(3, 88)
                before = x.clone()
                gate, up = x.chunk(2, -1)
                expected = F.silu(gate).mul_(up)
                for mode in ("eager", "auto"):
                    actual = SwiGLUActivation(mode)(x)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertNotEqual(actual.data_ptr(), x.data_ptr())
                self.assertTrue(torch.equal(x, before))

    def test_rounding_boundary_is_observable(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.linspace(-7, 7, 4000).reshape(10, 400).to(dtype)
            gate, up = x.chunk(2, -1)
            native = swiglu_eager(x)
            single_round = (F.silu(gate.float()) * up.float()).to(dtype)
            self.assertFalse(torch.equal(native, single_round))

    def test_shapes_noncontiguous_and_empty(self):
        for x in (torch.randn(2, 4, 12), torch.randn(12, 8).t(),
                  torch.empty(0, 8), torch.empty(3, 0), torch.randn(12)):
            actual = SwiGLUActivation("auto")(x)
            expected = swiglu_eager(x)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.shape, (*x.shape[:-1], x.shape[-1] // 2))

    def test_bad_shape_mode_and_strict_unsupported(self):
        with self.assertRaises(ValueError):
            SwiGLUActivation("native")
        for x in (torch.ones(()), torch.ones(3, 5)):
            with self.assertRaises(ValueError):
                swiglu_eager(x)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            SwiGLUActivation("triton")(torch.ones(3, 8))

    def test_cpu_never_loads_kernel_or_probes_cuda(self):
        with patch("comfy_extension.activation._load_kernel", side_effect=AssertionError), \
             patch("torch.cuda.is_available", side_effect=AssertionError), \
             patch("torch.cuda.init", side_effect=AssertionError):
            activation = SwiGLUActivation("auto")
            activation(torch.ones(2, 8))
            self.assertEqual(activation.last_backend, "eager")
            self.assertIsNone(activation._kernel)
            activation.clear()

    def test_autograd_fallback_preserves_gradient(self):
        x = torch.randn(2, 8, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        SwiGLUActivation("auto")(x).sum().backward()
        swiglu_eager(y).sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_launch_failure_and_backend_reporting_without_gpu(self):
        # Only override metadata. The mocked launcher never executes GPU code.
        x = torch.ones(2, 8, dtype=torch.bfloat16)
        with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True):
            activation = SwiGLUActivation("auto")
            with patch("comfy_extension.activation._load_kernel", side_effect=ImportError("Triton unavailable")) as load:
                with self.assertLogs(level="WARNING"):
                    torch.testing.assert_close(activation(x), swiglu_eager(x))
                activation(x)
                load.assert_called_once()
                self.assertEqual(activation.last_backend, "eager")
                activation.clear()
                with self.assertLogs(level="WARNING"):
                    activation(x)
                self.assertEqual(load.call_count, 2)
            with patch("comfy_extension.activation._load_kernel", side_effect=RuntimeError("strict compiler")):
                for mode in ("triton", "auto"):
                    with self.assertRaisesRegex(RuntimeError, "strict compiler"):
                        SwiGLUActivation(mode)(x)
            class FakeKernel:
                def __getitem__(self, grid):
                    def launch(source, out, n, width, block, **kwargs):
                        self_outer.assertEqual(grid, (1,))
                        self_outer.assertFalse(kwargs["enable_fp_fusion"])
                        out.copy_(swiglu_eager(source))
                    return launch
            self_outer = self
            with patch("comfy_extension.activation._load_kernel", return_value=FakeKernel()):
                strict = SwiGLUActivation("triton")
                torch.testing.assert_close(strict(x), swiglu_eager(x))
                self.assertEqual(strict.last_backend, "triton")

    def test_eager_exception_values(self):
        x = torch.tensor([[float("inf"), -float("inf"), float("nan"), 0.,
                           1., 1., 1., float("inf")]])
        torch.testing.assert_close(SwiGLUActivation("auto")(x), swiglu_eager(x),
                                   equal_nan=True, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
