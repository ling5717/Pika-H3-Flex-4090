# SPDX-License-Identifier: GPL-3.0-only
"""CPU orchestration tests with mocked compiler; no inductor/GPU execution."""
import gc
import unittest
from unittest.mock import patch
import weakref

import torch

from comfy_extension.compilation import TensorSubgraph


def delta(x, a, b):
    return (x @ a) @ b


def square(x):
    return x * x


class CompilationTests(unittest.TestCase):
    def test_default_off(self):
        wrapper = TensorSubgraph(square)
        with patch("torch.compile", side_effect=AssertionError), \
             patch("torch.cuda.init", side_effect=AssertionError):
            torch.testing.assert_close(wrapper(torch.ones(3)), torch.ones(3))
        self.assertEqual(wrapper.cache_size, 0)
        self.assertEqual(wrapper.last_backend, "eager")

    def test_compile_contract_and_reuse(self):
        args = (torch.randn(2, 3), torch.randn(3, 2), torch.randn(2, 4))
        wrapper = TensorSubgraph(delta, enabled=True)
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as compile:
            for _ in range(2):
                torch.testing.assert_close(wrapper(*args), delta(*args))
        compile.assert_called_once_with(delta, backend="inductor", fullgraph=True, dynamic=True)
        self.assertEqual(wrapper.last_backend, "inductor")

    def test_lru_bound_clear_close_and_no_input_cache(self):
        wrapper = TensorSubgraph(square, enabled=True, max_entries=2)
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as compile:
            x = torch.ones(2)
            ref = weakref.ref(x)
            wrapper(x)
            del x
            gc.collect()
            self.assertIsNone(ref())
            for n in (3, 4, 2):
                wrapper(torch.ones(n))
                self.assertLessEqual(wrapper.cache_size, 2)
            self.assertEqual(compile.call_count, 4)
            wrapper.clear()
            self.assertEqual(wrapper.cache_size, 0)
            wrapper(torch.ones(2))
        wrapper.close()
        self.assertIsNone(wrapper.fn)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            wrapper(torch.ones(2))

    def test_strict_compile_and_execution_failures(self):
        wrapper = TensorSubgraph(square, enabled=True)
        with patch("torch.compile", side_effect=RuntimeError("compiler failure")):
            with self.assertRaisesRegex(RuntimeError, "compiler failure"):
                wrapper(torch.ones(3))
        self.assertEqual(wrapper.cache_size, 0)
        with patch("torch.compile", return_value=lambda x: (_ for _ in ()).throw(RuntimeError("run failure"))):
            with self.assertRaisesRegex(RuntimeError, "run failure"):
                wrapper(torch.ones(3))
        self.assertEqual(wrapper.cache_size, 0)

    def test_non_strict_failure_logged_and_bounded(self):
        for at_compile in (True, False):
            wrapper = TensorSubgraph(square, enabled=True, strict=False, max_entries=1)
            kwargs = ({"side_effect": RuntimeError("bad compile")} if at_compile else
                      {"return_value": lambda x: (_ for _ in ()).throw(RuntimeError("bad run"))})
            with patch("torch.compile", **kwargs) as compile, self.assertLogs(level="WARNING"):
                for _ in range(2):
                    torch.testing.assert_close(wrapper(torch.ones(3)), torch.ones(3))
                self.assertEqual(compile.call_count, 1)
            self.assertEqual(wrapper.last_backend, "eager")
            self.assertEqual(wrapper.cache_size, 1)

    def test_metadata_guard_and_input_contract(self):
        wrapper = TensorSubgraph(square, enabled=True)
        with patch("torch.compile", side_effect=lambda fn, **kw: fn) as compile:
            wrapper(torch.ones(2))
            with torch.no_grad():
                wrapper(torch.ones(2))
            wrapper(torch.ones(2, dtype=torch.float64))
            self.assertEqual(compile.call_count, 3)
        for args in ((), (3,), (torch.ones(2), 3)):
            with self.assertRaises(TypeError):
                wrapper(*args)
        with self.assertRaises(TypeError):
            TensorSubgraph(torch.nn.Linear(2, 2))
        with self.assertRaises(TypeError):
            TensorSubgraph(torch.nn.Linear(2, 2).forward)
        captured = torch.ones(2)
        with self.assertRaises(TypeError):
            TensorSubgraph(lambda x: x + captured)
        with self.assertRaises(TypeError):
            TensorSubgraph(lambda x, weight=torch.ones(2): x + weight)
        for limit in (0, -1, True, 1.2):
            with self.assertRaises(ValueError):
                TensorSubgraph(square, max_entries=limit)
        with self.assertRaises(ValueError):
            TensorSubgraph(square, backend="eager")


if __name__ == "__main__":
    unittest.main()
