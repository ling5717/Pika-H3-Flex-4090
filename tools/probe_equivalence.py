# SPDX-License-Identifier: GPL-3.0-only
"""Explicit CUDA numerical/microbenchmark probe; no model weights are downloaded."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy_extension.lora import plain_lora_forward
from comfy_extension.activation import swiglu_eager, SwiGLUActivation
from comfy_extension.compilation import TensorSubgraph


def metrics(actual, reference):
    sq = ref_sq = dot = act_sq = 0.0
    max_error = 0.0
    for start in range(0, actual.shape[0], 512):
        a = actual[start:start + 512].float()
        b = reference[start:start + 512].float()
        if not bool(torch.isfinite(a).all()):
            raise ValueError("Nonfinite output")
        sq += float((a - b).square().sum())
        ref_sq += float(b.square().sum())
        act_sq += float(a.square().sum())
        dot += float((a * b).sum())
        max_error = max(max_error, float((a - b).abs().max()))
    return {"nrmse": (sq / max(ref_sq, 1e-30)) ** .5,
            "cosine": dot / max((act_sq * ref_sq) ** .5, 1e-30), "max_abs": max_error}


def milliseconds(fn, repeats):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        out = fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
        del out
    return sorted(times)[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda"], required=True)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(42)
    report = {"torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "rows": args.rows,
              "scope": "Synthetic real FFN shapes; not full-generation quality or latency.", "tests": []}
    for dtype in (torch.bfloat16, torch.float16):
        with torch.inference_mode():
            x = torch.randn((args.rows, 5376), device="cuda", dtype=dtype)
            base = torch.randn((args.rows, 28672), device="cuda", dtype=dtype)
            down = torch.randn((128, 5376), device="cuda", dtype=dtype) * .02
            up = torch.randn((28672, 128), device="cuda", dtype=dtype) * .02
            def reference():
                return base + F.linear(F.linear(x, down), up) * 1.0
            ref = reference()
            for mode in ("low_peak", "addmm"):
                fn = lambda: plain_lora_forward(x, base, up, down, mode=mode)
                out = fn()
                error = metrics(out, ref)
                if error["nrmse"] > .004 or error["cosine"] < .99998:
                    raise ValueError(f"LoRA {mode} exceeded tolerance: {error}")
                report["tests"].append({"kind": "lora", "mode": mode, "dtype": str(dtype),
                    **error, "median_cuda_ms": milliseconds(fn, args.repeats)})
                print(report["tests"][-1], flush=True)
                del out
            report["tests"].append({"kind": "lora", "mode": "reference", "dtype": str(dtype),
                                    "median_cuda_ms": milliseconds(reference, args.repeats)})
            del x, base, down, up, ref
            torch.cuda.empty_cache()
            packed = torch.randn((args.rows, 28672), device="cuda", dtype=dtype) * 3
            ref = swiglu_eager(packed)
            candidates = [("triton", SwiGLUActivation("triton"))]
            if args.compile:
                candidates.append(("compiled", TensorSubgraph(swiglu_eager, enabled=True)))
            for name, activation in candidates:
                try:
                    t = time.monotonic()
                    out = activation(packed)
                    torch.cuda.synchronize()
                    first = time.monotonic() - t
                    error = metrics(out, ref)
                    if error["nrmse"] > .003 or error["cosine"] < .99998:
                        raise ValueError(f"Activation {name} exceeded tolerance: {error}")
                    report["tests"].append({"kind": "activation", "mode": name, "dtype": str(dtype),
                        **error, "first_call_seconds": first,
                        "median_cuda_ms": milliseconds(lambda: activation(packed), args.repeats)})
                except Exception as exc:
                    if name == "triton":
                        raise
                    report["tests"].append({"kind": "activation", "mode": name,
                        "dtype": str(dtype), "status": "unavailable", "error_type": type(exc).__name__})
                finally:
                    activation.clear()
                print(report["tests"][-1], flush=True)
            report["tests"].append({"kind": "activation", "mode": "eager", "dtype": str(dtype),
                "median_cuda_ms": milliseconds(lambda: swiglu_eager(packed), args.repeats)})
            del packed, ref, candidates
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    print("PASS", args.output.name, flush=True)


if __name__ == "__main__":
    main()
