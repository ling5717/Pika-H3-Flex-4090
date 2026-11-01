# SPDX-License-Identifier: GPL-3.0-only
"""Explicit CUDA probe of shared CK RMS/AdaLN fusion at packed H3 token shapes."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import comfy_kitchen as ck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comfy_extension.modulation import _fused_segments
from tools.probe_equivalence import metrics, milliseconds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(42)
    rows, width = 109089, 5376
    segments = [(0, 629, 0), (629, 1233, 1), (1233, rows, 2)]
    report = {"gpu": torch.cuda.get_device_name(0), "shape": [rows, width],
              "scope": "Synthetic packed tokens, shared RMS/AdaLN op; not full model fidelity or latency", "tests": []}
    for dtype in (torch.bfloat16, torch.float16):
        with torch.inference_mode():
            x = torch.randn((rows, width), device="cuda", dtype=dtype)
            weight = 1 + torch.randn((width,), device="cuda", dtype=dtype) * .03
            shift = torch.randn((3, width), device="cuda", dtype=dtype) * .1
            scale = torch.randn_like(shift) * .1
            def ref_call():
                out = F.rms_norm(x, (width,), weight, 1e-5)
                for first, last, row in segments:
                    out[first:last].mul_(1 + scale[row]).add_(shift[row])
                return out
            ref = ref_call()
            def candidate():
                return _fused_segments(x, weight, shift, scale, segments, 1e-5, ck.rms_adaln)
            out = candidate()
            torch.cuda.synchronize()
            error = metrics(out, ref)
            report["tests"].append({"dtype": str(dtype), **error,
                "reference_median_cuda_ms": milliseconds(ref_call, 5),
                "fused_median_cuda_ms": milliseconds(candidate, 5),
                "within_tolerance": error["nrmse"] <= .005 and error["cosine"] >= .99995})
            print(report["tests"][-1], flush=True)
            del x, weight, scale, shift, out, ref
            torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf8")
    if not all(t["within_tolerance"] for t in report["tests"]):
        raise SystemExit("Fusion exceeded numerical tolerance")


if __name__ == "__main__":
    main()
