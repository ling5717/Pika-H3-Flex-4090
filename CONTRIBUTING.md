# Contributing

This repository preserves upstream Pika standalone code and adds an independent ComfyUI integration. Please keep changes in the smallest owning component and preserve upstream notices.

## Development

Use Python 3.12 and an isolated environment. Install a suitable CPU PyTorch build for unit tests, or the documented CUDA stack for GPU work. The root standalone `pyproject.toml` is not an extension installer.

```bash
python -m unittest discover -s tests -v
python tools/check_workflows.py
git diff --check
```

Set `COMFYUI_PATH` to run native CPU integration tests. Without it those tests explicitly skip; they must not be reported as passed integration/GPU qualification. New nodes use V3 schemas and native ModelPatcher injection/cleanup. No model tensors, graph outputs or weights in global caches. Preserve dtype, casting/offload lifetime, aliases, failure rollback and cancellation behavior. Optional backends must not silently masquerade as the candidate being benchmarked.

## Performance evidence

Keep official model artifacts, prompt, seed, 1344×768/362-frame geometry, 24 fps, six NFE, manual sigmas and audio enabled when comparing the published workload. Change one factor when attributing gains. Report fresh/warm process, node cache hits, startup/compilation boundaries, repetitions and sampled-vs-allocator memory scope. Never reuse final latents/videos as a generation benchmark.

GPU microbenchmarks are not end-to-end results. Small operator errors can propagate through denoising: report local NRMSE/cosine plus full decode, reference comparison and perceptual/audio limitations. Do not claim every prompt is faster or mathematically bitwise equivalent from one example.

Before publishing results, inspect summaries and media for private prompts, faces/references, usernames, paths and credentials. Never commit raw histories, environment files, model weights, wheels or caches. Generated examples must be labelled AI-generated.

## Pull requests

Describe the problem, changed behavior, tests, numerical differences and measured benefit. Keep stable defaults conservative. A candidate that is unsafe, slower or unqualified stays off or is omitted; a theoretical optimization is not a benchmark result. Use short direct commit subjects (`Add ...`, `Fix ...`, `Document ...`). Follow [the license scopes](THIRD_PARTY_NOTICES.md); do not replace original licenses with the extension license.
