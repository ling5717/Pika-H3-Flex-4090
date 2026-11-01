# Near-equivalent optimizations: scope and validation

This unofficial community extension targets the native ComfyUI quantized Pika H3 Flex path on an RTX 4090. It does not replace the retained upstream standalone implementation. Standalone dependencies are isolated in `requirements-standalone.txt`; the root requirements are comments-only to avoid custom-node-manager installation of the H100 stack. The extension adds no mandatory pip dependencies to a compatible ComfyUI runtime, and is not installed via the standalone `pyproject.toml`/editable package route. See [installation](<INSTALL.md>) and the [Chinese overview](<README_zh.md>).

## What “near-equivalent” means

The target is the **same already-quantized model, adapter, conditioning, sampler and six model evaluations**, with less intermediate allocation and better memory scheduling. It does not mean bitwise equality, equality to BF16/FP32 reference weights, or proven perceptual equivalence.

Do not describe the following as equivalent optimizations: fewer evaluations, shortened clips, reduced dimensions, replacing the adapter, sparse/windowed attention, cross-step prediction or latent/output cache reuse. None belongs in the recommended baseline. Comfy Kitchen INT8 attention is itself a quantized numerical approximation; its selection is part of the reference environment, not evidence of FP32 attention identity.

## 1. Token-wise FFN chunking

An MLP applied independently to each token satisfies, in exact arithmetic:

```text
MLP(concat(x_0, ..., x_n, token_axis))
  = concat(MLP(x_0), ..., MLP(x_n), token_axis)
```

Chunking limits the large gate/up and activation intermediates to a bounded token span (default 16384). The intention is lower peak FFN memory and less offload pressure. It **does not partition the attention sequence**, remove attention interactions or change SwiGLU weights. Full output tensors and other activations still exist; chunking cannot guarantee a fixed VRAM peak.

The historical implementation invoked the native MLP forward through a ModelPatcher injection. Its local CPU comparison covered 128 cases; this is historical evidence, not a substitute for testing the newly packaged implementation. Different matrix shapes can choose different GEMM kernels/reduction orders, so token chunking is not guaranteed bitwise identical. Smaller chunks can increase launch overhead; larger chunks can restore the memory problem. The best chunk size is workload- and backend-dependent.

## 2. Low-peak ordinary linear LoRA

The historical baseline used the native experimental `LoraLoaderBypassModelOnly`, avoiding repeated merging of a large adapter into quantized base weights. Its 208 mapped LoRA groups added approximately 1.822 GiB resident adapter memory in that environment. This does **not** measure the new extension's memory saving.

The new `PikaH3LowPeakLoRA` keeps ordinary linear LoRA as a side branch:

```text
base = base_linear(x)
delta = linear(linear(x, down), up)
output = base + delta * (alpha / rank) * strength
```

Its low-peak inference path reuses the newly allocated delta buffer for scaling and residual addition; it does not mutate the base output or the input. A unit scale needs no separate multiply. Autograd-enabled execution or a mismatched output dtype uses out-of-place math. These are allocation changes around native linear operations, not a new fused INT8 matmul or a new quantization format. This describes the default `mode=low_peak`.

The optional `mode=addmm` combines the second LoRA GEMM, scaling and residual with `torch.addmm` when the base/input dtypes match. That shares a final low-precision rounding boundary instead of the native separately materialized delta/multiply/add sequence. It is a **distinct experimental numerical candidate**, not a rounding-preserving substitute; it needs its own error and output validation and has no new measured speed claim here.

The loader uses native ComfyUI key conversion/mapping and reversible patcher injections. Unknown/unconsumed keys, incomplete pairs, duplicate targets, unsupported adapters or target shapes fail before hook attachment. Supported scope is **plain 2-D linear LoRA**; convolutional LoRA, DoRA, mid/reshape variants and stacked bypass LoRAs are not supported. Always begin from an unpatched model and use either the native bypass loader or this loader, not both. Do not silently discard unmapped weights to make an adapter load.

Cloning a ModelPatcher does not make all underlying modules independent. Injection/ejection and native offload lifecycle correctness matter, particularly when switching between patched and unpatched graphs. A CPU formula test alone cannot certify model switching, quantized module execution or GPU cleanup.

## 3. Optional SwiGLU activation path

The native gate-first convention is:

```text
gate, up = split(x)
half_silu = silu(gate)       # materialized/rounded in input dtype
output = half_silu * up
```

The eager candidate uses an independently owned SiLU result for the in-place multiplication; input/view storage is not overwritten. The optional Triton candidate combines activation work into one kernel while explicitly preserving the intermediate FP16/BF16 cast before multiplication. A single all-FP32 `silu(gate) * up` followed by only one final cast would change that rounding boundary and is not the intended contract.

This is **activation fusion only**, not a fused dual-output projection or a replacement of ComfyUI's INT8 linear kernels. Triton's exponential and arithmetic implementation can still differ from torch numerically. Preserving the cast boundary does not establish bitwise transcendental equality or end-to-end quality parity.

The activation helper provides `eager`, `auto` and `triton` modes. Eligible Triton inputs are contiguous, nonempty CUDA FP16/BF16 tensors with an even final dimension and no required autograd path. Explicit `triton` mode rejects unsupported inputs and raises on import/launch failure; `auto` uses eager semantics for unsupported inputs and can log and fall back on failure until reset. **An auto-mode timing is not evidence that Triton ran**: record `last_backend`. Triton import is lazy and the helper does not retain input/output tensors. Node-exposed options depend on the installed schema.

The FFN node policy defaults to `native` (the original MLP forward/activation). Its optional `triton` mode selects strict Triton activation; `compiled` selects the separately compiled eager tensor activation. These node policy choices are distinct from the activation helper's eager/auto/triton API. Windows Triton packaging/compiler compatibility, different drivers and other GPUs require separate qualification. Do not install an arbitrary Triton distribution or alter system CUDA merely because an optional optimization is unavailable.

## 4. Optional local compilation

Compilation is an experimental, opt-in path, not a requirement for FFN chunking or the low-peak LoRA branch. The `TensorSubgraph` helper is restricted to free tensor functions, without model/module closures, and calls `torch.compile` with `fullgraph=True` and `dynamic=True`. It must not trace Python model loading, CPU/GPU transfer or the whole queue execution. The node's `compiled` mode compiles the eager SwiGLU tensor activation only, not the full FFN or its INT8 linear projections. The wrapper's LRU bounds its metadata keys and compiled-callable references, without retaining input/output/weight tensors; PyTorch-owned compiler caches are outside that bound. `clear()` allows reuse and `close()` releases the callable permanently. ModelPatcher injection/ejection and cleanup remain the integration's responsibility. Dynamic VRAM/offload must not be disabled just to make a graph compile.

Compilation can introduce first-use compilation time, graph breaks, recompilation for new shapes and additional memory. A short or changing-shape workload may be slower. Windows Inductor support depends on the exact torch/compiler/Triton combination. Report first-use and warm timings separately, plus graph breaks/fallbacks. A configured compile flag without a verified compiled path is not a performance result. Keep compilation disabled until the specific configuration passes GPU acceptance and lifecycle tests.

## Optional fused RMSNorm/AdaLN modulation

`PikaH3Optimize.fused_modulation` is an optional boolean, **false by default**, independent of activation mode. The implementation uses the shared Comfy Kitchen `rms_adaln` operation through native ComfyUI casting/offload lifetimes rather than introducing a private RMS kernel. It folds the RMS affine weight into AdaLN scale, uses the model's epsilon, and preserves segment row selection, including scalar and per-token tensor indices. It does not allocate a full expanded token modulation table.

Affine folding changes low-precision rounding. Scalar/tensor-segment operator checks are limited evidence, not comprehensive full-generation qualification. Multiple segments may require an output allocation plus a segment temporary, so fusion is not automatically a peak-memory reduction or a speedup. Unsupported wrapped norms or conflicting block patches must fail rather than silently override another patch. Training/autograd and dtype-promotion cases retain the applicable native path. Keep this option off until its specific configuration passes numerical, end-to-end and lifecycle acceptance.

## VAE prefetch: not implemented

Asynchronous VAE prefetch is a **no-go in this release**, not a hidden optional speed flag. The inspected native runtime does not provide a suitable per-queue cleanup hook, and native VAE decode catches OOM and immediately retries with tiling. A prefetch path without correct cancellation/cleanup ownership could retain tensors across that retry. This extension therefore does not introduce private/global cleanup hooks, replace the native OOM-to-tiling behavior or claim VAE prefetch savings.

## 5. Historical baseline, not new-extension results

Source: the maintainer's pre-release local `PIKA-FAST-README.md`, structured request/cache records and two backend logs. This document transcribes their relevant public facts without publishing machine-specific paths or claiming that those private files are available to readers.

| Property | Historical configuration |
| --- | --- |
| Platform | Windows, Python 3.12.14, RTX 4090 24 GB, 64 GB RAM |
| ComfyUI | 0.43.0, `0dbfd493b4cc604db743c699f0f7ce01348e54a4` |
| Runtime | torch 2.9.1+cu130; torchvision 0.24.1; torchaudio 2.9.1 |
| Backends | comfy-kitchen 0.2.40 CUDA + CK attention; comfy-aimdo 0.6.0 |
| Weights | static pruned INT8 FL2VA; Qwen NVFP4; INT8 video VAE; FP32 audio VAE; Comfy Org Pika LoRA |
| LoRA / FFN | native model-only bypass, 208 groups; historical native FFN chunking, 16384 tokens |
| Sampling | BasicGuider + LCM + released seven sigma values; six evaluations |
| Geometry / seed | 1344 × 768, 362 frames, 24 fps, seed 42 |
| Memory policy | DynamicVRAM, native two-stream async offload, `--reserve-vram 3.5` |

Prompt: “A small paper airplane glides gently through a bright classroom. Soft classroom ambience and a gentle paper flutter, no speech or music.”

Sigmas: `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`. These are the released Comfy workflow values, not an interchangeable generic scheduler.

| Run | Backend full-request execution | Node-result cache |
| --- | ---: | --- |
| Historical FFN16 run, fresh process | 376.127 s | `cached_nodes=[]` |
| Historical FFN16 repeat, fresh process | 378.028 s | `cached_nodes=[]` |
| Historical native bypass + CK, no FFN chunking, one run | 561.938 s | Some loader nodes cached |

The first two times include actual model loading during the request, prompt encoding, sampling, audio/video decoding and MP4 saving, but exclude installation and server startup. OS filesystem and driver caches were not cleared: **fresh process is not strict cold-cache execution**. The no-chunk sample has different cache conditions and only one observation, so it cannot establish a controlled A/B speedup or statistically reliable percentage improvement.

The historical videos decoded all 362 frames with H.264 video and 32 kHz stereo AAC; audio was non-silent. First/middle/last video frames were visually inspected. There was **no subjective audio audition**, comprehensive motion/quality study or broad prompt coverage. Discrete whole-card VRAM polling is not per-process torch allocator peak evidence; there was no Linux `gpu_watch`/`resources_clear` certification.

**No historical timing above was produced with the new low-peak loader, optional fused Triton activation or newly packaged compilation path.** Current new-extension performance/validation status belongs in the central `docs/BENCHMARKS.md` report when published, not in a second changing matrix here. The [measured environment](<../../benchmarks/results/environment.json>) is the package-version source of truth; [Windows constraints](<../../constraints/4090-windows.txt>) constrain the installation recipe.

The [RTX 4090 operator probe](<../../benchmarks/results/operator-probe-4090.json>) publishes tensor shapes, metric definitions and repeated GPU measurements. Its tested low-peak LoRA cases matched native outputs exactly in BF16/FP16; tested Triton SwiGLU matched in BF16 and had small FP16 error, while compiled activation changed rounding more noticeably. Those observations are scoped to the recorded inputs, not a claim of exactness for every dtype/shape or a universal error bound. Compilation first-use cost, backend identity and output-validation scope remain material. Full-video decoded RGB agreement on a tested seed is also distinct from latent/audio equality and broad quality coverage. Small single-run wall-time differences must not be advertised as statistically significant improvements.

SOL/SLA/VSA sparse or predictive attention is not part of the recommendation. Earlier SOL experiments were aborted without a successful video. Do not report those attempts as successful speed tests, combine unfinished options into a claimed result, promise seven minutes for arbitrary prompts, or claim a global optimum.

## 6. Acceptance checklist for each candidate

1. **Freeze the reference:** exact model/LoRA hashes, ComfyUI and extension commits, versions, backend, prompt, seed, sigmas, sampler, shape, audio path and launch flags.
2. **Numerical unit checks:** compare against the native operation with representative shapes/dtypes, scale/alpha values, noncontiguous inputs and supported fallbacks. Report absolute/relative errors and nonfinite values; tolerances must be declared, not retrofitted to hide failures.
3. **GPU microbenchmarks:** synchronize CUDA; distinguish warm-up/compilation from measured work; verify selected backend; capture allocator and whole-card memory with their different scopes. Microkernel timings do not establish full-request speed.
4. **End-to-end A/B:** change one factor at a time, use multiple fresh-process requests under matching cache conditions, and publish both failures and successful runs. Check `cached_nodes`, not only a seed or changed filename.
5. **Outputs:** verify full frame count/duration, resolution, stream codecs, audio sample rate/channels and non-silence. Compare images/latents where available and inspect temporal behavior. Subjective soundtrack/synchronization assessment is separate and must not be inferred from non-silence.
6. **Lifecycle:** switch graphs/models and confirm injection ejection, original behavior restoration, no retained tensors or gradual VRAM growth. Record abnormal cleanup rather than claiming resources cleared.

CPU-only tests, an import check, one successful frame or a single full video do not prove broad perceptual equivalence. No universal image/audio error threshold or quality guarantee is established by this release.

## License boundaries

New extension components carry **GPL-3.0-only** notices and integrate with GPL-licensed ComfyUI. Retained upstream source remains [Apache-2.0](<../../LICENSE-CODE>); neither attribution nor license is overwritten. Model components and derivatives/adapters remain subject to the [MiniMax H3 Community License](<../../licenses/MiniMax-H3-LICENSE.txt>), including territory/use/commercial restrictions. Code licenses do not relicense weights. This project is unaffiliated with the named model/framework vendors; descriptive names imply no endorsement or trademark grant.
