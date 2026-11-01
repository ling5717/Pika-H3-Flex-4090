# RTX 4090 measurements and acceptance

## Scope

The main performance workload is **1344×768, 362 frames, 24 fps (15.083 s), six Pika model evaluations, stereo audio**, seed 42 and the public paper-airplane prompt in the API workflow. BasicGuider, LCM, all seven manual sigmas, model artifacts and full decoding are retained. No sparse attention, predicted/skipped steps or smaller output is used in the recommended preset.

[Exact runtime versions](../benchmarks/results/environment.json) · [Model identities](../configs/models.json) · [Raw structured development summaries and media comparisons](../benchmarks/results/development-4090.json) · [Operator probe](../benchmarks/results/operator-probe-4090.json) · [RMS/AdaLN probe](../benchmarks/results/modulation-probe-4090.json)

Machine: Windows 11, Python 3.12.14, RTX 4090 24 GB, 64 GB RAM, driver 621.01, PyTorch2.9.1+cu130, Comfy Kitchen0.2.40, ComfyUI backend `0dbfd493b4cc604db743c699f0f7ce01348e54a4`. This is an interactive desktop, not a resource-isolated laboratory. Other hardware/platforms are not qualified by CPU CI.

## Historical and development runs

| Configuration | Complete request seconds | Decoded RGB/PCM vs local reference | Recommendation |
|---|---:|---|---|
| Earlier native bypass + FFN16384 + CK, first request | 376.127 | Historical baseline | Retained reference |
| Earlier baseline, independent process repeat | 378.028 | Same reference sample | Retained reference |
| New low-peak LoRA + FFN16384/native activation | 370.639 | **All 362 RGB frames and decoded PCM exactly equal** | Conservative default |
| Low-peak + Triton SwiGLU | 370.109 | **All RGB frames and decoded PCM exactly equal** | Optional; no significant total gain established |
| `addmm` LoRA + Triton | 370.739 | Changed RGB/PCM; PSNR14.71 dB vs low-peak | Research only; not reference-preserving |
| Low-peak + compiled SwiGLU | 376.790 | Changed RGB/PCM; PSNR14.55 dB | Research only; no cold-request benefit shown |
| Low-peak + fused RMS/AdaLN | 370.366 | Changed RGB/PCM; PSNR14.16 dB | Research only; no total gain established |
| Low-peak + FFN32768/native | 374.285 | Changed RGB/PCM; PSNR14.07 dB | Keep16384; no gain shown |

These are one request per new candidate, not confidence intervals or matched repeated trials. The approximately0.53s difference between low-peak/native and Triton is within ordinary run-to-run uncertainty. Do not claim a large overall speedup by summing individual operator speedup percentages.

The earlier no-FFN-chunk request was561.938s and already had loader-cache hits, so it is not a strict paired comparison. It motivated the activation-memory fix; it is not a marketing baseline for every new feature.

### Small numerical error is not the same as an identical generated clip

At the measured synthetic FFN shape, low-peak LoRA was bit-exact for BF16/FP16; the Triton activation was exact for the BF16 probe and had tiny FP16 error. `addmm`, Inductor and folded RMS/AdaLN had small **local** errors, but those propagated to substantial changes in the fixed-seed generated pixels and audio waveform. They still generated decodable classroom/airplane clips; this does **not** establish reference preservation or perceptual superiority.

Consequently **only the conservative defaults and the separately qualified Triton case are recommended for the requested near-equivalent/reference-preserving goal**. The other implemented modes remain explicit research options, disabled by default, with their negative results retained. FFN batching is tokenwise mathematically equivalent, but changing GEMM shapes can change floating-point reductions and downstream generation.16384 is the tested reference setting, not an assertion that any chunk size is pixel-identical.

PSNR/MAE here compare decoded8-bit RGB from the same local seed/artifacts. Audio comparison is aligned decoded PCM. Neither metric is latent equivalence, LPIPS, subjective quality or an assertion that an uncorrelated official showcase should match pixels. Exact equality on one input is not a proof for every prompt.

## Measurement protocol

1. Each candidate was submitted to a fresh dedicated service process. All listed candidate histories report `cached_nodes=[]`; sampler/decoder result caching is invalid as a generation speed measurement.
2. The reported complete time is Comfy `execution_start→execution_success`: request loading, prompt encoding, six steps, audio/video decoding and MP4 saving. Server launch and dependency installation are excluded. The batch tool additionally records startup time.
3. OS file/page cache, CUDA driver and Triton/Inductor disk caches were not cleared. A fresh process is **not** a fresh machine or a guarantee of cold compilation. The separate operator probe measured a first BF16 compile call around24.42s; do not hide that cost in a hot-kernel result.
4. Whole-card VRAM is sampled approximately every5s by `nvidia-smi`; desktop/other GPU activity is included. Samples can miss instantaneous peaks and are not per-process allocated/reserved memory or a `resources_clear` certificate.
5. All T2V outputs were fully decoded:362 video frames, H264,24fps and32kHz stereo AAC. Ref2VA outputs have their separately stated 243 frames. Non-silence and finite PCM were checked; first/middle/last contact sheets were inspected. No claim of subjective listening, full perceptual rating or broad prompt coverage is made.
6. Development runs predate final lifecycle safety fixes. The final qualification below uses a recorded inference-code commit. Those safety fixes reject unsupported bypass stacking and preserve rollback/ownership; they do not change the numerical default path.

## Final source qualification

The tested inference source has aggregate SHA-256 `926e262019aabfe2de2087db1cf1ba9cf3f6e273293a2eaf52685e50e41dd8e1` (UTF-8 relative path, NUL, file bytes, NUL, for root `__init__.py` then lexically sorted `comfy_extension/*.py`). The original test commit is an unpublished local snapshot; this digest identifies the actual code independently of Git history. A new-process repeat of the main 362-frame workflow completed in **379.497 seconds**, without result-cache hits. All decoded RGB frames and audio PCM were exactly equal to the earlier local native-bypass reference on this prompt and seed. See [the sanitized qualification receipt](../benchmarks/results/qualified-4090.json). This is not proof of bitwise equality for all inputs.

## Official showcase comparison

The [official balcony case](4090/OFFICIAL_SAMPLE.md) is **328 frames**, not362. It is a prompt-matched qualitative reference with undisclosed official seed/checkpoint/preprocessing. Its runtime must not be mixed into the15-second table or used to claim faster15-second generation. A local stock-bypass/control versus optimized run checks the optimization itself under matched local inputs; the official clip provides a separate visual/audio reference.

## Ref2VA multimodal reference workload

Pika's Record-shop dance sample was tested separately with two reference images and one reference video. The video contained an embedded audio track. Each request used a fresh Comfy process with no cached node results and produced 243 frames at 24fps (10.125 seconds):

| Output | Audio route | Node execution seconds | Whole-card sampled maximum |
|---|---|---:|---:|
| 1344×768 | Paired video soundtrack | 397.957 | 20,693 MiB |
| 864×480 | Paired video soundtrack | 190.218 | 21,192 MiB |
| 864×480 | Separate WAV derived from the same video | 190.815 | 21,318 MiB |

This is a different workload from the 362-frame T2V results above and cannot be directly compared as a speedup. All outputs decoded fully with non-silent 32kHz stereo audio; subjective listening and reference-ablation were not performed. The published character reference used locally depicts a human man, whereas the official showcase depicts a raccoon; the earlier claim that the local output lost a raccoon face/tail was incorrect. This is not a same-character identity comparison. Official render seed/checkpoint and actual render-time inputs are unknown; the Comfy reference node trims the 241-frame source video to 226 frames. See [Ref2VA demonstration, exact model links and reproduction](4090/REF2VA.md) and [sanitized measurements](../benchmarks/results/ref2va-4090.json).

## Reproduce

Use the installation guide, exact artifacts and public API workflow. `tools/run_isolated_bench.py --help` accepts explicit Python/ComfyUI/model-config paths and starts/stops only owned process trees. Keep raw logs private. The benchmark driver writes a sanitized summary plus ignored raw history; validate every output with `tools/validate_video.py`.

The public candidate graphs are in `benchmarks/cases/`. They retain the exact bytes hashed in development receipts. They are reproducibility artifacts, not all approved presets. A later safety check may reject unsupported stacking even when an old experimental script attempted it. Do not install the standalone H100 requirements for these ComfyUI measurements.
