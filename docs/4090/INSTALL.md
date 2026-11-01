# RTX 4090 installation: independent ComfyUI runtime

This is an **unofficial community integration**, not a Pika, MiniMax or Comfy Org product. Read the [Chinese overview](<README_zh.md>) and [technical notes](<OPTIMIZATIONS.md>) before changing a working installation.

The retained upstream standalone application targets a different deployment (representative H100 80 GB tests). Its original dependencies are preserved separately in `requirements-standalone.txt`; the root `requirements.txt` is comments-only so custom-node managers do not automatically install the H100 stack. The ComfyUI extension has no additional mandatory pip dependencies beyond its compatible ComfyUI runtime. **Do not install the standalone requirements into the 4090 environment, run its standalone downloader, or automatically acquire the approximately 210 GB BF16 base snapshot.** Do not `pip install -e .`: the retained `pyproject.toml` and `src` describe the standalone application, not installation of these custom nodes. The Docker/Conda standalone recipes also use the separate requirements. The route below uses native ComfyUI with selected quantized components. No weights are bundled here.

## 1. Requirements and qualification boundary

Historical baseline machine: Windows, RTX 4090 24 GB, 64 GB system RAM, Python 3.12.14. A fast SSD and additional free space for models, temporary downloads, Python packages and output are required; 64 GB is an observed configuration, not a guaranteed minimum. A recent NVIDIA driver compatible with PyTorch's CUDA 13.0 runtime is required. An optional compiler/Triton toolchain may require additional system software.

| Component | Historical baseline |
| --- | --- |
| ComfyUI | 0.43.0, commit `0dbfd493b4cc604db743c699f0f7ce01348e54a4` |
| PyTorch / torchvision / torchaudio | `2.9.1+cu130` / `0.24.1` / `2.9.1` |
| comfy-kitchen / comfy-aimdo | `0.2.40` / `0.6.0` |
| Frontend / workflow templates | `1.57.6` / `0.11.100` |

The full measured Windows environment is recorded in [environment.json](<../../benchmarks/results/environment.json>), including the exact CUDA package suffixes, frontend, PyAV and optional Triton distribution. Use [4090-windows.txt](<../../constraints/4090-windows.txt>) as pip constraints for ComfyUI's requirements; constraints bound resolution but do not themselves install every listed package. They are a measured Windows stack, not a complete universal lockfile or Linux certification. These baseline pins do not imply acceptance of every new extension option. Linux installation commands below are provided for portability, **not a Linux benchmark or compatibility certification**. CPU-only execution, other GPUs, macOS, ROCm, DirectML, maximum reference counts and other ComfyUI versions have not been qualified here.

## 2. Create a separate runtime

Commands are manual recipes for the reader; choose one environment, not all three. Run them in a new working directory. The ComfyUI requirements in these commands belong to **ComfyUI**, not this repository. Review dependency resolution before proceeding: installing a newer transitive dependency can change a benchmark environment.

### Windows PowerShell, standard Python

```powershell
git clone https://github.com/Comfy-Org/ComfyUI.git ComfyUI-4090
Set-Location ComfyUI-4090
git checkout 0dbfd493b4cc604db743c699f0f7ce01348e54a4
py -3.12 -m venv .venv
$Python = Join-Path $PWD '.venv\Scripts\python.exe'
git clone https://github.com/ling5717/Pika-H3-Flex-4090.git custom_nodes/Pika-H3-Flex-4090
$Constraints = 'custom_nodes/Pika-H3-Flex-4090/constraints/4090-windows.txt'
& $Python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
& $Python -m pip install -r requirements.txt -c $Constraints
& $Python -m pip install comfy-kitchen==0.2.40 comfy-aimdo==0.6.0 -c $Constraints
& $Python -m pip check
& $Python -c "import torch, importlib.metadata as m; print(torch.__version__); print({k:m.version(k) for k in ['comfy-kitchen','comfy-aimdo']}); print(torch.cuda.is_available())"
```

### Windows portable ComfyUI

Use a **separate** portable distribution. Its embedded Python must be compatible with the desired wheels; a different embedded Python/runtime is not the baseline. From the portable distribution root:

```powershell
$Python = Join-Path $PWD 'python_embeded\python.exe'
git clone https://github.com/ling5717/Pika-H3-Flex-4090.git ComfyUI/custom_nodes/Pika-H3-Flex-4090
& $Python -s -c "import sys, torch; print(sys.version); print(torch.__version__)"
& $Python -s -m pip check
& $Python -s ComfyUI/main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --offline --use-ck-attention --reserve-vram 3.5
```

Do not blindly replace an embedded runtime's torch packages. If its versions differ, either qualify that environment independently or use the standard Python recipe. Installing the custom node means cloning the **whole repository** under `custom_nodes`; the root entry point loads `comfy_extension`. Do not copy that package alone or install the upstream standalone package with editable pip.

### POSIX shell, independent Python 3.12 environment

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git ComfyUI-4090
cd ComfyUI-4090
git checkout 0dbfd493b4cc604db743c699f0f7ce01348e54a4
python3.12 -m venv .venv
PYTHON="$PWD/.venv/bin/python"
git clone https://github.com/ling5717/Pika-H3-Flex-4090.git custom_nodes/Pika-H3-Flex-4090
CONSTRAINTS='custom_nodes/Pika-H3-Flex-4090/constraints/4090-windows.txt'
"$PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
"$PYTHON" -m pip install -r requirements.txt -c "$CONSTRAINTS"
"$PYTHON" -m pip install comfy-kitchen==0.2.40 comfy-aimdo==0.6.0 -c "$CONSTRAINTS"
"$PYTHON" -m pip check
"$PYTHON" main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --offline --use-ck-attention --reserve-vram 3.5
```

Pin and record the extension checkout commit for any published run; the `main` branch can change. Do not modify your unrelated ComfyUI install to match this recipe.

## 3. Acquire only the selected model components

First review the model license and access conditions. Select the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main); use the repository's own per-file download controls, not a full snapshot download. Public source filenames are listed for identification; the files themselves are not in this repository.

| ComfyUI model folder | Selected component |
| --- | --- |
| `models/diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `models/loras` | `pika_h3_flex_fl2va_v1_bf16.safetensors` |

This is a static prequantized/pruned INT8 model, NVFP4 text encoder, INT8 video VAE, FP32 audio VAE and the Comfy Org Pika adapter. It is not a claim of equivalence to an unquantized BF16 model. Record source revision and SHA-256 for every local artifact; filenames alone do not establish identity. Model availability, download authorization and upstream revisions may change.

An existing model store can be referenced through ComfyUI's `--extra-model-paths-config` instead of copying weights. Keep that YAML private when it contains machine-specific paths; never publish credentials or personal paths.

## 4. Launch and verify the actual backend

From the standard Windows installation directory:

```powershell
& $Python main.py --listen 127.0.0.1 --port 8189 --disable-auto-launch --offline --use-ck-attention --reserve-vram 3.5
```

Open `http://127.0.0.1:8189`. `--offline` is for startup after manual downloads; it does not install dependencies or fetch weights. Keep the listener local unless you deliberately configure authentication and network security.

Read the server log, not just the CLI flags. The historical successful configuration reported:

- `comfy_kitchen backend cuda` available and not disabled;
- `Using Comfy Kitchen attention`;
- async weight offloading with two streams and DynamicVRAM enabled;
- the expected torch, ComfyUI, comfy-kitchen and comfy-aimdo versions.

The old log had Comfy Kitchen's Triton backend disabled while its CUDA backend was active. That does not certify the extension's separate optional Triton SwiGLU path. Missing CUDA capabilities or backend fallback invalidate a like-for-like comparison. Do not add `--highvram`, disable dynamic/offload behavior or stack experimental flags without measuring their consequences.

## 5. Workflow setup and safe first run

Start with ComfyUI's official Pika H3 Flex FL2VA/text-to-video template and confirm the selected model files. Preserve native text encoding, audio/video conditioning and decoders. The historical long test used:

- `BasicGuider`, `KSamplerSelect` set to `lcm`, and `SamplerCustomAdvanced`;
- `ManualSigmas`: `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`;
- seed 42; 1344 × 768; 362 frames at 24 fps; six model evaluations;
- model-only bypass LoRA at strength 1.0 and token FFN chunk size 16384.

First qualify a short supported resolution/duration before attempting the long case. The long case is about 15.083 seconds of output, not exactly 15.000 seconds. A generic six-step Euler workflow is not equivalent to the released schedule.

For the new extension, use `PikaH3LowPeakLoRA` **instead of**, not on top of, the native bypass LoRA loader; connect its MODEL output through `PikaH3Optimize` to the guider. Both experimental nodes return one MODEL output:

| Node | Inputs and starting settings |
| --- | --- |
| `PikaH3LowPeakLoRA` | `model`, `lora_name`, `strength=1.0`, `mode=low_peak` (alternative: `addmm`) |
| `PikaH3Optimize` | `model`, `chunk_size=16384` (combo strings: 8192/16384/32768), `activation=native` (alternatives: triton/compiled), optional `fused_modulation=false` |

`native` retains the original MLP activation and keeps compilation off. `addmm` is separately experimental: it combines the second LoRA GEMM and residual, changing low-precision rounding. Leave it off for initial validation. Also leave `fused_modulation` false: this optional shared Comfy Kitchen RMSNorm/AdaLN path changes rounding and has separate validation requirements. Optional Triton and local compilation remain experimental; use the central benchmark report for current acceptance status rather than inferring a speedup from historical timings.

To reproduce the historical baseline, use the original native `LoraLoaderBypassModelOnly` and the historical FFN behavior described in [technical notes](<OPTIMIZATIONS.md>), not the new low-peak node. A successful import or CPU test does not establish full model/video correctness.

## 6. Troubleshooting and reproducibility

- **Unknown nodes:** confirm the full repository is directly under this runtime's `custom_nodes`, restart the server and inspect import errors. Do not install the standalone requirements as a fix.
- **Unsupported LoRA or incomplete mapping:** use the matching plain linear adapter. Do not suppress errors, drop unmatched keys or stack loaders.
- **OOM:** close other GPU workloads, verify offload/VRAM reservation and start smaller. Different chunk sizes can affect throughput. Lowering resolution, duration or steps is a different workload and must be labelled as such.
- **Missing Triton/compiler:** keep eager mode and compilation off. Windows Triton and Inductor availability depend on the exact Python, torch and toolchain; no automatic installation is promised.
- **Unexpectedly tiny execution time:** inspect ComfyUI cache events. Re-saving cached latents or video is not a generation benchmark.
- **Performance report:** record extension and ComfyUI commits, all versions, artifact hashes, flags, graph, prompt/seed, actual backend, complete timing scope and cache state. Review logs for private prompts and paths before publication.

New performance and full-generation comparisons are maintained centrally in `docs/BENCHMARKS.md` when published; this guide deliberately does not duplicate that changing matrix. The [operator probe](<../../benchmarks/results/operator-probe-4090.json>) records scoped GPU tensor tests and must not be read as universal model-quality or full-request performance evidence.

## Licensing

The newly added ComfyUI integration components carry **GPL-3.0-only** notices; ComfyUI has its own GPL terms. Retained upstream inference code remains under [Apache-2.0](<../../LICENSE-CODE>). The base model and model derivatives/adapters are governed separately by the [MiniMax H3 Community License](<../../licenses/MiniMax-H3-LICENSE.txt>), not by GPL or Apache permission to use weights. The model license includes territorial exclusions (EU, UK, Republic of Korea and USA), use restrictions and commercial conditions; read the complete agreement and obtain separate authorization where required. These notes do not replace any license or grant trademark rights.
