# Ref2VA on one RTX 4090: Record-shop dance

This is an **unofficial community validation**, not an official Pika benchmark or an exact reproduction of its showcase. [Watch Pika's official demonstration](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/motion-transfer.mp4); it is not our output. The [official ComfyUI Ref2VA template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_ref2va.json) provides the model selection and sampling schedule, but its example assets and prompt are different from this Record-shop dance case.

## Measured local results

Windows, one RTX 4090 (24,564 MiB), approximately 64 GiB RAM, ComfyUI commit `0dbfd493b4cc604db743c699f0f7ce01348e54a4`, PyTorch `2.9.1+cu130`, Comfy Kitchen `0.2.40`, CK attention. Each row is one run, not an average. All outputs are 243 frames at 24 fps (10.125 seconds).

| Case | Input audio route | Output | Node execution | Maximum observed whole-card memory |
| --- | --- | --- | ---: | ---: |
| 768p | Embedded audio from reference video | 1344 × 768 | 397.957 s | 20,693 MiB |
| 480p | Embedded audio from reference video | 864 × 480 | 190.218 s | 21,192 MiB |
| 480p independent-audio interface test | Separate WAV extracted from that same video | 864 × 480 | 190.815 s | 21,318 MiB |

Each request used a fresh server process and reported `execution_cached=[]`: no node results were cached. Timing includes model loading, reference and audio encoding, text encoding, six sampling evaluations, video and audio VAE decoding and MP4 saving; it excludes server startup and installation. OS file and driver caches may already have been warm. VRAM is the largest whole-GPU `nvidia-smi` sample at approximately five-second intervals, including other processes such as the desktop; it is not an allocator peak and can miss transient peaks. These single measurements do not establish a general speedup or a worst-case memory bound.

All three MP4s decoded to 243 H.264 frames with non-silent 32 kHz stereo AAC. Visual inspection was limited to selected frames, not a full motion evaluation; neither soundtrack was subjectively listened to. Non-silence does not prove correct audio content or reference influence.

**Correction:** the published Picture 1 used locally is a human male character sheet, and Picture 2 is a record-shop interior. The local output depicts a human in a similar shop. The official showcase depicts a raccoon, but its actual render-time character input is undisclosed. Our earlier statement that the local result lost the showcase raccoon's face and tail was incorrect: no raccoon character image was provided to the local graph. Do not use this pair as a like-for-like identity-fidelity comparison. Image order and current file hashes have been checked; no reference ablation or broad input coverage was established by the original run.

[Watch the labelled local 4090 output](https://github.com/ling5717/Pika-H3-Flex-4090/raw/main/assets/demos/ref2va-4090-labeled.mp4). The repository owner confirmed separate authorization for public display of this local result; this does not grant viewers additional rights. The [MiniMax H3 Community License](<../../licenses/MiniMax-H3-LICENSE.txt>) restricts use in excluded territories, including the EU, UK, Republic of Korea and USA, including use and display of output; review its complete terms and obtain any required authorization before downloading, generating, using or displaying outputs. The official demo above and the official input links below are provided for reference, not as a license grant.

## Exact model components

The two Ref2VA-specific files are the exact artifacts used locally; their full-file SHA-256 values and sizes also match [Comfy-Org's ModelScope metadata](https://www.modelscope.cn/api/v1/models/Comfy-Org/MiniMax-H3/repo/files?Revision=master&Recursive=true). These Hugging Face links are the official template's `main` links; `main` can change, so verify the downloaded bytes and record the resolved upstream revision.

| ComfyUI folder and download | Bytes | SHA-256 |
| --- | ---: | --- |
| `models/diffusion_models/` [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | 20,970,379,616 | `9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779` |
| `models/loras/` [pika_h3_flex_ref2va_v1_bf16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/pika_h3_flex_ref2va_v1_bf16.safetensors) | 1,956,172,304 | `de3745ceb7d8d5c20898dd8fa3a05f1c0b00089ccd55297b2a8bab95c766b0e0` |

The other three files are shared with the existing FL2VA setup: [Qwen NVFP4 text encoder](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors), [INT8 video VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors), and [FP32 audio VAE](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors). All five Ref2VA paths, byte counts and hashes are in [the Ref2VA model manifest](<../../configs/models-ref2va.json>); do not substitute the FL2VA base or LoRA. No full BF16 base is required. Place each file under the corresponding `models` subdirectory or configure an existing model store. Model terms and access conditions apply independently of the hash.

## Inputs and conditioning

The [showcase manifest](<../../assets/showcases/manifest.json>) identifies the source assets and prompt. Its input list is video first, then two images; the graph's modality-specific order is **Picture 1 = character**, **Picture 2 = environment**, **Video 1 = motion**, with the video's own audio connected as `ref_video_audios.ref_video_audio_0`. The derived audio graph instead connects the WAV as `ref_audios.ref_audio_0` and does not additionally connect the video-audio port.

| Official input, fixed asset revision | Local graph filename in ComfyUI `input/` | SHA-256 |
| --- | --- | --- |
| [motion-transfer-input-1.mp4](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/motion-transfer-input-1.mp4) | `pika_official_motion_ref.mp4` | `fd5ef02c10aaa096cf558fdc641ac438a4779e0b5a8affd486a9ca50ed45ebef` |
| [motion-transfer-input-2.png](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/motion-transfer-input-2.png) | `pika_official_character_ref.png` | `05aaa2918f3af99cf202d2047fa111763f2c0456fe20c2c22445719d2017b0a7` |
| [motion-transfer-input-3.png](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/motion-transfer-input-3.png) | `pika_official_environment_ref.png` | `f42a00e91f1c903c9826042edd6a9d444d9209fa417f6733ae255e023779f805` |

The original prompt's `@Image1`, `@Image2`, and `@Video1` were translated to ComfyUI's `<Picture 1>`, `<Picture 2>`, and `<Video 1>`; otherwise the prompt is unchanged. The reference video is 854 × 480, 241 frames at 24 fps with audio. The reference node truncates its video frames to 226 (`17 × 13 + 5`); the paired audio remains the full input. `ref_image_size=match` and seed 42 were local choices. The official showcase's seed, generating model/checkpoint revision and precision, and preprocessing are unknown: the fixed **asset** revision is not a generating model revision. Exact visual or audio reproduction is not promised.

## Reproduction steps

1. Review applicable licenses and terms, then prepare the separate ComfyUI environment and custom-node checkout using the [installation guide](<INSTALL.md>). Use its tested versions and CK attention startup flags; record the extension commit and verify the actual backend. Do not install the standalone H100 stack or overwrite an unrelated runtime.
2. Download the two Ref2VA files above and the three shared files into their indicated ComfyUI model folders. From the ComfyUI root, verify all five using `python custom_nodes/Pika-H3-Flex-4090/tools/verify_models.py --model-root models --task ref2va`. This reads already-downloaded files; it does not download them. Do not substitute the FL2VA model or adapter.
3. From the ComfyUI root, download the three official inputs with the names required by the graphs. Example PowerShell commands (downloads require network access):

   ```powershell
   $base = 'https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases'
   Invoke-WebRequest "$base/motion-transfer-input-1.mp4" -OutFile 'input/pika_official_motion_ref.mp4'
   Invoke-WebRequest "$base/motion-transfer-input-2.png" -OutFile 'input/pika_official_character_ref.png'
   Invoke-WebRequest "$base/motion-transfer-input-3.png" -OutFile 'input/pika_official_environment_ref.png'
   Get-FileHash 'input/pika_official_motion_ref.mp4','input/pika_official_character_ref.png','input/pika_official_environment_ref.png' -Algorithm SHA256
   ```

4. For the optional independent-audio case only, derive a WAV from the input video's audio, for example with FFmpeg installed and on PATH:

   ```powershell
   ffmpeg -i input/pika_official_motion_ref.mp4 -vn -c:a pcm_s16le input/pika_official_motion_audio_standalone.wav
   ```

   This WAV is a **local derivative**, not an official standalone audio sample. Do not use it to claim independently sourced audio reference validation.
5. Start the separate server as described in the installation guide. The graphs are ComfyUI **API-format** graphs, not UI drag-and-drop workflows: [768p](<../../workflows/ref2va-official-768p-10s-api.json>), [480p](<../../workflows/ref2va-official-480p-10s-api.json>), and [480p independent audio](<../../workflows/ref2va-standalone-audio-480p-10s-api.json>). From the ComfyUI root, submit one by wrapping it in the `/prompt` request envelope (choose the corresponding filename):

   ```powershell
   $graph = Get-Content -Raw 'custom_nodes/Pika-H3-Flex-4090/workflows/ref2va-official-768p-10s-api.json' | ConvertFrom-Json
   $body = @{ prompt = $graph } | ConvertTo-Json -Depth 100 -Compress
   Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8189/prompt' -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
   ```

   Inspect the returned prompt ID and server history/logs for completion, errors and cache events; output is written to ComfyUI `output/`. Alternatively, from the ComfyUI root run `python custom_nodes/Pika-H3-Flex-4090/tools/benchmark_comfy.py custom_nodes/Pika-H3-Flex-4090/workflows/ref2va-official-768p-10s-api.json --timeout 2400`. The local-only tool records node time, cache hits and sampled whole-card memory. Run each measurement in a fresh process. Never publish private logs, credentials, machine paths or generated output without appropriate rights.

The graphs preserve `BasicGuider`, `lcm`, `SamplerCustomAdvanced` and the official six-evaluation sigmas `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`. They use `PikaH3LowPeakLoRA` (`low_peak`, strength 1.0) and `PikaH3Optimize` (FFN chunk 16384, `native`) rather than claiming identical computation to the stock template. The sample covers two images and one video with one audio route, not all possible reference combinations or maximum durations.
