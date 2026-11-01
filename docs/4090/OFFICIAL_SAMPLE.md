# Official showcase: prompt-matched visual comparison

This case prepares a **qualitative comparison**, not an exact reproduction claim or a new performance benchmark. It uses Pika-Labs' public **balcony-orbit / “A journey between balconies”** showcase, the only entry in the checked-in [showcase manifest](<../../assets/showcases/manifest.json>) with `inputs=[]`. No reference image, video conditioning or model download is required to prepare this case.

The official manifest explicitly says showcase seeds, exact rendering checkpoints and service preprocessing are unspecified. Its download revision identifies the published reference asset, **not the checkpoint used to render it**. Our local seed 42 is a comparison choice, not the undisclosed official seed.

## Prepared inputs

- [Unmodified public prompt](<../../examples/official_sample/prompt.txt>): exact UTF-8 prompt string from the manifest, with no added trailing newline or rewriting.
- [ComfyUI API graph](<../../examples/official_sample/api-balcony.json>): ready for a separately authorized local generation; an API graph, not a drag-and-drop UI workflow.
- [Provenance and decoded media metadata](<../../examples/official_sample/provenance.json>): source URL/revision, hashes, frame-grid reasoning, local settings and unknown official settings.

The manifest names a separate prompt asset that is absent from this checkout. The embedded prompt string is the source; it was not reconstructed from the video or abbreviated.

## Reference acquisition and actual media properties

The [official reference video](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/balcony-orbit.mp4) was fetched successfully from the fixed public revision `0ddf55ad7067ca1e76812253bd7b4600df87f759`; no `main` fallback was needed. Its SHA-256 matched the manifest:

```text
fcc6901cd5b29d7cf21cd0980d9520ee19bdca8c4941fa01b7340dc70c19ac00
```

CPU-only PyAV 19.0.2 decoding verified all frames and the audio stream:

| Property | Actual reference |
| --- | --- |
| File size | 18,189,846 bytes |
| Video | H.264, 1344 × 768, 328 decoded frames, 24 fps |
| Video duration | 328 / 24 = 13.6666667 seconds |
| First / last video timestamp | 0.0 / 13.625 seconds |
| Container duration | 13.675 seconds |
| Audio | AAC, 32 kHz, stereo, 437,600 decoded samples per channel |
| Audio stream duration | 13.675 seconds |

The manifest's 13.675-second duration is the container/audio duration, not a reason to request 329 frames or round the video to 15 seconds. The API graph uses **328 = 17 × 19 + 5**, a legal model frame-grid length, at 24 fps. Minor video/audio end-time differences are recorded rather than silently padded or trimmed to manufacture equality.

The downloaded MP4 remains only in the ignored local cache `benchmarks/local/official/balcony-orbit.mp4`. It is not part of the committed comparison case. Readers can fetch the public URL themselves, subject to upstream terms, and verify the SHA-256. Do not persist Hugging Face's expiring signed CDN redirect URLs or publish private proxy/credential configuration.

## Local candidate contract

The case derives from the repository's [768p API graph](<../../workflows/api-t2v-768p-15s.json>), changing the prompt, actual frame count and output prefix while preserving:

- the static pruned INT8 FL2VA model, Qwen NVFP4 encoder, INT8 video VAE, FP32 audio VAE and matching public Comfy Org Pika LoRA;
- `PikaH3LowPeakLoRA`: strength 1.0, mode `low_peak`;
- `PikaH3Optimize`: chunk size `16384`, activation `native`, `fused_modulation=false`;
- `BasicGuider`, `lcm`, six model evaluations and sigmas `1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0`;
- seed 42, 1344 × 768, full audio/video decoding and MP4 saving.

See the [installation guide](<INSTALL.md>) for the separate ComfyUI runtime. The graph has no input-media connections: the conditioning node's `ImageToVideo` class name does not mean an image was supplied here. Do not feed the reference MP4 into the graph; that would change this text-only comparison into another task.

The API graph belongs inside the `prompt` member of a normal ComfyUI `/prompt` request envelope; do not POST the bare graph as though it were an envelope. Generation is performed separately by the operator, not by opening the prompt or provenance files. The intended output prefix is `pika4090_official_balcony_low_peak_native_seed42`.

**The local candidate was generated and decoded** using the [digest-identified tested inference source](../BENCHMARKS.md#final-source-qualification), in 327.994 seconds with no node-result cache hits. The same prompt, seed and quantized artifacts with stock bypass completed in 340.461 seconds; all 328 decoded RGB frames and decoded PCM matched the low-peak version exactly on this local comparison. These single timings are not a controlled performance claim. See the [qualification receipt](../../benchmarks/results/qualified-4090.json).

[Watch the official reference](https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/0ddf55ad7067ca1e76812253bd7b4600df87f759/assets/showcases/balcony-orbit.mp4). The locally generated result remains local and is not redistributed in this repository; review the model license before generating, using or publicly displaying outputs.

A six-time-point visual inspection found that both show a white tower with three balconies, coordinated stretches and changing camera framing. Framing, motion timing and people differ; still frames cannot establish motion quality. We did not listen to or subjectively rate either soundtrack. The official seed and generating checkpoint remain unknown, so this is not an exact reproduction claim.

## How to compare without inventing a ground truth

Watch reference and candidate at native speed, same display size, with sound. Compare complete motion as well as selected opening/middle/end frames. Use matching time positions to discuss camera phases, not pixel-error scores against an unrelated seed.

| Checkpoint from the published prompt | What to inspect in both clips |
| --- | --- |
| Opening composition | Building on the right, open pale-blue sky on the left; three stacked balconies and distinct people/outfits |
| 0–3 s orbit | Continuous corner orbit and convincing balcony/building/sky parallax, not only a flat zoom |
| 3–5 s push-in | Middle balcony and rust-red tracksuit subject; railing foreground and coherent perspective |
| 5–7 s boom upward | Traversal past the slab underside to the lilac-clothed top-balcony subject |
| 7–10 s descent | Controlled move past the middle balcony to the older bottom-balcony subject |
| 10–13 s return | Pullback/reverse orbit toward the opening wide composition, without an unrequested cut |
| Human motion | Calm continuous stretches, believable balance, no freezes, limb duplication or abrupt jerks |
| Temporal stability | Balcony geometry, railings, person identity, clothing and palette remain coherent through occlusion and camera motion |
| Audio | Wind/city/birds/fabric ambience; listen for unwanted music, speech, voices or other prompt violations |
| Technical integrity | Full frame count/duration, no missing tail frames, expected audio format and no decode errors |

These are **review questions, not assertions that the official or local clip satisfies every prompt instruction**. The preparation task decoded the media but did not conduct a subjective image-quality or audio audition. Report observed failures, not only attractive stills. Non-silent audio alone cannot establish synchronization, naturalness or compliance with “no music/no voices.”

## Interpretation and exclusions

- Official seed, exact checkpoint/adapter revisions, service prompt processing, sampling/runtime configuration and any editing/postprocessing are unknown.
- Identical text and matching geometry do not imply identical conditioning, latents, pixels, sound or creative outcome.
- Do not use the official showcase as bitwise/pixel/audio truth for testing near-equivalent code changes. Those need same-input, same-model local reference-versus-candidate runs.
- A visually different result alone does not isolate an optimization bug: quantization, checkpoint identity, unknown seed and upstream service behavior are confounders.
- This 328-frame case is **separate from the 362-frame / approximately 15-second performance workload**. Do not add its runtime to that table or claim a speedup from the shorter clip.
- No reference video, model weights or personal configuration is committed. Existing upstream/model license obligations and [license boundaries](<OPTIMIZATIONS.md#license-boundaries>) still apply; the comparison does not grant additional media or trademark rights.
