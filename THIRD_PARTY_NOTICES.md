# Third-party code notices

The preserved upstream standalone inference and loader code (`src/`) is distributed under Apache-2.0 (LICENSE-CODE). The root LICENSE and NOTICE describe the MiniMax-H3-derived model adapters, not a replacement for the code licenses.

## 4090 ComfyUI extension

The new root `__init__.py`, `comfy_extension/`, new optimization tests, and tools marked `SPDX-License-Identifier: GPL-3.0-only` are GNU GPLv3-only. The complete license is in `comfy_extension/LICENSE`. This component uses ComfyUI native loaders, adapter interfaces and ModelPatcher lifecycle, and follows its GPLv3 integration patterns:

- https://github.com/Comfy-Org/ComfyUI/tree/0dbfd493b4cc604db743c699f0f7ce01348e54a4
- https://github.com/Comfy-Org/ComfyUI/blob/0dbfd493b4cc604db743c699f0f7ce01348e54a4/LICENSE

No ComfyUI or LightX2V source tree, compiled kernel wheel, or model weight is bundled. LightX2V provided design references (projection/activation fusion and offload scheduling), not a claim that its incompatible quantization/LoRA flags are enabled:
https://github.com/ModelTC/LightX2V

The UI workflow is modified from the Comfy-Org Pika T2V workflow template (MIT, Copyright (c) 2023-present Comfy Org); its license is retained in `licenses/Comfy-Workflow-Templates-MIT.txt`. Changes replace LoRA/FFN nodes and set the measured prompt, dimensions, duration and seed. The API workflow follows the same native pipeline and published sampling schedule. Source: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_pika_h3_flex_t2v.json . This is an independent, unofficial extension, not an endorsement by Pika, MiniMax, Comfy-Org, Qwen or LightX2V.

GPU dependencies are installed separately from their publishers. Their code licenses do not change the terms applicable to model weights. The MiniMax H3 model license includes territorial and use restrictions; this repository does not waive them or describe the weights as unrestricted OSI open source.

## Diffusers

Pinned source: https://github.com/huggingface/diffusers/tree/3a2f35d4efa4c059c8bfb3bc0d6c906264895c81

The transformer and scheduler patches retain their original copyright and Apache-2.0 headers and identify Mellis modifications. Portions of geometry.py, packing.py and vae.py are adapted from the MiniMax H3 modular pipeline: Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved. See licenses/Apache-2.0.txt.

## Model dependencies

Base model and Qwen3-VL encoder weights are downloaded separately at runtime; they are not bundled in this code repository or its Docker image. The downloader preserves available upstream LICENSE/NOTICE files. If encoder files are redistributed separately, retain their Apache-2.0 license and any applicable upstream NOTICE.
