# SPDX-License-Identifier: GPL-3.0-only
"""ComfyUI custom-node entry point; the upstream standalone package is separate."""

async def comfy_entrypoint():
    from .comfy_extension.nodes import Pika4090Extension
    return Pika4090Extension()
