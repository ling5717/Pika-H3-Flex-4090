# SPDX-License-Identifier: GPL-3.0-only
"""Thin Comfy V3 nodes; optimization lifetime belongs to ModelPatcher injections."""
import folder_paths
import comfy.utils
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.minimax.model import MiniMaxH3Model

from .ffn import optimize_model
from .lora import load_low_peak_lora


class PikaH3LowPeakLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PikaH3LowPeakLoRA", display_name="Pika H3 Low-Peak LoRA",
            category="model/patch/pika_h3_4090", is_experimental=True,
            description="Plain linear LoRA only; do not stack bypass loaders. low_peak preserves native rounding. addmm is a research mode: small operator errors changed the measured fixed-seed video/audio.",
            inputs=[io.Model.Input("model"),
                    io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                    io.Float.Input("strength", default=1.0, min=-100.0, max=100.0, step=0.01),
                    io.Combo.Input("mode", options=["low_peak", "addmm"], default="low_peak")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, lora_name, strength=1.0, mode="low_peak"):
        if type(model.get_model_object("diffusion_model")) is not MiniMaxH3Model:
            raise TypeError("PikaH3LowPeakLoRA supports MiniMax-H3 only")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        state = comfy.utils.load_torch_file(path, safe_load=True)
        return io.NodeOutput(load_low_peak_lora(model, state, strength, mode=mode))


class PikaH3Optimize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PikaH3Optimize", display_name="Pika H3 FFN Optimization",
            category="model/patch/pika_h3_4090", is_experimental=True,
            description="Token-independent FFN chunks. Attention, weights and the six-step schedule are unchanged. Triton and compile are explicit candidates.",
            inputs=[io.Model.Input("model"),
                    io.Combo.Input("chunk_size", options=["8192", "16384", "32768"], default="16384",
                                   tooltip="16384 is the reference-qualified default. 32768 changed the fixed-seed output and was not faster in the measured case."),
                    io.Combo.Input("activation", options=["native", "triton", "compiled"], default="native",
                                   tooltip="Native and Triton preserved the measured reference. Compiled activation changed output and had startup overhead; research only."),
                    io.Boolean.Input("fused_modulation", default=False, optional=True,
                                     tooltip="Research only: changes rounding and the fixed-seed video/audio; no demonstrated end-to-end gain.")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, chunk_size="16384", activation="native", fused_modulation=False):
        result = optimize_model(model, int(chunk_size), activation)
        if fused_modulation:
            from .modulation import enable_fused_modulation
            result = enable_fused_modulation(result)
        return io.NodeOutput(result)


class Pika4090Extension(ComfyExtension):
    async def get_node_list(self):
        return [PikaH3LowPeakLoRA, PikaH3Optimize]
