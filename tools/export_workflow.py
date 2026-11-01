# SPDX-License-Identifier: GPL-3.0-only
"""Adapt the official MIT Pika UI template to the tested custom-node graph.

Download the official video_pika_h3_flex_t2v.json separately, retaining the MIT
notice. This tool performs no network requests or model downloads.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    root = Path(__file__).resolve().parents[1]
    api = json.loads((root / "workflows/api-t2v-768p-15s.json").read_text(encoding="utf8"))
    w = json.loads(a.template.read_text(encoding="utf8"))
    outer = next(n for n in w["nodes"] if n["id"] == 140)
    prompt = api["131"]["inputs"]["prompt"]
    outer["widgets_values"][0:5] = [prompt, 1344, 768, 15, 42]
    outer["widgets_values_named"].update(prompt=prompt, width=1344, height=768, value_1=15, noise_seed=42)
    for item in outer["inputs"]:
        if item["name"] in ("width", "height"):
            item["link"] = None
    w["nodes"] = [n for n in w["nodes"] if n["id"] != 115]
    w["links"] = [link for link in w["links"] if link[0] not in (246, 247)]
    save = next(n for n in w["nodes"] if n["id"] == 92)
    save["widgets_values"][0] = "video/Pika4090"
    save["widgets_values_named"]["filename_prefix"] = "video/Pika4090"
    note = next(n for n in w["nodes"] if n["id"] == 117)
    note["widgets_values"][0] = "# Unofficial RTX4090 integration\nOpen the subgraph to change optional optimizations. Keep the official Pika six-NFE schedule.\n\n" + note["widgets_values"][0]
    g = w["definitions"]["subgraphs"][0]
    lora = next(n for n in g["nodes"] if n["id"] == 134)
    lora["type"] = "PikaH3LowPeakLoRA"
    lora["properties"]["Node name for S&R"] = lora["type"]
    lora["properties"].pop("cnr_id", None)
    lora["properties"].pop("ver", None)
    for item in lora["inputs"]:
        if item["name"] == "strength_model":
            item["name"] = "strength"
            if "widget" in item:
                item["widget"]["name"] = "strength"
    lora["inputs"].append({"name": "mode", "type": "COMBO", "widget": {"name": "mode"}, "link": None})
    lora["widgets_values"] = ["pika_h3_flex_fl2va_v1_bf16.safetensors", 1.0, "low_peak"]
    lora["widgets_values_named"] = {"lora_name": lora["widgets_values"][0], "strength": 1., "mode": "low_peak"}
    lora["outputs"][0]["links"] = [263]
    for link in g["links"]:
        if link["id"] == 256:
            link["origin_id"] = 201
    chunk = {"id": 201, "type": "PikaH3Optimize", "pos": [-1230, 5480], "size": [340, 150], "flags": {}, "order": 16, "mode": 0,
             "inputs": [{"name": "model", "type": "MODEL", "link": 263},
                        *[{"name": name, "type": kind, "widget": {"name": name}, "link": None}
                          for name, kind in (("chunk_size", "COMBO"), ("activation", "COMBO"), ("fused_modulation", "BOOLEAN"))]],
             "outputs": [{"name": "MODEL", "type": "MODEL", "links": [256]}],
             "properties": {"Node name for S&R": "PikaH3Optimize"},
             "widgets_values": ["16384", "native", False],
             "widgets_values_named": {"chunk_size": "16384", "activation": "native", "fused_modulation": False}}
    g["nodes"].append(chunk)
    g["links"].append({"id": 263, "origin_id": 134, "origin_slot": 0, "target_id": 201, "target_slot": 0, "type": "MODEL"})
    g["state"].update(lastNodeId=201, lastLinkId=263)
    condition = next(n for n in g["nodes"] if n["id"] == 131)
    condition["widgets_values"] = [prompt, 1344, 768, 362]
    condition["widgets_values_named"].update(prompt=prompt, width=1344, height=768, length=362)
    noise = next(n for n in g["nodes"] if n["id"] == 129)
    noise["widgets_values"][0] = 42
    noise["widgets_values_named"]["noise_seed"] = 42
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(w, indent=2, ensure_ascii=False), encoding="utf8")
    print(a.output)


if __name__ == "__main__":
    main()
