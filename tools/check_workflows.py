# SPDX-License-Identifier: GPL-3.0-only
"""Static validation of public workflows without ComfyUI or model weights."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = [1., .972973, .923077, .8, .571429, .333333, 0.]


def check(path):
    text = path.read_text(encoding="utf8")
    if re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|/Users/|/mnt/', text):
        raise ValueError(f"Machine-specific path in {path.name}")
    graph = json.loads(text)
    if "nodes" in graph:
        ids = {n["id"] for n in graph["nodes"]}
        for link in graph["links"]:
            assert link[1] in ids and link[3] in ids, (path.name, link)
        return
    for key, node in graph.items():
        assert "class_type" in node and "inputs" in node, (path.name, key)
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph, (path.name, key, value)
    inputs = graph["131"]["inputs"]
    assert (inputs["width"], inputs["height"], inputs["length"]) == (1344, 768, 362)
    assert graph["130"]["inputs"]["fps"] == 24
    assert graph["123"]["inputs"]["sampler_name"] == "lcm"
    assert graph["126"]["class_type"] == "BasicGuider"
    sigmas = [float(x) for x in graph["143"]["inputs"]["sigmas"].split(",")]
    assert len(sigmas) == 7 and all(abs(x - y) < 1e-7 for x, y in zip(sigmas, SIGMAS))


if __name__ == "__main__":
    files = list((ROOT / "workflows").glob("*.json"))
    if not files:
        raise SystemExit("No public workflows found")
    for path in files:
        check(path)
    print(f"PASS {len(files)} workflows")
