# SPDX-License-Identifier: GPL-3.0-only
"""Benchmark one explicit Pika API graph against a dedicated loopback ComfyUI.

Raw histories stay in an ignored local directory. Only the small public summary
is suitable for review/publication; it contains no prompt, tensor repr or paths.
"""
import argparse
import hashlib
import ipaddress
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

SIGMAS = [1., .972973, .923077, .8, .571429, .333333, 0.]


def loopback_url(value):
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.path not in ("", "/"):
        raise ValueError("Use a plain loopback HTTP origin")
    if parsed.hostname != "localhost" and not ipaddress.ip_address(parsed.hostname).is_loopback:
        raise ValueError("Benchmarking is local-only")
    return value.rstrip("/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph", type=Path)
    p.add_argument("--server", type=loopback_url, default="http://127.0.0.1:8189")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--output-dir", type=Path, default=Path("benchmarks/local"))
    a = p.parse_args()
    graph = json.loads(a.graph.read_text(encoding="utf8"))
    condition = graph["131"]["inputs"]
    values = [float(x) for x in graph["143"]["inputs"]["sigmas"].split(",")]
    if len(values) != 7 or any(abs(x - y) > 1e-7 for x, y in zip(values, SIGMAS)):
        raise ValueError("Benchmark requires the official Pika six-NFE sigmas")
    if graph["123"]["inputs"]["sampler_name"] != "lcm":
        raise ValueError("Benchmark requires the Pika LCM sampler")
    session = requests.Session()
    session.trust_env = False
    def get(path):
        r = session.get(a.server + path, timeout=30, allow_redirects=False)
        r.raise_for_status()
        return r.json()
    queue = get("/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("Use a dedicated idle ComfyUI instance")
    output = a.output_dir / (a.graph.stem + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    r = session.post(a.server + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=30, allow_redirects=False)
    r.raise_for_status()
    if r.json().get("node_errors"):
        raise RuntimeError(str(r.json()["node_errors"]))
    pid = r.json()["prompt_id"]
    print("SUBMITTED", pid, "local output", output, flush=True)
    max_used = None
    samples = 0
    while time.monotonic() - start < a.timeout:
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=8)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            smi = None
        if smi and smi.returncode == 0:
            with (output / "gpu-samples.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps({"elapsed": time.monotonic() - start, "gpu": smi.stdout}) + "\n")
            first = smi.stdout.splitlines()[0].split(",")[0].strip()
            used = float(first)
            max_used = used if max_used is None else max(max_used, used)
            samples += 1
        history = get("/history/" + pid)
        if pid in history:
            raw = history[pid]
            (output / "history.json").write_text(json.dumps(raw, indent=2), encoding="utf8")
            events = dict(raw["status"]["messages"])
            success = raw["status"]["status_str"] == "success"
            cache = events.get("execution_cached", {}).get("nodes", [])
            seconds = ((events["execution_success"]["timestamp"] - events["execution_start"]["timestamp"]) / 1000) if success else None
            media_cache = any(node in cache for node in ("125", "121", "122"))
            summary = {"status": "success" if success else "error", "execution_seconds": seconds,
                       "observed_seconds": time.monotonic() - start, "cached_nodes": cache,
                       "valid_generation_timing": success and not media_cache,
                       "width": condition["width"], "height": condition["height"], "frames": condition["length"],
                       "fps": graph["130"]["inputs"]["fps"], "nfe": 6,
                       "graph_sha256": hashlib.sha256(a.graph.read_bytes()).hexdigest(),
                       "lora_mode": graph["134"]["inputs"].get("mode", "native_bypass"),
                       "ffn": graph["201"]["inputs"], "outputs": raw.get("outputs", {}),
                       "whole_card_sampled_max_MiB": max_used, "gpu_samples": samples,
                       "timing_scope": "execution_start to success, excluding server startup; file/kernel caches may be warm"}
            # The model input link is structural metadata, not a file or prompt.
            (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
            print(json.dumps(summary, indent=2), flush=True)
            if not success:
                print("Error details are in ignored history.json", flush=True)
            raise SystemExit(0 if success and not media_cache else 1)
        time.sleep(5)
    queue = get("/queue")
    if any(row[1] == pid for row in queue.get("queue_running", [])):
        session.post(a.server + "/interrupt", timeout=10, allow_redirects=False).raise_for_status()
    session.post(a.server + "/queue", json={"delete": [pid]}, timeout=10, allow_redirects=False).raise_for_status()
    print("TIMEOUT: interrupt requested for this prompt; CUDA may still be busy. Stop the dedicated server if necessary.", flush=True)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
