# SPDX-License-Identifier: GPL-3.0-only
"""Run each explicit graph in a fresh dedicated ComfyUI process, then stop it.

No dependencies/models are installed and no source is patched. stdout and raw
histories may contain local paths: keep output-dir private/ignored.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


def stop_owned(server):
    if server.poll() is not None:
        return
    if os.name == "nt":
        result = subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        if result.returncode and server.poll() is None:
            raise RuntimeError("Could not stop the owned ComfyUI process tree")
    else:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
    server.wait(timeout=30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--comfyui", type=Path, required=True)
    p.add_argument("--models-config", type=Path, required=True)
    p.add_argument("--graphs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--port", type=int, default=8189)
    p.add_argument("--reserve-vram", type=float, default=3.5)
    p.add_argument("--streams", type=int, choices=(1, 2), default=2)
    p.add_argument("--timeout", type=int, default=650)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.trust_env = False
    url = f"http://127.0.0.1:{a.port}"
    python, comfy, config = a.python.resolve(), a.comfyui.resolve(), a.models_config.resolve()
    root = Path(__file__).resolve().parents[1]
    outcomes = []
    for graph in a.graphs:
        try:
            session.get(url + "/system_stats", timeout=2, allow_redirects=False)
        except requests.ConnectionError:
            pass
        else:
            raise RuntimeError("Port already in use; refusing to stop someone else's server")
        target = a.output_dir / graph.stem
        target.mkdir()
        args = [str(python), "-u", "main.py", "--listen", "127.0.0.1", "--port", str(a.port),
                "--disable-auto-launch", "--offline", "--use-ck-attention", "--reserve-vram", str(a.reserve_vram),
                "--async-offload", str(a.streams), "--extra-model-paths-config", str(config)]
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
        with (target / "server.log").open("w", encoding="utf8") as log:
            server = subprocess.Popen(args, cwd=comfy, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=os.name != "nt")
            started = time.monotonic()
            try:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("ComfyUI failed; inspect private server.log")
                    try:
                        ready = session.get(url + "/system_stats", timeout=2, allow_redirects=False).status_code == 200
                    except requests.ConnectionError:
                        ready = False
                    if ready:
                        break
                    if time.monotonic() - started > 120:
                        raise TimeoutError("ComfyUI startup exceeded 120 seconds")
                    time.sleep(.5)
                startup = time.monotonic() - started
                with (target / "client.log").open("w", encoding="utf8") as client_log:
                    run = subprocess.run([str(python), "-B", str(root / "tools/benchmark_comfy.py"), str(graph.resolve()),
                                          "--server", url, "--timeout", str(a.timeout), "--output-dir", str(target.resolve())],
                                         cwd=root, env=env, stdout=client_log, stderr=subprocess.STDOUT,
                                         timeout=a.timeout + 90)
                outcomes.append({"graph": graph.name, "exit_code": run.returncode,
                                 "startup_seconds": startup, "fresh_process": True})
                print(outcomes[-1], flush=True)
            finally:
                stop_owned(server)
        # Wait only for the owned port to close; never kill an unrelated process.
        time.sleep(1)
    (a.output_dir / "batch-summary.json").write_text(json.dumps(outcomes, indent=2), encoding="utf8")
    raise SystemExit(0 if all(x["exit_code"] == 0 for x in outcomes) else 1)


if __name__ == "__main__":
    main()
