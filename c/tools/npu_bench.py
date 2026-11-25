#!/usr/bin/env python3
"""Reproducible end-to-end benchmark for one colibri configuration (standard library only).

Starts `coli serve` with the given environment, sends the same fixed prompt (about 550
tokens) as a streamed chat completion at temperature 0, and records, per run:

  - time to first token (s) and prompt throughput (prompt tokens / time to first token);
  - decode throughput (generated tokens after the first / time from first to last token);
  - the engine process's peak working set (Windows) or max RSS (elsewhere);
  - the generated text's SHA-256, so two configurations can be compared for equality.

Every number printed is measured; nothing is estimated. Output: one JSON line per run on
stdout and, with --out, a JSON report.

usage:
  python tools/npu_bench.py --model DIR --label cpu --runs 2 [--env COLI_VULKAN=0 ...]
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
COLI = os.path.join(os.path.dirname(HERE), "coli")

# A fixed English prompt: a technical passage the model must summarise. Deterministic and
# long enough (~550 tokens with the chat template) to measure prompt processing.
PASSAGE = (
    "A mixture-of-experts language model replaces the single feed-forward block of each "
    "transformer layer with many smaller feed-forward networks, the experts, and a router "
    "that sends every token to a few of them. The parameter count grows with the number of "
    "experts, but the compute per token grows only with the number of experts each token "
    "visits. Inference engines exploit this in two ways. On machines with plenty of fast "
    "memory they keep every expert resident and batch the tokens that visit the same expert "
    "together, so that each expert's weights are read once per batch. On machines with little "
    "memory they keep only the dense part of the model resident and stream the experts from "
    "storage when the router asks for them, through a cache that learns which experts the "
    "workload uses most. The second approach turns a memory problem into a bandwidth problem: "
    "the speed of decoding is set by how many expert bytes must be read for each token and by "
    "how fast the slowest tier that serves them can deliver them. Prompt processing behaves "
    "differently, because a long prompt visits almost every expert of every layer, and the "
    "cost of reading an expert is shared by all the prompt tokens routed to it. A good engine "
    "therefore groups the prompt's tokens by expert before computing, uses matrix-matrix "
    "kernels instead of matrix-vector ones, and overlaps the reading of the next experts with "
    "the computation of the current ones. Accelerators such as integrated GPUs and neural "
    "processing units add compute without adding memory bandwidth, since they share the "
    "system memory with the processor; whether they help depends on whether the workload is "
    "limited by arithmetic or by the bytes that must be moved, and on the fixed cost of every "
    "submission to the device. "
)
PROMPT = ("Read the following passage twice, carefully.\n\n" + PASSAGE + "\n\n" + PASSAGE +
          "\n\nIn exactly three sentences, explain when an accelerator that shares system memory "
          "helps a mixture-of-experts engine and when it does not.")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def peak_mem_gb(image="inkling.exe"):
    """Peak working set of the newest process named `image` (Windows), else None."""
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True).stdout.strip().splitlines()
        pids = [int(l.split('","')[1]) for l in out if l.startswith('"')]
        if not pids:
            return None

        class PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        best = 0.0
        for pid in pids:
            h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
            if not h:
                continue
            c = PMC(); c.cb = ctypes.sizeof(PMC)
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                best = max(best, c.PeakWorkingSetSize / 2**30)
            ctypes.windll.kernel32.CloseHandle(h)
        return round(best, 3)
    except Exception:
        return None


def wait_ready(base, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with status {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(1)
    raise RuntimeError("server not ready")


def model_id(base):
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        return json.loads(r.read())["data"][0]["id"]


def one_request(base, max_tokens, mid, tag):
    # The tag makes every request's first tokens differ, so no run can reuse a previous run's
    # KV state (the prefix store is also turned off in main): each run reads its whole prompt.
    body = json.dumps({"model": mid, "stream": True, "temperature": 0, "max_tokens": max_tokens,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": f"Request {tag}. {PROMPT}"}]}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; chunks = 0; text = []; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    if t_first is None:
                        t_first = now
                    t_last = now; chunks += 1; text.append(piece)
    ptok = (usage or {}).get("prompt_tokens"); ctok = (usage or {}).get("completion_tokens") or chunks
    ttft = (t_first - t0) if t_first else None
    dec = (ctok - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first and ctok > 1 else None
    txt = "".join(text)
    return {"ttft_s": round(ttft, 3) if ttft else None,
            "prompt_tokens": ptok, "prompt_tok_s": round(ptok / ttft, 2) if ptok and ttft else None,
            "completion_tokens": ctok, "decode_tok_s": round(dec, 2) if dec else None,
            "text_sha256": hashlib.sha256(txt.encode()).hexdigest()[:16], "text_head": txt[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the engine")
    ap.add_argument("--serve-arg", action="append", default=[], help="extra argument for coli serve")
    ap.add_argument("--out")
    a = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("COLI_PREFIX_STORE", "0")      # no prompt reuse across runs
    for kv in a.env:
        k, v = kv.split("=", 1); env[k] = v
    port = free_port(); base = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, COLI, "serve", "--model", a.model, "--port", str(port)] + a.serve_arg
    log = open(os.path.join(HERE, f"npu_bench_{a.label}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    report = {"label": a.label, "env": a.env, "serve_args": a.serve_arg, "runs": []}
    try:
        report["load_s"] = round(wait_ready(base, proc), 1)
        mid = model_id(base)
        one_request(base, 8, mid, "W")            # warm-up: page cache, expert cache, compiled graphs
        for i in range(a.runs):
            r = one_request(base, a.max_tokens, mid, str(i + 1))
            r["run"] = i + 1; r["peak_ws_gb"] = peak_mem_gb()
            report["runs"].append(r)
            print(json.dumps({"label": a.label, **r}), flush=True)
    finally:
        if platform.system() == "Windows":         # the gateway spawns the engine: end the whole tree
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
