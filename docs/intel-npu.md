# Intel NPU lane (Windows, OpenVINO)

colibrì includes an **experimental, opt-in** lane that runs the routed experts of
Inkling-2 prompt chunks on an Intel NPU ("Intel AI Boost", Core Ultra) through
OpenVINO. It is off by default, never selected automatically, and computes in
**fp16**, so output may differ slightly from the CPU path.

```powershell
coli run   --npu --model <inkling-2 gguf model> "your prompt"
coli serve --npu --npu-stats --model <inkling-2 gguf model>
```

Without `--npu` nothing changes: no helper is loaded, no device is opened, and
the engine produces exactly the results it produces today.

## What runs on the NPU, and what does not

| work | where |
|---|---|
| routed experts of a **prompt chunk** (≥ `COLI_NPU_MIN_S` rows, default 16), experts with ≥ `COLI_NPU_MIN_ROWS` rows (default 8) | **NPU** |
| routed experts of a prompt chunk with fewer rows | CPU, all of an expert's rows in one batched call |
| **decode** (one token a step) | CPU (or Vulkan / CUDA if enabled), unchanged |
| attention, router, dense and shared matrices, KV cache | the normal path, unchanged |
| expert storage: NVMe streaming, RAM expert cache, pins, prefetch | colibrì's existing mechanisms, unchanged; the lane reads experts from the same slot cache |

Models: **Inkling-2 / Inkling-2-Mini from a GGUF** (`coli import`), the format the
lane was written and measured for. Any other model or format gets a one-line
`[NPU] requested, but ...` message and the normal path.

Why only prompt chunks: decode evaluates 8 experts per layer-pass, one row each.
An NPU submission costs about 1 ms plus the weight transfer, which makes decode
slower than colibrì's CPU kernels (measured, see
[`c/tools/npu_poc`](../c/tools/npu_poc/README.md)). In a prompt chunk the lane
groups the chunk's (token, expert) pairs **by expert**, so each expert becomes
one matrix-matrix product over all its rows, which is what the NPU is good at.

## How it works

```
moe_ex (inkling.c), chunk of S >= COLI_NPU_MIN_S rows
  ├─ group the chunk's pairs by expert
  ├─ for each group of experts that fits the slot cache:
  │    acquire + fill slots (RAM cache hit, or NVMe read), as moe() does
  │    for each expert:
  │      rows >= MIN_ROWS ─► decode GGUF blocks (IQ2/IQ3/IQ4_NL...) to f16
  │                          straight into a free NPU request, submit (async)
  │      rows <  MIN_ROWS ─► CPU block kernels, all rows at once
  │    wait for the experts in flight, scatter-add  y·weight  into the output
  └─ release slots
```

- **Host side** (`c/backend_npu.h`, `c/backend_npu.c`) is plain C, compiled in
  with `NPU=1`. It never includes or links OpenVINO: it loads
  `coli_npu.dll` from beside the executable at run time, checks its ABI version
  and that every entry point exists. An `NPU=1` build starts normally on a
  machine with no NPU, no OpenVINO and no helper.
- **Helper** (`c/backend_npu_helper.cpp`, MSVC, links OpenVINO) compiles **one
  graph per row bucket** (default 16, 64, 256 rows) for one expert:
  `x·W13ᵀ → split → softcap(g,4)·sigmoid(g)·softcap(u,25) → ·W2ᵀ`.
  The weights are **graph inputs** in f16, so every expert runs on the same
  compiled graph without recompiling. Each bucket keeps a few infer requests
  (`COLI_NPU_REQS`, default 3) so the NPU computes one expert while the CPU
  decodes the next one's weights.
- **No silent fallback to the CPU inside OpenVINO.** After compiling, the helper
  reads every graph's `EXECUTION_DEVICES` and refuses to run unless it is the
  requested device (`NPU`). The startup banner prints it:
  `[NPU] ... Graphs execute on: NPU.`
- **Compile once.** Graphs are cached in `%LOCALAPPDATA%\colibri\npu-cache`
  (`COLI_NPU_CACHE`); later starts load them from there.
- **Failure handling.** If a request fails, its rows are recomputed on the CPU,
  the lane turns itself off and says so on stderr. Nothing fails silently.
- **Memory.** The lane's host buffers are capped by `COLI_NPU_MEM_MB`
  (default 512 MiB; the defaults use about 160 MiB). The expert cache size is
  colibrì's usual `--cap` / `--ram`.

## Precision

The NPU computes in fp16 with f16 weights decoded from the GGUF's blocks.
`COLI_NPU_CHECK=N` recomputes the first N NPU experts on the CPU from the same
weights and prints the difference. Measured on Inkling-2-Mini abliterated
Mix-12G, Core Ultra 5 325:

| compared with | relative L2 of the expert output |
|---|---|
| the engine's exact float path (`GQ_EXACT=1`) | 8.3e-4 – 1.1e-3 |
| the engine's default CPU path (int8-quantized activations) | 1.0e-2 – 1.2e-2 |

The NPU is closer to exact arithmetic than the default CPU kernels, but it is
not bit-identical to them, so generated text can diverge after some tokens.

## Hardware

| hardware | status |
|---|---|
| Intel Core Ultra 5 325, NPU `DEVICE_ARCHITECTURE` 5010, driver 32.0.100.5540, Windows 11, OpenVINO 2026.3.1 | **tested** (all numbers in this document) |
| other Core Ultra NPUs (Meteor Lake, Lunar Lake, Arrow Lake, Panther Lake) | not tested; expected to work where OpenVINO's NPU plugin supports the device |
| Linux | the host side builds (it uses `dlopen`), the helper and its build are Windows-only and untested on Linux |
| AMD Ryzen AI NPU | not this lane; see [xdna.md](xdna.md) |

## Install and build (Windows)

You need, besides colibrì's usual Windows build ([windows.md](windows.md)):

1. **The Intel NPU driver** from Intel or your laptop maker (Device Manager shows
   *Intel(R) AI Boost* under *Neural processors*). No driver or BIOS change is
   needed beyond a normal installed driver.
2. **An OpenVINO runtime with C++ headers.** The simplest free source is the pip
   wheel (`py -3 -m pip install openvino`): it ships `include\`, `libs\openvino.lib`
   and the DLLs under `<python>\Lib\site-packages\openvino`. The OpenVINO archive
   from Intel (`runtime\include`, `runtime\lib\intel64\Release`) works too.
3. **MSVC** (Visual Studio Build Tools 2022, *Desktop development with C++*) for
   the helper: OpenVINO's C++ API is MSVC-ABI.

Build the engine with the lane (MSYS2 UCRT64, as for any Windows build):

```sh
make -C c inkling NPU=1            # add VK=1 for Vulkan as usual
```

Build the helper (a *Developer* or plain prompt; the script finds MSVC via vswhere):

```powershell
$env:OPENVINO_DIR = "C:\path\to\python\Lib\site-packages\openvino"
c\tools\build_npu_helper.cmd          # writes c\coli_npu.dll
```

At run time `openvino.dll` and its plugins must be found: put the helper's
directory and OpenVINO's `libs` on `PATH`, copy them beside the executable, or set

```powershell
$env:COLI_NPU_OPENVINO = "C:\path\to\python\Lib\site-packages\openvino\libs"
```

Check that it works:

```powershell
coli run --npu --npu-stats --model <inkling-2 gguf model> "Explain NVMe in one paragraph."
```

The engine prints the banner with `Graphs execute on: NPU` and the device memory
the lane allocated (`[NPU] device memory: 156 MiB allocated of 8192 MiB` here). At the
end it prints a line such as `[NPU] run: ... NPU x experts / y rows ...`, and a second
line with the expert cache's hits and misses and the GiB of expert weights copied from
the GGUF file. Those bytes come from the OS file cache or from the disk. With
`--cap 32` on the 16 GB machine above, one chat turn (prompt and a cancelled reply)
printed 77.5% hits and 114.77 GiB copied. That is why loading experts, not the NPU, sets
the prompt time there. If anything is missing it
prints `[NPU] requested, but <reason>: continuing on the normal path`.

## Options

| flag / variable | default | meaning |
|---|---|---|
| `--npu` / `COLI_NPU=1` | off | turn the lane on |
| `--npu-stats` / `COLI_NPU_STATS=1` | off | per-turn and per-run counts and times, expert cache hits/misses, GiB of expert weights copied |
| `COLI_NPU_DEVICE` | `NPU` | OpenVINO device to compile for; only `NPU` has been tested through the engine |
| `COLI_NPU_BUCKETS` | `16,64,256` | compiled row buckets |
| `COLI_NPU_REQS` | `3` | requests in flight per bucket |
| `COLI_NPU_MEM_MB` | `512` | cap on the lane's host buffers |
| `COLI_NPU_MIN_ROWS` | `8` | fewer rows per expert run on the CPU |
| `COLI_NPU_MIN_S` | `16` | smallest chunk the lane takes |
| `COLI_NPU_CACHE` | `%LOCALAPPDATA%\colibri\npu-cache` | compiled-graph cache (`0` disables) |
| `COLI_NPU_OPENVINO` | unset | directory with `openvino.dll` |
| `COLI_NPU_CHECK` | `0` | compare the first N NPU experts with the CPU |
| `COLI_NPU_VERBOSE` | `0` | helper diagnostics |

All of them are also in [ENVIRONMENT.md](ENVIRONMENT.md#inkling-engine-inkling).

## Benchmarks

See [Measured results](#measured-results) below; the script that produced them
is `bench/run_npu_benchmarks.ps1` (same model, same ~755-token prompt,
temperature 0, 100 generated tokens, the same `--cap`, a warm-up request, then
measured runs):

```powershell
powershell -File bench\run_npu_benchmarks.ps1 -Model <dir.colibri> -Cap 32 -Runs 2 `
    -OpenVino C:\path\to\openvino\libs
```

## Known limitations

- **Where the time goes.** The NPU waits about 0.1 s per 755-token prompt in
  every measured run: it is never the bottleneck. Decoding the experts' GGUF
  blocks to f16 on the host now takes about 3.5 s of the prompt's 23.6 s of MoE
  (0.6–1 ms an expert, `backend_npu_rows.h`); most of the rest is reading the
  experts into the RAM cache (`--cap 32` on a machine with about 5 GB free).
  More RAM for the expert cache, not a faster NPU, is the next lever here.
- **Short prompts gain less.** Below `COLI_NPU_MIN_S` rows (default 16) a chunk
  stays on the CPU, and with few rows per expert most experts stay under
  `COLI_NPU_MIN_ROWS`.
- **No decode speed-up**, by design (see above).
- **Padding.** Rows are padded to the next bucket (16/64/256): about 0.76 padding
  rows per real row on the measured prompt. It costs NPU time, which is not the
  bottleneck, and no host time.
- **Only the listed GGUF types are fast.** IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
  IQ3_S, IQ4_NL, IQ4_XS and Q4_0 use the vector decode; other types (K-quants,
  Q8_0, ...) work through the scalar decode (for the IQ types it replaced, about
  18× slower; other types not measured).
- **Inkling-2 GGUF only.** Other families and formats are not wired.
- **fp16**, not bit-identical to the CPU path.
- **Untested** on other Intel NPU generations and on Linux.
- The Vulkan expert tier and the NPU lane are independent: with both enabled,
  prompt chunks go to the NPU lane, decode to the Vulkan tier.

## Measured results

All numbers below are measured, none estimated. Machine: HP OmniBook X Flip 14,
Core Ultra 5 325, 16 GB (about 5 GB free), NVMe SSD, Windows 11, OpenVINO
2026.3.1. Model: Inkling-2-Mini abliterated Mix-12G GGUF. Every configuration:
the same 755-token prompt, temperature 0, 100 generated tokens, `--cap 32`, one
warm-up request, then measured runs. Median (min–max); peak RAM is the engine's
working set. Produced by `bench/run_npu_benchmarks.ps1`, summarized by
`bench/summarize_npu_benchmarks.py`; raw values and the lane's own counters in
[experiments/inkling2-npu-lane-2026-11-25-raw.txt](experiments/inkling2-npu-lane-2026-11-25-raw.txt),
the manifest (`COLI_NPU` 0 → 1) in
[experiments/inkling2-npu-lane-2026-11-25.json](experiments/inkling2-npu-lane-2026-11-25.json).

**Current lane** (commit 0d8128c4, vector f16 decode), 3 runs each:

| configuration | TTFT (s) | prompt tok/s | decode tok/s | peak RAM (GB) |
|---|---|---|---|---|
| CPU | 187.1 (172.0–209.7) | 4.04 (3.60–4.39) | 1.86 (1.84–2.45) | 4.27 |
| Vulkan (iGPU expert tier) | 55.8 (44.5–56.4) | 13.54 (13.39–16.96) | 1.88 (1.78–2.01) | 5.29 |
| **CPU + NPU** (`--npu`) | **33.6 (33.5–40.2)** | **22.50 (18.78–22.51)** | 2.44 (2.42–2.82) | 4.56 |
| CPU + NPU, `COLI_NPU_MIN_ROWS=2` | 34.0 (33.9–34.7) | 22.24 (21.76–22.25) | 2.78 (2.71–2.79) | 4.56 |
| Vulkan + NPU | 34.1 (33.8–34.7) | 22.17 (21.73–22.34) | 2.81 (2.67–2.82) | 4.86 |

Prompt processing with the NPU: **5.6× the CPU** and **1.7× the Vulkan iGPU tier**
on this machine. Decode is unchanged by design (it never uses the NPU); its spread
follows the machine's state. CPU + iGPU + NPU (Vulkan + NPU) measures the same as
CPU + NPU: prompt chunks go to the NPU lane and the iGPU tier, which streams
experts through 13 MiB on this machine, does not speed up decode here.

Per prompt (the lane's counters, CPU + NPU): 3713 experts / 284 490 rows on the
NPU (+216 150 padding rows), 1499 experts / 5046 rows on the CPU; host f16
decode 3.56 s, NPU wait 0.09 s, CPU experts 0.92 s, MoE total 23.65 s. The
graphs compiled once (cold) and loaded from the cache in about 50 ms afterwards.

**First version** (scalar f16 decode, commit 1cad8d99), 2 runs each, an earlier
session of the same machine (its CPU baseline ran faster, so compare within the
table only):

| configuration | TTFT (s) | prompt tok/s | decode tok/s |
|---|---|---|---|
| CPU | 150.8 (147.7–154.0) | 5.01 (4.90–5.11) | 2.41 (2.33–2.49) |
| Vulkan | 45.1 (41.4–48.7) | 16.87 (15.50–18.24) | 2.44 (2.33–2.55) |
| CPU + NPU | 70.2 (70.1–70.3) | 10.75 (10.74–10.77) | 2.67 (2.62–2.71) |
| CPU + NPU, `COLI_NPU_MIN_ROWS=32` | 57.5 (57.3–57.7) | 13.12 (13.08–13.17) | 2.73 (2.69–2.78) |

There the host's f16 decode took 39.5 s of 58.4 s of MoE per prompt (10.6 ms an
expert) while the NPU waited 0.11 s, which is what the vector decode fixed. The
generated text per prompt is identical before and after it.

Not measured: energy (no reliable per-device power counter was available), NPU
device memory through the engine (the banner prints it from this version on),
other prompt lengths and longer generations.
