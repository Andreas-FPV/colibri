# Intel NPU feasibility probe (OpenVINO)

`npu_poc.cpp` is a standalone program, not part of any engine. It measures, on the machine it
runs on, whether the operations of an Inkling-2-Mini routed expert can run on an Intel NPU
through OpenVINO, and at what cost. It was written before any engine code so that the backend
design rests on measurements instead of datasheet TOPS.

## Build and run

Requirements: MSVC (Build Tools 2022) and an OpenVINO runtime (the `openvino` pip wheel ships
the C++ headers and import libraries under `site-packages/openvino`).

```bat
set OPENVINO_DIR=<venv>\Lib\site-packages\openvino
c\tools\npu_poc\build_poc.cmd
set PATH=%OPENVINO_DIR%\libs;%PATH%
c\tools\npu_poc\npu_poc.exe            &rem every test, every device
c\tools\npu_poc\npu_poc.exe expert     &rem one test group: matmul | expert | layer
c\tools\npu_poc\npu_poc.exe layer --dev NPU
```

`gguf_types.py FILE.gguf` prints a GGUF's metadata and the ggml type of every tensor kind (it
reads only the header, so it works on a partial download).

## What it tests

| group | question |
|---|---|
| devices | does OpenVINO see the NPU; architecture, driver, memory, precisions |
| `matmul` | y = x W^T at decode and prompt shapes; f16, int8 per channel, int4 per 128 |
| `expert` | one Inkling-2-Mini expert (2304 -> 2 x 1152 -> 2304, softcap(g,4) sigmoid(g) softcap(u,25)) |
| `expert` (8 in one graph) | the 8 routed experts of one decode step as one submission |
| `layer` | a whole layer-pass of a prompt chunk: 128 experts with R rows each, weights swapped every call, 1-4 requests in flight |

Weights are either compiled into the graph (`W-const`) or are graph **inputs** (`W-in`), so a
compiled model can be fed any expert without recompiling. Every output is compared with a
float64 reference computed from the exact weights the device received (so the error shown is
the device arithmetic, not quantization).

## Results on the development machine

HP OmniBook X Flip 14, Intel Core Ultra 5 325 (8 cores), 16 GB, Intel Graphics iGPU,
Intel AI Boost NPU (`DEVICE_ARCHITECTURE` 5010, driver 32.0.100.5540, 8 GiB addressable,
`OPTIMIZATION_CAPABILITIES` FP16 INT8, `DEVICE_GOPS` f16 23 962 / i8 47 923), Windows 11,
OpenVINO 2026.3.1. Raw output: [results/](results/). All numbers are measured.

**One expert, M rows (ms per call, steady state).**

| | CPU | iGPU | NPU |
|---|---|---|---|
| M=1, f16, W-const | 1.34 | 3.50 | 1.23 |
| M=1, f16, W-in (+ copy of 16 MB) | 2.52 + 2.76 | 0.96 + 1.95 | 0.91 + 2.16 |
| M=1, i4/g128, W-in | 26.0 | 2.06 | 0.86 + 0.18 |
| M=128, f16, W-in | 13.6 | 1.12 | 0.94 |
| M=128, i8/ch, W-in | 19.4 | 2.26 | 4.76 |
| M=128, i4/g128, W-in | 46.5 | 2.76 | 1.08 |

Compiling one expert graph costs 0.3-3.3 s on the NPU. Error vs float64: relative L2 about
1.1e-3 on the NPU (fp16 arithmetic), 8e-4 on the iGPU, 1e-6 on the CPU (with W-in; with
W-const int8/int4 the CPU and iGPU plugins quantize the activations too, 1-2e-2).

**A layer-pass of a prompt chunk** (128 experts, f16 weights as inputs, replaced every call):

| rows per expert | best NPU | best iGPU | CPU (OpenVINO) |
|---|---|---|---|
| 8 | 358 ms | 191 ms | 1163 ms |
| 32 | 147 ms | 146 ms | 1468 ms |
| 64 | 149 ms | 145 ms | 946 ms |
| 128 | 142 ms | 147 ms | 1324 ms |

In every NPU and iGPU row above, the time is the **host filling the weight inputs**
(128 x 16 MB = 2 GB of f16 per layer-pass, single-threaded memcpy): the device work is fully
hidden behind it with 2-4 requests in flight.

## Conclusions that drive the backend design

1. The NPU works and is accurate to fp16: an Inkling expert (two matmuls, tanh, sigmoid,
   products) compiles and runs, relative error about 1e-3.
2. **Weights must be graph inputs.** Compiling per expert (0.3-3.3 s each, 3072 experts) is
   out of the question; one compiled graph per row bucket, fed any expert, works and needs no
   recompilation.
3. **Decode is not an NPU workload.** A decode step evaluates 8 experts in each of 48
   layer-passes; at 0.9-1 ms per NPU submission (plus the weight copy) that is about 0.4 s a
   token for the experts alone, slower than colibri's CPU path. Putting the 8 experts in one
   graph did not help (3-11 ms per submission).
4. **Prompt processing is.** Grouping a prompt chunk's rows by expert turns 128 GEMV into 128
   GEMM per layer-pass; the NPU computes one 128-row expert in about 1 ms, and its time is
   then set by the weight bytes the host provides.
5. **The bytes are the bottleneck**, so the weight format matters more than the arithmetic:
   f16 costs 16 MB per expert, int8 with a scale per 32 weights about 8.5 MB, and int4 is
   only an option for 4-bit linear sources. The int8-weight graph compiled by the NPU plugin
   was slower than f16 here (4.8 vs 0.9 ms at M=128); int4 was fast (1.1 ms).
6. The target GGUF (Inkling-2-Mini abliterated Mix-12G) stores the routed experts as IQ2_XXS,
   IQ2_S, IQ3_XXS, IQ3_S (gate/up) and IQ4_NL (down): non-linear codebooks the NPU cannot
   decode itself. The host decodes them (colibri's `gq_dequant_row`) into the NPU's input
   tensors, which also replaces the memcpy measured above.
