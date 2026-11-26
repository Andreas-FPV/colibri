# Reproducible comparison of colibri backends for Inkling-2-Mini on one Windows machine.
# Every configuration: the same model file, the same prompt (tools/npu_bench.py, ~755 tokens),
# temperature 0, 100 generated tokens, the same expert cache (--cap), a warm-up request, then
# measured runs. Results: bench/results/<label>.json, engine logs: c/tools/npu_bench_<label>.log.
#
#   powershell -File bench\run_npu_benchmarks.ps1 -Model <dir.colibri> [-Cap 32] [-Runs 2]
#       [-OpenVino <dir with openvino.dll>] [-Only cpu,npu,...]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Cap = 32,
    [int]$Runs = 2,
    [string]$OpenVino = "",
    [string[]]$Only = @()
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$c = Join-Path $root "c"
$out = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force $out | Out-Null
if ($OpenVino) { $env:COLI_NPU_OPENVINO = $OpenVino }

# label, environment, extra serve arguments
$configs = @(
    @{ label = "cpu";            env = @("COLI_VULKAN=0");                          args = @() },
    @{ label = "vulkan";         env = @("COLI_VULKAN=1");                          args = @() },
    @{ label = "cpu+npu";        env = @("COLI_VULKAN=0");                          args = @("--npu", "--npu-stats") },
    @{ label = "cpu+npu-min2";   env = @("COLI_VULKAN=0", "COLI_NPU_MIN_ROWS=2");   args = @("--npu", "--npu-stats") },
    @{ label = "cpu+npu-min32";  env = @("COLI_VULKAN=0", "COLI_NPU_MIN_ROWS=32");  args = @("--npu", "--npu-stats") },
    @{ label = "vulkan+npu";     env = @("COLI_VULKAN=1");                          args = @("--npu", "--npu-stats") }
)
foreach ($cfg in $configs) {
    if ($Only.Count -and -not ($Only -contains $cfg.label)) { continue }
    $a = @("tools\npu_bench.py", "--model", $Model, "--label", $cfg.label, "--runs", "$Runs",
           "--out", (Join-Path $out "$($cfg.label).json"), "--serve-arg=--cap", "--serve-arg=$Cap")
    foreach ($e in $cfg.env) { $a += @("--env", $e) }
    foreach ($s in $cfg.args) { $a += "--serve-arg=$s" }
    Write-Host "== $($cfg.label)"
    Push-Location $c
    try { & py -3 @a } finally { Pop-Location }
}
