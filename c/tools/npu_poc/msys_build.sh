#!/usr/bin/env bash
# Build colibri's engines from an MSYS2 UCRT64 toolchain, e.g. from PowerShell:
#   $env:MSYSTEM='UCRT64'; C:\msys64\usr\bin\bash.exe -l c/tools/npu_poc/msys_build.sh inkling.exe VK=1
export PATH=/ucrt64/bin:/usr/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
make -j"$(nproc)" "$@" 2>&1 | tail -n 25
exit "${PIPESTATUS[0]}"
