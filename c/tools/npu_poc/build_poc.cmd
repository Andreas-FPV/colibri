@echo off
rem Build the NPU feasibility probe with MSVC against an OpenVINO runtime.
rem   OPENVINO_DIR: a directory holding include\ and lib\ (or libs\) of OpenVINO.
rem   Default: the "openvino" package of a Python environment (pip install openvino).
setlocal
if "%OPENVINO_DIR%"=="" (
  echo set OPENVINO_DIR to an OpenVINO runtime, e.g. ^<venv^>\Lib\site-packages\openvino
  exit /b 1
)
set OVLIB=%OPENVINO_DIR%\libs
if not exist "%OVLIB%\openvino.lib" set OVLIB=%OPENVINO_DIR%\lib\intel64\Release
if "%VCVARS%"=="" set VCVARS=C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /MD /W3 /I"%OPENVINO_DIR%\include" npu_poc.cpp /Fe:npu_poc.exe /link /LIBPATH:"%OVLIB%" openvino.lib || exit /b 1
echo built %~dp0npu_poc.exe (run with %OVLIB% on PATH)
