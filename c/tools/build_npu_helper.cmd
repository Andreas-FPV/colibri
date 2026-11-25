@echo off
rem Build coli_npu.dll, the optional Intel NPU helper (docs/intel-npu.md), with MSVC.
rem   OPENVINO_DIR  an OpenVINO runtime: a directory with include\ and lib\intel64\Release\
rem                 (the archive) or include\ and libs\ (the pip wheel's site-packages\openvino)
rem   VCVARS        vcvars64.bat of Visual Studio / Build Tools 2022 (found with vswhere if unset)
rem The DLL is written beside the engines (c\coli_npu.dll).
setlocal
if "%OPENVINO_DIR%"=="" (
  echo set OPENVINO_DIR to an OpenVINO runtime ^(e.g. ^<venv^>\Lib\site-packages\openvino^)
  exit /b 1
)
set OVLIB=%OPENVINO_DIR%\libs
if not exist "%OVLIB%\openvino.lib" set OVLIB=%OPENVINO_DIR%\lib\intel64\Release
if not exist "%OVLIB%\openvino.lib" (
  echo openvino.lib not found under %OPENVINO_DIR%
  exit /b 1
)
if "%VCVARS%"=="" for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0.."
cl /nologo /O2 /EHsc /std:c++17 /MD /W4 /LD /external:I"%OPENVINO_DIR%\include" /external:W0 backend_npu_helper.cpp ^
   /Fe:coli_npu.dll /Fo:backend_npu_helper.obj /link /LIBPATH:"%OVLIB%" openvino.lib || exit /b 1
del /q coli_npu.exp coli_npu.lib backend_npu_helper.obj 2>nul
echo built %CD%\coli_npu.dll  (OpenVINO's DLLs: beside it, on PATH, or in COLI_NPU_OPENVINO=%OVLIB%)
