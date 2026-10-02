@echo off
chcp 65001 >nul
REM === lora-train 依存環境の再構築/修復用 (通常は構築済み) ===
setlocal
set ROOT=%~dp0
set PY=%ROOT%venv\Scripts\python.exe

if not exist "%PY%" (
    echo [setup] venv を作成します...
    py -3.10 -m venv "%ROOT%venv"
)

echo [setup] pip 更新...
"%PY%" -m pip install --upgrade pip wheel setuptools

echo [setup] torch 2.6.0 (cu124)...
"%PY%" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

echo [setup] sd-scripts requirements...
pushd "%ROOT%sd-scripts"
"%PY%" -m pip install --upgrade -r requirements.txt
popd

echo [setup] xformers...
"%PY%" -m pip install xformers --index-url https://download.pytorch.org/whl/cu124

echo [setup] onnx / onnxruntime-gpu (WD14 tagger)...
"%PY%" -m pip install onnx onnxruntime-gpu

echo [setup] DONE
pause
