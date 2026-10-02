@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
set ROOT=%~dp0
set PY=%ROOT%venv\Scripts\python.exe
set SC=%ROOT%scripts

:menu
echo.
echo ============================================
echo    LoRA Style Training Tool  (Illustrious SDXL / 8GB)
echo ============================================
echo    0) Download base model (Illustrious-XL-v0.1)
echo    1) Prepare dataset   (datasets\PROJECT\raw -^> img)
echo    2) Auto-tag (WD14 tagger)
echo    3) Make training config (toml)
echo    4) TRAIN
echo    5) Test generate (via reForge API)
echo    A) Auto: run 1 -^> 2 -^> 3 -^> 4
echo    Q) Quit
echo --------------------------------------------
set /p c="Select: "

if /i "%c%"=="0" ( "%PY%" "%SC%\00_download_base.py" & goto done )
if /i "%c%"=="1" ( "%PY%" "%SC%\01_prepare.py"       & goto done )
if /i "%c%"=="2" ( "%PY%" "%SC%\02_tag.py"           & goto done )
if /i "%c%"=="3" ( "%PY%" "%SC%\03_make_config.py"   & goto done )
if /i "%c%"=="4" ( "%PY%" "%SC%\04_train.py"         & goto done )
if /i "%c%"=="5" ( "%PY%" "%SC%\05_test.py"          & goto done )
if /i "%c%"=="A" (
    "%PY%" "%SC%\01_prepare.py"     || goto done
    "%PY%" "%SC%\02_tag.py"         || goto done
    "%PY%" "%SC%\03_make_config.py" || goto done
    "%PY%" "%SC%\04_train.py"
    goto done
)
if /i "%c%"=="Q" ( exit /b 0 )
echo Invalid choice.
goto menu

:done
echo.
pause
goto menu
