@echo off
REM Inference script for One-Shot Object Detection (Windows Batch)
REM Usage: run_inference.bat

REM ============================================
REM Configuration - Modify these paths as needed
REM ============================================

REM Model checkpoint path
REM You can use:
REM - A pretrained checkpoint name (will auto-download): rf-detr-base.pth, rf-detr-large.pth, etc.
REM - Path to your trained checkpoint: path\to\checkpoint.pth
if "%CHECKPOINT%"=="" set CHECKPOINT=rf-detr-base.pth
if "%REF_IMAGE1%"=="" set REF_IMAGE1=ref1.jpg
if "%REF_IMAGE2%"=="" set REF_IMAGE2=ref2.jpg
if "%REF_IMAGE3%"=="" set REF_IMAGE3=ref3.jpg
if "%TARGET_IMAGE%"=="" set TARGET_IMAGE=target.jpg
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=output
if "%SIMILARITY_THRESHOLD%"=="" set SIMILARITY_THRESHOLD=0.7
if "%DEVICE%"=="" set DEVICE=cuda
if "%RESOLUTION%"=="" set RESOLUTION=560
if "%SAVE_VISUALIZATION%"=="" set SAVE_VISUALIZATION=true

REM ============================================
REM Script execution
REM ============================================

echo ==========================================
echo One-Shot Object Detection Inference
echo ==========================================
echo.
echo Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   Reference Images:
echo     - %REF_IMAGE1%
echo     - %REF_IMAGE2%
echo     - %REF_IMAGE3%
echo   Target Image: %TARGET_IMAGE%
echo   Output Directory: %OUTPUT_DIR%
echo   Similarity Threshold: %SIMILARITY_THRESHOLD%
echo   Device: %DEVICE%
echo   Resolution: %RESOLUTION%
echo   Save Visualization: %SAVE_VISUALIZATION%
echo.

REM Check if checkpoint exists (will be auto-downloaded if it's a hosted model name)
REM The inference script will handle downloading pretrained checkpoints automatically
REM Available pretrained checkpoints:
REM   - rf-detr-base.pth (default)
REM   - rf-detr-large.pth
REM   - rf-detr-nano.pth
REM   - rf-detr-small.pth
REM   - rf-detr-medium.pth
REM   - rf-detr-base-o365.pth
REM   - rf-detr-base-2.pth

REM Check if reference images exist
if not exist "%REF_IMAGE1%" (
    echo ERROR: Reference image not found: %REF_IMAGE1%
    echo Please set REF_IMAGE1 environment variable or modify the script.
    exit /b 1
)

if not exist "%REF_IMAGE2%" (
    echo ERROR: Reference image not found: %REF_IMAGE2%
    echo Please set REF_IMAGE2 environment variable or modify the script.
    exit /b 1
)

if not exist "%REF_IMAGE3%" (
    echo ERROR: Reference image not found: %REF_IMAGE3%
    echo Please set REF_IMAGE3 environment variable or modify the script.
    exit /b 1
)

REM Check if target image exists
if not exist "%TARGET_IMAGE%" (
    echo ERROR: Target image not found: %TARGET_IMAGE%
    echo Please set TARGET_IMAGE environment variable or modify the script.
    exit /b 1
)

REM Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Build command
set CMD=python inference_oneshot.py
set CMD=%CMD% --checkpoint %CHECKPOINT%
set CMD=%CMD% --ref_images %REF_IMAGE1% %REF_IMAGE2% %REF_IMAGE3%
set CMD=%CMD% --target_image %TARGET_IMAGE%
set CMD=%CMD% --output_dir %OUTPUT_DIR%
set CMD=%CMD% --similarity_threshold %SIMILARITY_THRESHOLD%
set CMD=%CMD% --device %DEVICE%
set CMD=%CMD% --resolution %RESOLUTION%

if "%SAVE_VISUALIZATION%"=="true" (
    set CMD=%CMD% --save_visualization
)

echo Running inference...
echo Command: %CMD%
echo.

REM Run inference
%CMD%

REM Check exit status
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo Inference completed successfully!
    echo ==========================================
    echo Results saved to: %OUTPUT_DIR%
    echo   - results.json
    if "%SAVE_VISUALIZATION%"=="true" (
        echo   - visualization.jpg
    )
) else (
    echo.
    echo ==========================================
    echo Inference failed!
    echo ==========================================
    exit /b 1
)

