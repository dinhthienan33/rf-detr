@echo off
REM Example script showing how to use run_inference.bat with custom paths
REM Copy this file and modify the paths for your use case

REM Set paths directly
REM Model checkpoint - use pretrained name or path to your trained checkpoint
set CHECKPOINT=rf-detr-base.pth

REM Reference images (3 images required)
set REF_IMAGE1=D:\4th\ZALOAI2025\Track1\dataset\observing\train\samples\Backpack_0\object_images\img_1.jpg
set REF_IMAGE2=D:\4th\ZALOAI2025\Track1\dataset\observing\train\samples\Backpack_0\object_images\img_2.jpg
set REF_IMAGE3=D:\4th\ZALOAI2025\Track1\dataset\observing\train\samples\Backpack_0\object_images\img_3.jpg

REM Target image to detect objects in (must be an image file, not video)
set TARGET_IMAGE=D:\4th\ZALOAI2025\Track1\coco_format\images\Backpack_0_frame_004616.jpg

REM Output directory (must be a directory, not a file)
set OUTPUT_DIR=.\output

REM Similarity threshold (0.0 to 1.0)
set SIMILARITY_THRESHOLD=0.7

REM Device (cuda, cpu, mps)
set DEVICE=cpu

REM Resolution (default: 560)
set RESOLUTION=560

REM Save visualization flag
set SAVE_VISUALIZATION=true

REM Run inference
call run_inference.bat

