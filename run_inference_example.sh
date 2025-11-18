#!/bin/bash

# Example script showing how to use run_inference.sh with custom paths
# Copy this file and modify the paths for your use case

# Example: Set paths directly in the script
# Model checkpoint - use pretrained name or path to your trained checkpoint
export CHECKPOINT="rf-detr-base.pth"

# Reference images (3 images required)
# For Windows paths, use forward slashes: D:/path/to/file.jpg
export REF_IMAGE1="D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_1.jpg"
export REF_IMAGE2="D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_2.jpg"
export REF_IMAGE3="D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_3.jpg"

# Target image to detect objects in (must be an image file, not video)
export TARGET_IMAGE="D:/4th/ZALOAI2025/Track1/coco_format/images/Backpack_0_frame_004616.jpg"

# Output directory (must be a directory, not a file)
export OUTPUT_DIR="D:/4th/ZALOAI2025/Track1/output"

# Similarity threshold (0.0 to 1.0)
export SIMILARITY_THRESHOLD="0.7"

# Device (cuda, cpu, mps)
export DEVICE="cuda"

# Resolution (default: 560)
export RESOLUTION="560"

# Save visualization flag
export SAVE_VISUALIZATION="true"

# Run inference
./run_inference.sh

