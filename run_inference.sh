#!/bin/bash

# Inference script for One-Shot Object Detection
# Usage: ./run_inference.sh

# ============================================
# Configuration - Modify these paths as needed
# ============================================

# Model checkpoint path
# You can use:
# - A pretrained checkpoint name (will auto-download): rf-detr-base.pth, rf-detr-large.pth, etc.
# - Path to your trained checkpoint: path/to/checkpoint.pth
CHECKPOINT="${CHECKPOINT:-rf-detr-base.pth}"

# Reference images (3 images required)
# For Windows: Use forward slashes or set via environment variable
# Example: export REF_IMAGE1="D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_1.jpg"
REF_IMAGE1="${REF_IMAGE1:-D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_1.jpg}"
REF_IMAGE2="${REF_IMAGE2:-D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_2.jpg}"
REF_IMAGE3="${REF_IMAGE3:-D:/4th/ZALOAI2025/Track1/dataset/observing/train/samples/Backpack_0/object_images/img_3.jpg}"

# Target image to detect objects in
# Note: Should be an image file (.jpg, .png), NOT a video file (.mp4)
# If you have a video, extract a frame first using ffmpeg
TARGET_IMAGE="${TARGET_IMAGE:-D:/4th/ZALOAI2025/Track1/coco_format/images/Backpack_0_frame_004616.jpg}"

# Output directory (must be a directory, not a file)
OUTPUT_DIR="${OUTPUT_DIR:-D:/4th/ZALOAI2025/Track1/output}"

# Similarity threshold (0.0 to 1.0)
SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.7}"

# Device (cuda, cpu, mps)
DEVICE="${DEVICE:-cuda}"

# Resolution (default: 560)
RESOLUTION="${RESOLUTION:-560}"

# Save visualization flag
SAVE_VISUALIZATION="${SAVE_VISUALIZATION:-true}"

# ============================================
# Script execution
# ============================================

echo "=========================================="
echo "One-Shot Object Detection Inference"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Reference Images:"
echo "    - $REF_IMAGE1"
echo "    - $REF_IMAGE2"
echo "    - $REF_IMAGE3"
echo "  Target Image: $TARGET_IMAGE"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Similarity Threshold: $SIMILARITY_THRESHOLD"
echo "  Device: $DEVICE"
echo "  Resolution: $RESOLUTION"
echo "  Save Visualization: $SAVE_VISUALIZATION"
echo ""

# Check if checkpoint exists (will be auto-downloaded if it's a hosted model name)
# The inference script will handle downloading pretrained checkpoints automatically
# Available pretrained checkpoints:
#   - rf-detr-base.pth (default)
#   - rf-detr-large.pth
#   - rf-detr-nano.pth
#   - rf-detr-small.pth
#   - rf-detr-medium.pth
#   - rf-detr-base-o365.pth
#   - rf-detr-base-2.pth

# Convert Windows paths to Unix format if running on Windows (Git Bash/WSL)
# This handles paths like D:\path\to\file -> /d/path/to/file
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Convert Windows drive letters to Unix format
    REF_IMAGE1=$(echo "$REF_IMAGE1" | sed 's|^\([A-Z]\):|/\L\1|' | sed 's|\\|/|g')
    REF_IMAGE2=$(echo "$REF_IMAGE2" | sed 's|^\([A-Z]\):|/\L\1|' | sed 's|\\|/|g')
    REF_IMAGE3=$(echo "$REF_IMAGE3" | sed 's|^\([A-Z]\):|/\L\1|' | sed 's|\\|/|g')
    TARGET_IMAGE=$(echo "$TARGET_IMAGE" | sed 's|^\([A-Z]\):|/\L\1|' | sed 's|\\|/|g')
    OUTPUT_DIR=$(echo "$OUTPUT_DIR" | sed 's|^\([A-Z]\):|/\L\1|' | sed 's|\\|/|g')
fi

# Check if reference images exist
for ref_img in "$REF_IMAGE1" "$REF_IMAGE2" "$REF_IMAGE3"; do
    if [ ! -f "$ref_img" ]; then
        echo "ERROR: Reference image not found: $ref_img"
        echo "Please set REF_IMAGE1, REF_IMAGE2, REF_IMAGE3 environment variables or modify the script."
        exit 1
    fi
done

# Check if target image exists
if [ ! -f "$TARGET_IMAGE" ]; then
    echo "ERROR: Target image not found: $TARGET_IMAGE"
    echo "Please set TARGET_IMAGE environment variable or modify the script."
    echo "Note: Target should be an image file (.jpg, .png), not a video file."
    exit 1
fi

# Check if target is a video file (common mistake)
if [[ "$TARGET_IMAGE" == *.mp4 ]] || [[ "$TARGET_IMAGE" == *.avi ]] || [[ "$TARGET_IMAGE" == *.mov ]]; then
    echo "WARNING: Target appears to be a video file: $TARGET_IMAGE"
    echo "Please extract a frame from the video first, or use an image file."
    echo "You can use ffmpeg to extract a frame:"
    echo "  ffmpeg -i $TARGET_IMAGE -ss 00:00:01 -vframes 1 output_frame.jpg"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python inference_oneshot.py"
CMD="$CMD --checkpoint $CHECKPOINT"
CMD="$CMD --ref_images $REF_IMAGE1 $REF_IMAGE2 $REF_IMAGE3"
CMD="$CMD --target_image $TARGET_IMAGE"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --similarity_threshold $SIMILARITY_THRESHOLD"
CMD="$CMD --device $DEVICE"
CMD="$CMD --resolution $RESOLUTION"

if [ "$SAVE_VISUALIZATION" = "true" ]; then
    CMD="$CMD --save_visualization"
fi

echo "Running inference..."
echo "Command: $CMD"
echo ""

# Run inference
$CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Inference completed successfully!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "  - results.json"
    if [ "$SAVE_VISUALIZATION" = "true" ]; then
        echo "  - visualization.jpg"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ Inference failed!"
    echo "=========================================="
    exit 1
fi

