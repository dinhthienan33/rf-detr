# One-Shot Object Detection Inference Guide

This guide explains how to use the inference script for One-Shot Object Detection.

## Quick Start

### Method 1: Using the Bash Script (Recommended)

1. **Make the script executable:**
   ```bash
   chmod +x run_inference.sh
   ```

2. **Set environment variables and run:**
   ```bash
   export CHECKPOINT="path/to/checkpoint.pth"
   export REF_IMAGE1="ref1.jpg"
   export REF_IMAGE2="ref2.jpg"
   export REF_IMAGE3="ref3.jpg"
   export TARGET_IMAGE="target.jpg"
   export OUTPUT_DIR="output"
   
   ./run_inference.sh
   ```

3. **Or modify the script directly:**
   Edit `run_inference.sh` and change the default paths at the top of the file.

### Method 2: Direct Python Command

```bash
python inference_oneshot.py \
    --checkpoint path/to/checkpoint.pth \
    --ref_images ref1.jpg ref2.jpg ref3.jpg \
    --target_image target.jpg \
    --output_dir output \
    --similarity_threshold 0.7 \
    --device cuda \
    --save_visualization
```

## Configuration Options

### Required Parameters

- `--checkpoint`: Path to trained model checkpoint (.pth file)
- `--ref_images`: Three reference images (space-separated)
- `--target_image`: Target image to detect objects in

### Optional Parameters

- `--output_dir`: Output directory (default: `output`)
- `--similarity_threshold`: Similarity threshold for filtering (default: `0.7`)
- `--device`: Device to use - `cuda`, `cpu`, or `mps` (default: `cuda`)
- `--resolution`: Input resolution (default: `560`)
- `--save_visualization`: Flag to save visualization image

## Environment Variables

You can set these environment variables before running `run_inference.sh`:

```bash
export CHECKPOINT="path/to/checkpoint.pth"
export REF_IMAGE1="ref1.jpg"
export REF_IMAGE2="ref2.jpg"
export REF_IMAGE3="ref3.jpg"
export TARGET_IMAGE="target.jpg"
export OUTPUT_DIR="output"
export SIMILARITY_THRESHOLD="0.7"
export DEVICE="cuda"
export RESOLUTION="560"
export SAVE_VISUALIZATION="true"
```

## Output

The script generates:

1. **`results.json`**: JSON file containing:
   - Target image path
   - Reference image paths
   - Number of detections
   - List of detections with bounding boxes and similarity scores

2. **`visualization.jpg`** (if `--save_visualization` is set): Image with bounding boxes drawn

### Example Output JSON

```json
{
  "target_image": "target.jpg",
  "reference_images": ["ref1.jpg", "ref2.jpg", "ref3.jpg"],
  "similarity_threshold": 0.7,
  "num_detections": 2,
  "detections": [
    {
      "box": [100, 200, 300, 400],
      "similarity_score": 0.85
    },
    {
      "box": [500, 600, 700, 800],
      "similarity_score": 0.72
    }
  ]
}
```

## Examples

### Example 1: Basic Usage

```bash
./run_inference.sh
```

(Modify paths in the script first)

### Example 2: Custom Threshold

```bash
export SIMILARITY_THRESHOLD="0.8"
./run_inference.sh
```

### Example 3: CPU Inference

```bash
export DEVICE="cpu"
./run_inference.sh
```

### Example 4: Batch Processing

Create a script to process multiple images:

```bash
#!/bin/bash
for target in data/targets/*.jpg; do
    export TARGET_IMAGE="$target"
    export OUTPUT_DIR="results/$(basename $target .jpg)"
    ./run_inference.sh
done
```

## Troubleshooting

### Error: Checkpoint file not found
- Make sure the checkpoint path is correct
- Check that the file exists and has `.pth` extension

### Error: Reference image not found
- Ensure all 3 reference images exist
- Check file paths are correct

### Error: CUDA out of memory
- Reduce batch size (currently 1)
- Use CPU: `export DEVICE="cpu"`
- Reduce resolution: `export RESOLUTION="384"`

### No detections found
- Try lowering the similarity threshold: `export SIMILARITY_THRESHOLD="0.5"`
- Check that reference images match objects in target image
- Verify model checkpoint is trained for similar objects

## Notes

- The script automatically detects CUDA availability
- Images are preprocessed with SquareResize and normalization
- Bounding boxes are returned in [x1, y1, x2, y2] format
- Similarity scores range from 0.0 to 1.0 (higher = more similar)

