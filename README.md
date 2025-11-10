# RF-DETR: SOTA Real-Time Detection and Segmentation Model

[![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
[![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)

[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SkalskiP/RF-DETR)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/rf-detr)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR is a real-time, transformer-based object detection and instance segmentation model architecture developed by Roboflow and released under the Apache 2.0 license.

RF-DETR is the first real-time model to exceed 60 AP on the [Microsoft COCO object detection benchmark](https://cocodataset.org/#home) alongside competitive performance at base sizes. It also achieves state-of-the-art performance on [RF100-VL](https://github.com/roboflow/rf100-vl), an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is fastest and most accurate for its size when compared current real-time objection models.

On image segmentation, RF-DETR Seg (Preview) is 3x faster and more accurate than the largest YOLO when evaluated on the Microsoft COCO Segmentation benchmark, defining a new real-time state-of-the-art for the industry-standard benchmark in segmentation model evaluation.

[![rf-detr-tutorial-banner](https://github.com/user-attachments/assets/555a45c3-96e8-4d8a-ad29-f23403c8edfd)](https://youtu.be/-OvpdLAElFA)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Model Variants](#model-variants)
- [News](#news)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)
- [Contribute](#contribute)

## Overview

RF-DETR (Roboflow Detection Transformer) represents a breakthrough in real-time computer vision, combining the accuracy of transformer-based architectures with the speed required for production deployments. Built on the foundations of DETR (DEtection TRansformer) and enhanced with novel optimizations, RF-DETR delivers state-of-the-art performance across multiple benchmarks while maintaining inference speeds suitable for edge devices and real-time applications.

### What Makes RF-DETR Special?

**Transformer-Based Architecture**: Unlike traditional CNN-based detectors, RF-DETR uses a pure transformer architecture that processes images holistically, enabling better context understanding and more accurate predictions.

**Real-Time Performance**: Achieves inference speeds of 2-5ms depending on model size, making it suitable for video processing, robotics, and edge deployment scenarios.

**Domain Adaptability**: Exceptional performance on RF100-VL benchmark demonstrates superior generalization to diverse real-world scenarios compared to other state-of-the-art models.

**Multi-Task Support**: Seamlessly handles both object detection and instance segmentation tasks with a unified architecture.

**Production-Ready**: Includes built-in optimization tools, ONNX export, and integration with Roboflow's Inference server for easy deployment.

## Key Features

### Core Capabilities

- **🎯 Object Detection**: Detect and localize objects with bounding boxes across 80 COCO classes
- **🖼️ Instance Segmentation**: Generate precise pixel-level masks for detected objects (RF-DETR Seg Preview)
- **⚡ Real-Time Inference**: Optimized for speed with sub-5ms latency on modern GPUs
- **🔄 Batch Processing**: Efficient batch inference for processing multiple images simultaneously
- **📦 Multiple Model Sizes**: Choose from Nano, Small, Medium, and Large variants based on your accuracy/speed requirements

### Training & Fine-Tuning

- **🎓 Transfer Learning**: Fine-tune on custom datasets starting from COCO-pretrained weights
- **💾 Checkpoint Management**: Automatic saving of best models with EMA (Exponential Moving Average) support
- **🔄 Resume Training**: Continue training from saved checkpoints with full optimizer state restoration
- **🛑 Early Stopping**: Automatic training termination when validation performance plateaus
- **🔧 Gradient Checkpointing**: Reduce memory usage for training larger models
- **📊 Multi-GPU Training**: Distributed training support via PyTorch DDP
- **📈 Metrics Logging**: Integration with TensorBoard and Weights & Biases

### Deployment & Optimization

- **🚀 Inference Optimization**: Built-in `optimize_for_inference()` method for up to 2x speedup
- **📤 ONNX Export**: Export models to ONNX format for cross-platform deployment
- **☁️ Cloud Deployment**: Seamless integration with Roboflow Inference server
- **🖥️ Edge Support**: Optimized for deployment on edge devices and embedded systems
- **🔌 Framework Integration**: Easy integration with OpenCV, Supervision, and other CV frameworks

### Developer Experience

- **🐍 Pythonic API**: Clean, intuitive Python interface for all operations
- **📝 Type Hints**: Full type annotation support for better IDE integration
- **📚 Comprehensive Documentation**: Extensive guides, tutorials, and API reference
- **🧪 Example Notebooks**: Ready-to-run Google Colab notebooks for common tasks
- **🤝 Active Community**: Discord server and GitHub discussions for support

## Architecture

RF-DETR builds upon the DETR (DEtection TRansformer) architecture with several key innovations that enable real-time performance:

### Core Components

**1. Vision Backbone (DINOv2)**
- Uses DINOv2 as the vision encoder for robust feature extraction
- Self-supervised pre-training provides strong visual representations
- Windowed attention mechanism reduces computational complexity
- Multi-scale feature extraction for detecting objects at various sizes

**2. Transformer Encoder-Decoder**
- Lightweight transformer design optimized for speed
- Deformable attention mechanism focuses computation on relevant spatial locations
- Multi-scale deformable attention reduces memory footprint
- Query-based object detection eliminates need for anchor boxes

**3. Detection Head**
- Parallel classification and bounding box regression branches
- Hungarian matching for optimal assignment during training
- Set-based loss functions eliminate duplicate predictions
- Confidence scoring for reliable post-processing

**4. Segmentation Head (RF-DETR Seg)**
- Efficient mask prediction branch built on detection features
- Shares backbone computation with detection for minimal overhead
- Dynamic mask generation conditioned on detected objects
- High-resolution mask output for precise instance segmentation

### Technical Innovations

- **Optimized Attention**: Deformable attention reduces complexity from O(N²) to O(N)
- **Feature Pyramid Integration**: Multi-scale features enable detection across object sizes
- **Efficient Decoding**: Streamlined decoder with fewer layers maintains accuracy while reducing latency
- **Export-Friendly Design**: Architecture choices enable efficient ONNX conversion and deployment

## Model Variants

RF-DETR offers four model sizes to balance accuracy and speed based on your requirements:

| Model | Resolution | Params | COCO AP | Latency | Best For |
|-------|-----------|--------|---------|---------|----------|
| **RF-DETR-N** (Nano) | 384×384 | 30.5M | 48.4 | 2.32ms | Edge devices, maximum speed |
| **RF-DETR-S** (Small) | 512×512 | 32.1M | 53.0 | 3.52ms | Balanced speed and accuracy |
| **RF-DETR-M** (Medium) | 576×576 | 33.7M | 54.7 | 4.52ms | High accuracy real-time applications |
| **RF-DETR-L** (Large) | 640×640 | ~40M | TBD | ~6ms | Maximum accuracy |

**RF-DETR Seg (Preview)**: Available at multiple resolutions (312×312, 384×384, 432×432) for instance segmentation tasks.

### Choosing the Right Model

- **Nano**: Best for mobile devices, embedded systems, or when processing many video streams
- **Small**: Ideal for single-stream video processing with good accuracy
- **Medium**: Recommended for most applications; excellent accuracy-speed tradeoff
- **Large**: When accuracy is paramount and computational resources are available

## News

- `2025/10/02`: We release RF-DETR-Seg (Preview), a preview of our instance segmentation head for RF-DETR.
- `2025/07/23`: We release three new checkpoints for RF-DETR: Nano, Small, and Medium.
    - RF-DETR Base is now deprecated. We recommend using RF-DETR Medium which offers subtantially better accuracy at comparable latency.
- `2025/05/16`: We release an 'optimize_for_inference' method which speeds up native PyTorch by up to 2x, depending on platform.
- `2025/04/03`: We release early stopping, gradient checkpointing, metrics saving, training resume, TensorBoard and W&B logging support.
- `2025/03/20`: We release RF-DETR real-time object detection model. **Code and checkpoint for RF-DETR-large and RF-DETR-base are available.**

## Results

RF-DETR achieves state-of-the-art performance on both the Microsoft COCO and the RF100-VL benchmarks, demonstrating superior accuracy and speed compared to existing real-time detection models.

### Understanding the Benchmarks

**Microsoft COCO**: The industry-standard benchmark for object detection with 80 object categories. Metrics include:
- **AP<sub>50:95</sub>**: Average Precision across IoU thresholds from 0.5 to 0.95 (primary metric)
- **AP<sub>50</sub>**: Average Precision at IoU threshold of 0.5 (easier detections)

**RF100-VL**: A domain adaptability benchmark measuring how well models generalize to diverse real-world scenarios across 100 datasets. This metric is crucial for production deployments where domain shift is common.

**Latency**: End-to-end inference time measured on NVIDIA T4 GPU with TensorRT optimization, including pre-processing and post-processing.

### Object Detection Benchmarks

The pareto curve below shows RF-DETR's superior accuracy-speed tradeoff compared to other real-time models:

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto1.png)

**Key Highlights**:
- 🏆 **RF-DETR-M achieves 54.7 COCO AP** at only 4.52ms latency (comparable to YOLO11-M's latency but +6.1 AP higher)
- 🎯 **Exceptional RF100-VL performance** demonstrates superior real-world domain adaptability
- ⚡ **Sub-5ms latency** across all model sizes enables true real-time applications
- 📊 **Better accuracy at every speed point** compared to YOLO11 and D-FINE models

| Architecture | COCO AP<sub>50</sub> |  COCO AP<sub>50:95</sub>   |  RF100VL AP<sub>50</sub>   | RF100VL AP<sub>50:95</sub>  |  Latency (ms)   |   Params (M) |   Resolution  |
|:------------:|:--------------------:|:--------------------------:|:--------------------------:|:---------------------------:|:---------------:|:------------:|:-------------:|
|  RF-DETR-N   |         67.6         |            48.4            |            84.1            |            57.1             |      2.32       |         30.5 |       384x384 |
|  RF-DETR-S   |         72.1         |            53.0            |            85.9            |            59.6             |      3.52       |         32.1 |       512x512 |
|  RF-DETR-M   |         73.6         |            54.7            |            86.6            |            60.6             |      4.52       |         33.7 |       576x576 |
|   YOLO11-N   |         52.0         |            37.4            |            81.4            |            55.3             |      2.49       |          2.6 |       640x640 |
|   YOLO11-S   |         59.7         |            44.4            |            82.3            |            56.2             |      3.16       |          9.4 |       640x640 |
|   YOLO11-M   |         64.1         |            48.6            |            82.5            |            56.5             |      5.13       |         20.1 |       640x640 |
|   YOLO11-L   |         65.3         |            50.2            |             x              |              x              |      6.65       |         25.3 |       640x640 |
|   YOLO11-X   |         66.5         |            51.2            |             x              |              x              |      11.92      |         56.9 |       640x640 |
|  LW-DETR-T   |         60.7         |            42.9            |             x              |              x              |      1.91       |         12.1 |       640x640 |
|  LW-DETR-S   |         66.8         |            48.0            |            84.5            |            58.0             |      2.62       |         14.6 |       640x640 |
|  LW-DETR-M   |         72.0         |            52.6            |            85.2            |            59.4             |      4.49       |         28.2 |       640x640 |
|   D-FINE-N   |         60.2         |            42.7            |            83.6            |            57.7             |      2.12       |          3.8 |       640x640 |
|   D-FINE-S   |         67.6         |            50.7            |            84.5            |            59.9             |      3.55       |         10.2 |       640x640 |
|   D-FINE-M   |         72.6         |            55.1            |            84.6            |            60.2             |      5.68       |         19.2 |       640x640 |

[See our benchmark notes in the RF-DETR documentation.](https://rfdetr.roboflow.com/learn/benchmarks/)

_We are actively working on RF-DETR Large and X-Large models using the same techniques we used to achieve the strong accuracy that RF-DETR Medium attains. This is why RF-DETR Large and X-Large is not yet reported on our pareto charts and why we haven't benchmarked other models at similar sizes. Check back in the next few weeks for the launch of new RF-DETR Large and X-Large models._

### Instance Segmentation Benchmarks

RF-DETR Seg (Preview) establishes new state-of-the-art performance for real-time instance segmentation:

![rf-detr-coco-rf100-vl-9](https://media.roboflow.com/rfdetr/pareto_segmentation.png)

**Key Highlights**:
- 🥇 **44.3 mAP at 5.6ms** - 3x faster than YOLO11x-Seg with higher accuracy
- 💪 **Significantly outperforms YOLOv8/v11** across all model sizes
- 🎯 **Unified architecture** shares backbone with detection for efficient multi-task learning

| Model Name              | Reported Latency | Reported mAP | Measured Latency | Measured mAP |
|-------------------------|------------------|--------------|------------------|--------------|
| RF-DETR Seg-Preview@312 |                  |              | 3.3              | 39.4         |
| YOLO11n-Seg             | 1.8              | 32.0         | 3.6              | 30.0         |
| YOLOv8n-Seg             |                  | 30.5         | 3.5              | 28.3         |
| RF-DETR Seg-Preview@384 |                  |              | 4.5              | 42.7         |
| YOLO11s-Seg             | 2.9              | 37.8         | 4.6              | 35.0         |
| YOLOv8s-Seg             |                  | 36.8         | 4.2              | 34.0         |
| RF-DETR Seg-Preview@432 |                  |              | 5.6              | 44.3         |
| YOLO11m-Seg             | 6.3              | 41.5         | 6.9              | 38.5         |
| YOLOv8m-Seg             |                  | 40.8         | 7.0              | 37.3         |
| YOLO11l-Seg             | 7.8              | 42.9         | 8.3              | 39.5         |
| YOLOv8l-Seg             |                  | 42.6         | 9.7              | 39.0         |
| YOLO11x-Seg             | 15.8             | 43.8         | 13.7             | 40.1         |
| YOLOv8x-Seg             |                  | 43.4         | 14.0             | 39.5         |

For more information on measuring end-to-end latency for models, see our open source [Single Artifact Benchmarking tool](https://github.com/roboflow/single_artifact_benchmarking).

## Installation

To install RF-DETR, install the `rfdetr` package in a [**Python>=3.9**](https://www.python.org/) environment with `pip`:

```bash
pip install rfdetr
```

<details>
<summary>Install from source</summary>

<br>

By installing RF-DETR from source, you can explore the most recent features and enhancements that have not yet been officially released. Please note that these updates are still in development and may not be as stable as the latest published release.

```bash
pip install git+https://github.com/roboflow/rf-detr.git
```

</details>

## Quick Start

Get up and running with RF-DETR in minutes:

### 1. Install RF-DETR

```bash
pip install rfdetr
```

### 2. Run Detection on an Image

```python
from rfdetr import RFDETRMedium
from PIL import Image

# Load model
model = RFDETRMedium()

# Run inference
image = Image.open("path/to/your/image.jpg")
detections = model.predict(image, threshold=0.5)

# Results contain bounding boxes, class IDs, and confidence scores
print(f"Found {len(detections)} objects")
```

### 3. Visualize Results

```python
import supervision as sv
from rfdetr.util.coco_classes import COCO_CLASSES

# Create labels
labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# Annotate image
annotated = image.copy()
annotated = sv.BoxAnnotator().annotate(annotated, detections)
annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)

# Display or save
sv.plot_image(annotated)
```

### 4. Train on Custom Data

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()

model.train(
    dataset_dir="path/to/coco/dataset",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="outputs"
)
```

That's it! You're ready to use RF-DETR. Check the [full documentation](https://rfdetr.roboflow.com) for advanced features.

## Inference

The easiest path to deployment is using Roboflow's [Inference](https://github.com/roboflow/inference) package. 

The code below lets you run `rfdetr-base` on an image:

```python
import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

model = get_model("rfdetr-base")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)
```

To use segmentation, use the `rfdetr-seg-preview` model ID. This model will return segmentation masks from a RF-DETR-Seg (Preview) model trained on the Microsoft COCO dataset.

## Predict

You can also use the .predict method to perform inference during local development. The `.predict()` method accepts various input formats, including file paths, PIL images, NumPy arrays, and torch tensors. Please ensure inputs use RGB channel order. For `torch.Tensor` inputs specifically, they must have a shape of `(3, H, W)` with values normalized to the `[0..1)` range. If you don't plan to modify the image or batch size dynamically at runtime, you can also use `.optimize_for_inference()` to get up to 2x end-to-end speedup, depending on platform.

```python
import io
import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()

model.optimize_for_inference()

url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"

image = Image.open(io.BytesIO(requests.get(url).content))
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

### Train a Model

You can fine-tune an RF-DETR Nano, Small, Medium, and Base model with a custom dataset using the `rfdetr` Python package.

[Learn how to train an RF-DETR model.](https://rfdetr.roboflow.com/learn/train/)

## Use Cases

RF-DETR's combination of accuracy and speed makes it suitable for a wide range of applications:

### 🚗 Autonomous Vehicles
- Real-time object detection for self-driving cars
- Pedestrian and vehicle detection with high accuracy
- Multi-object tracking at video frame rates
- Lane detection and road scene understanding

### 🏭 Industrial Automation
- Quality control and defect detection on production lines
- Real-time inventory monitoring in warehouses
- Safety compliance monitoring in manufacturing facilities
- Automated sorting and classification systems

### 🛡️ Security & Surveillance
- Real-time threat detection in video feeds
- People counting and crowd analysis
- Intrusion detection systems
- Suspicious behavior recognition

### 🏥 Healthcare & Medical Imaging
- Cell detection and counting in microscopy images
- Anomaly detection in medical scans
- Surgical instrument tracking
- Patient monitoring systems

### 🏬 Retail & E-Commerce
- Automated checkout systems
- Shelf monitoring and inventory management
- Customer behavior analysis
- Product recognition and search

### 🤖 Robotics
- Object manipulation and grasping
- Navigation and obstacle avoidance
- Human-robot interaction
- Agricultural automation (crop detection, pest identification)

### 📱 Mobile & Edge Applications
- Augmented reality applications
- Real-time translation of visual content
- Accessibility tools for visually impaired users
- Smart home automation

### 🎮 Entertainment & Media
- Automated video tagging and content moderation
- Sports analytics and player tracking
- Virtual production and special effects
- Interactive gaming experiences

## Project Structure

Understanding the repository structure helps navigate and contribute to the project:

```
rf-detr/
├── rfdetr/                      # Main package directory
│   ├── models/                  # Model architectures
│   │   ├── backbone/           # Vision backbone implementations (DINOv2)
│   │   ├── ops/                # Custom operations (deformable attention)
│   │   ├── transformer.py      # Transformer encoder/decoder
│   │   ├── matcher.py          # Hungarian matcher for training
│   │   ├── position_encoding.py # Positional encoding
│   │   ├── segmentation_head.py # Instance segmentation head
│   │   └── lwdetr.py           # LW-DETR base implementation
│   ├── datasets/               # Dataset loading and preprocessing
│   │   ├── coco.py            # COCO dataset handler
│   │   ├── transforms.py       # Data augmentation
│   │   └── o365.py            # Objects365 dataset handler
│   ├── deploy/                 # Deployment utilities
│   │   └── onnx_export.py     # ONNX export functionality
│   ├── util/                   # Utility functions
│   │   ├── coco_classes.py    # COCO class definitions
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── misc.py            # Miscellaneous utilities
│   │   └── utils.py           # Helper functions
│   ├── cli/                    # Command-line interface
│   ├── config.py              # Configuration classes
│   ├── detr.py                # Main RF-DETR classes
│   ├── engine.py              # Training and evaluation engine
│   └── main.py                # Core model logic
├── docs/                       # Documentation source
│   ├── learn/                 # Learning resources
│   │   ├── train/             # Training guides
│   │   ├── pretrained.md      # Using pretrained models
│   │   ├── deploy.md          # Deployment guides
│   │   └── benchmarks.md      # Benchmark details
│   ├── reference/             # API reference
│   └── tutorials/             # Step-by-step tutorials
├── tests/                      # Test suite (if applicable)
├── .github/                    # GitHub configuration
│   ├── workflows/             # CI/CD workflows
│   └── ISSUE_TEMPLATE/        # Issue templates
├── pyproject.toml             # Project metadata and dependencies
├── mkdocs.yaml                # Documentation configuration
├── README.md                  # This file
├── LICENSE                    # Apache 2.0 license
├── CONTRIBUTING.md            # Contribution guidelines
└── CITATION.cff              # Citation information
```

### Key Files

- **`rfdetr/detr.py`**: Main entry point defining `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`, and `RFDETRSegPreview` classes
- **`rfdetr/config.py`**: Configuration dataclasses for models and training
- **`rfdetr/engine.py`**: Training loop, evaluation, and metrics computation
- **`rfdetr/models/lwdetr.py`**: Core detection model implementation
- **`rfdetr/models/segmentation_head.py`**: Instance segmentation head
- **`rfdetr/datasets/coco.py`**: COCO dataset loading with transforms

## Requirements and Dependencies

### System Requirements

- **Python**: 3.9 or higher
- **CUDA**: 11.7+ (for GPU acceleration)
- **RAM**: 16GB minimum, 32GB recommended for training
- **GPU**: NVIDIA GPU with 8GB+ VRAM for training, 4GB+ for inference
- **Storage**: 10GB for model weights and cache

### Core Dependencies

RF-DETR relies on the following major packages:

- **PyTorch** (>=1.13.0): Deep learning framework
- **torchvision** (>=0.14.0): Computer vision utilities
- **timm**: PyTorch image models
- **transformers**: Hugging Face transformers library
- **supervision**: Computer vision utilities for visualization
- **pycocotools**: COCO dataset utilities
- **fairscale**: Training utilities for distributed training
- **einops**: Tensor operations
- **roboflow**: Dataset management and deployment

### Optional Dependencies

Install additional features with:

```bash
# ONNX export support
pip install rfdetr[onnxexport]

# Metrics and logging (TensorBoard, Weights & Biases)
pip install rfdetr[metrics]

# Documentation building
pip install rfdetr[docs]

# All extras
pip install rfdetr[onnxexport,metrics,docs]
```

## Troubleshooting & FAQ

### Common Issues

**Q: I'm getting CUDA out of memory errors during training**

A: Try these solutions:
- Reduce `batch_size` and increase `grad_accum_steps` to maintain effective batch size
- Enable `gradient_checkpointing=True` to trade compute for memory
- Use a smaller model variant (Nano or Small)
- Reduce input resolution

**Q: Inference is slower than expected**

A: Optimize your setup:
- Call `model.optimize_for_inference()` before running predictions
- Use ONNX export for production deployments
- Ensure you're using GPU (`device='cuda'`)
- Use batch inference when processing multiple images

**Q: Model accuracy is poor on my custom dataset**

A: Consider these improvements:
- Increase training epochs (100+ recommended)
- Verify annotation quality and format
- Use appropriate data augmentation
- Try a larger model variant
- Adjust learning rate (`lr` parameter)

**Q: How do I convert my dataset to COCO format?**

A: Options:
- Use [Roboflow](https://roboflow.com) to annotate and export in COCO format
- Use conversion scripts for common formats (YOLO, Pascal VOC, etc.)
- Follow the COCO JSON format specification manually

**Q: Can I train on a dataset with different classes than COCO?**

A: Yes! RF-DETR supports custom datasets with any number of classes. Just ensure your COCO JSON annotations include the correct class definitions.

**Q: How do I resume training after interruption?**

A: Use the `resume` parameter:
```python
model.train(
    dataset_dir="path/to/dataset",
    output_dir="path/to/output",
    resume="path/to/output/checkpoint.pth"
)
```

**Q: What's the difference between `checkpoint.pth` and `checkpoint_best_total.pth`?**

A: 
- `checkpoint.pth`: Latest training state (includes optimizer, scheduler)
- `checkpoint_best_total.pth`: Best model weights only (for inference)

### Performance Tips

1. **Use Mixed Precision Training**: Automatic mixed precision (AMP) is enabled by default
2. **Optimize Data Loading**: Increase `num_workers` in data loader settings
3. **Monitor GPU Utilization**: Use `nvidia-smi` to ensure full GPU usage
4. **Profile Your Code**: Use PyTorch profiler to identify bottlenecks
5. **Batch Inference**: Process multiple images together for better throughput

### Getting Help

- 📖 Check the [documentation](https://rfdetr.roboflow.com)
- 💬 Join our [Discord community](https://discord.gg/GbfgXGJ8Bk)
- 🐛 Report bugs on [GitHub Issues](https://github.com/roboflow/rf-detr/issues)
- 📧 Contact [support@roboflow.com](mailto:support@roboflow.com) for enterprise support

## Documentation

Visit our [documentation website](https://rfdetr.roboflow.com) to learn more about how to use RF-DETR.

## License

Both the code and the weights pretrained on the COCO dataset are released under the [Apache 2.0 license](https://github.com/roboflow/r-flow/blob/main/LICENSE).

## Acknowledgements

RF-DETR builds upon outstanding research and open-source contributions from the computer vision community. We are grateful to the authors and maintainers of the following works:

### Foundation Models

**[LW-DETR](https://arxiv.org/pdf/2406.03459)** (Lightweight Detection Transformer)
- Provided the efficient transformer architecture that serves as RF-DETR's foundation
- Introduced key optimizations for real-time DETR-based detection
- Demonstrated the viability of transformer models for edge deployment

**[DINOv2](https://arxiv.org/pdf/2304.07193)** (Self-Distillation with No Labels v2)
- Self-supervised vision transformer providing robust feature extraction
- Pre-trained on diverse image datasets for strong visual representations
- Enables excellent transfer learning to downstream tasks

**[Deformable DETR](https://arxiv.org/pdf/2010.04159)**
- Introduced deformable attention mechanism reducing computational complexity
- Multi-scale feature processing for detecting objects of various sizes
- Set-based prediction approach eliminating need for hand-crafted anchors

### Related Research

**[DETR](https://arxiv.org/abs/2005.12872)** (End-to-End Object Detection with Transformers)
- Original transformer-based detection approach
- Introduced set-based prediction and Hungarian matching for training

**[D-FINE](https://github.com/Peterande/D-FINE)**
- Fine-grained feature enhancement for real-time detection
- Inspiration for optimization techniques

### Community Contributions

We also acknowledge the broader open-source ecosystem:
- **PyTorch Team**: For the excellent deep learning framework
- **Hugging Face**: For transformers library and model hosting
- **Microsoft**: For the COCO dataset and pycocotools
- **Roboflow Community**: For feedback, testing, and real-world deployment insights

### Special Thanks

Special thanks to the research teams at Carnegie Mellon University and the Roboflow engineering team for their collaboration and contributions to making RF-DETR production-ready.

---

**Standing on the shoulders of giants**: RF-DETR's success is built on the foundation of excellent research and engineering from the computer vision community. We're committed to continuing this tradition of open collaboration and knowledge sharing.

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@software{rf-detr,
  author = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  license = {Apache-2.0},
  title = {RF-DETR},
  howpublished = {\url{https://github.com/roboflow/rf-detr}},
  year = {2025},
  note = {SOTA Real-Time Object Detection Model}
}
```

## Contribute

We welcome and appreciate all contributions! If you notice any issues or bugs, have questions, or would like to suggest new features, please [open an issue](https://github.com/roboflow/rf-detr/issues/new) or pull request. By sharing your ideas and improvements, you help make RF-DETR better for everyone.

<div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://discuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>
</div>
