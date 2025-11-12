# KẾ HOẠCH PHÁT TRIỂN VÀ TÙY CHỈNH RF-DETR

## TỔNG QUAN DỰ ÁN

RF-DETR là mô hình phát hiện đối tượng và phân đoạn instance real-time dựa trên transformer, được phát triển bởi Roboflow. Dự án này tập trung vào việc tùy chỉnh và phát triển RF-DETR cho các tác vụ cụ thể.

---

## PHÂN TÍCH KIẾN TRÚC CODEBASE

### 1. CẤU TRÚC THƯ MỤC CHÍNH

```
rf-detr/
├── rfdetr/                    # Package chính
│   ├── main.py               # Entry point, training logic
│   ├── detr.py               # High-level API (RFDETR class)
│   ├── config.py             # Cấu hình các model variants
│   ├── engine.py             # Training và evaluation functions
│   ├── models/               # Kiến trúc model
│   │   ├── lwdetr.py        # Model chính LWDETR
│   │   ├── backbone/        # Backbone (DINOv2)
│   │   ├── transformer.py   # Transformer decoder
│   │   ├── matcher.py       # Hungarian matcher
│   │   └── segmentation_head.py  # Segmentation head
│   ├── datasets/             # Dataset loaders
│   │   ├── coco.py          # COCO dataset
│   │   └── transforms.py    # Data augmentation
│   ├── util/                 # Utilities
│   │   ├── misc.py          # Helper functions
│   │   ├── metrics.py       # Metrics tracking
│   │   └── benchmark.py     # Performance benchmarking
│   └── deploy/               # Deployment
│       └── export.py        # ONNX export
├── docs/                     # Documentation
└── tests/                    # Unit tests
```

### 2. CÁC MODEL VARIANTS

- **RF-DETR-Nano**: 30.5M params, 384x384 resolution
- **RF-DETR-Small**: 32.1M params, 512x512 resolution  
- **RF-DETR-Medium**: 33.7M params, 576x576 resolution
- **RF-DETR-Base**: 560x560 resolution (deprecated)
- **RF-DETR-Large**: 384 hidden dim, multi-scale features
- **RF-DETR-Seg-Preview**: Segmentation model

---

## KẾ HOẠCH PHÁT TRIỂN

### PHASE 1: HIỂU VÀ SETUP MÔI TRƯỜNG (Tuần 1-2)

#### 1.1 Cài đặt và Kiểm tra
- [ ] Cài đặt dependencies: `pip install rfdetr`
- [ ] Kiểm tra GPU/CUDA availability
- [ ] Test inference với pretrained model
- [ ] Chạy benchmark để verify performance

#### 1.2 Nghiên cứu Codebase
- [ ] Đọc và hiểu `main.py` - training pipeline
- [ ] Nghiên cứu `detr.py` - high-level API
- [ ] Phân tích `config.py` - các cấu hình model
- [ ] Hiểu `engine.py` - training/evaluation logic
- [ ] Nghiên cứu `models/lwdetr.py` - kiến trúc model

#### 1.3 Dataset Preparation
- [ ] Chuẩn bị dataset theo format COCO
- [ ] Tạo cấu trúc: `train/`, `valid/`, `test/`
- [ ] Mỗi folder có `_annotations.coco.json`
- [ ] Verify dataset loading với `build_dataset()`

### PHASE 2: TRAINING CƠ BẢN (Tuần 3-4)

#### 2.1 Fine-tuning với Custom Dataset
- [ ] Sử dụng pretrained weights (rf-detr-medium.pth)
- [ ] Cấu hình training parameters:
  - Batch size: 4-8 (tùy GPU memory)
  - Learning rate: 1e-4 (encoder: 1.5e-4)
  - Epochs: 50-100
  - Gradient accumulation: 4 steps
- [ ] Setup callbacks cho monitoring
- [ ] Enable TensorBoard/W&B logging

#### 2.2 Training Configuration
```python
# Ví dụ config training
train_config = {
    "dataset_dir": "path/to/dataset",
    "output_dir": "output/experiment1",
    "batch_size": 4,
    "grad_accum_steps": 4,
    "epochs": 100,
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "use_ema": True,
    "early_stopping": True,
    "tensorboard": True,
    "wandb": True
}
```

#### 2.3 Monitoring và Evaluation
- [ ] Track metrics: mAP@50, mAP@50:95
- [ ] Monitor loss curves
- [ ] Evaluate trên validation set mỗi epoch
- [ ] Save best checkpoint (regular và EMA)

### PHASE 3: TÙY CHỈNH VÀ TỐI ƯU (Tuần 5-6)

#### 3.1 Hyperparameter Tuning
- [ ] Learning rate scheduling (cosine/step)
- [ ] Warmup epochs tuning
- [ ] Drop path rate adjustment
- [ ] Loss coefficients tuning:
  - `cls_loss_coef`
  - `bbox_loss_coef`
  - `giou_loss_coef`

#### 3.2 Data Augmentation
- [ ] Multi-scale training (`multi_scale=True`)
- [ ] Expanded scales (`expanded_scales=True`)
- [ ] Random resize via padding
- [ ] Custom augmentation strategies

#### 3.3 Model Architecture Tweaks
- [ ] Thử các model variants khác nhau
- [ ] Điều chỉnh `num_queries` (default: 300)
- [ ] Tune `group_detr` (default: 13)
- [ ] Experiment với `dec_layers` (decoder layers)

### PHASE 4: SEGMENTATION (Nếu cần) (Tuần 7-8)

#### 4.1 Segmentation Setup
- [ ] Sử dụng `RFDETRSegPreviewConfig`
- [ ] Dataset phải có segmentation masks
- [ ] Cấu hình segmentation head:
  - `mask_downsample_ratio: 4`
  - `mask_ce_loss_coef: 5.0`
  - `mask_dice_loss_coef: 5.0`

#### 4.2 Segmentation Training
- [ ] Fine-tune từ detection checkpoint
- [ ] Monitor segmentation metrics
- [ ] Evaluate mask quality

### PHASE 5: DEPLOYMENT VÀ OPTIMIZATION (Tuần 9-10)

#### 5.1 Model Optimization
- [ ] Sử dụng `optimize_for_inference()`:
  ```python
  model.optimize_for_inference(compile=True, batch_size=1)
  ```
- [ ] Benchmark inference speed
- [ ] Compare với baseline

#### 5.2 Export Models
- [ ] Export to ONNX format:
  ```python
  model.export(
      output_dir="onnx_models",
      simplify=True,
      opset_version=17
  )
  ```
- [ ] Test ONNX inference
- [ ] Optimize ONNX model

#### 5.3 Integration
- [ ] Integrate với Roboflow Inference
- [ ] Test trên production data
- [ ] Performance profiling

### PHASE 6: ADVANCED FEATURES (Tuần 11-12)

#### 6.1 Custom Backbone
- [ ] Nghiên cứu backbone architecture
- [ ] Thử nghiệm với different encoders
- [ ] Custom feature extraction

#### 6.2 Loss Function Customization
- [ ] Implement custom loss functions
- [ ] Experiment với `ia_bce_loss`
- [ ] Tune loss weights dynamically

#### 6.3 Advanced Training Techniques
- [ ] Gradient checkpointing
- [ ] Mixed precision training (AMP)
- [ ] Distributed training setup
- [ ] Knowledge distillation (nếu có teacher model)

---

## CÁC TASK CỤ THỂ

### Task 1: Setup Training Pipeline
**Mục tiêu**: Tạo script training hoàn chỉnh

**Steps**:
1. Tạo `train.py` script
2. Load dataset từ COCO format
3. Initialize model với pretrained weights
4. Setup optimizer và scheduler
5. Implement training loop với callbacks
6. Save checkpoints và logs

**Code template**:
```python
from rfdetr import RFDETRMedium
from rfdetr.config import TrainConfig

model = RFDETRMedium()
config = TrainConfig(
    dataset_dir="data/custom_dataset",
    output_dir="output/exp1",
    batch_size=4,
    epochs=100,
    class_names=["class1", "class2", ...]
)

model.train(**config.dict())
```

### Task 2: Custom Dataset Integration
**Mục tiêu**: Tích hợp dataset custom vào pipeline

**Steps**:
1. Convert dataset sang COCO format
2. Verify annotations
3. Test dataset loading
4. Implement custom transforms nếu cần

### Task 3: Hyperparameter Optimization
**Mục tiêu**: Tìm best hyperparameters cho dataset

**Approach**:
- Grid search hoặc random search
- Focus vào: lr, batch_size, lr_drop
- Use validation mAP làm metric

### Task 4: Model Evaluation
**Mục tiêu**: Đánh giá model performance

**Metrics to track**:
- mAP@50, mAP@50:95
- Per-class AP
- Inference latency
- Model size

### Task 5: Deployment Pipeline
**Mục tiêu**: Tạo pipeline deploy model

**Steps**:
1. Export model to ONNX
2. Optimize ONNX model
3. Create inference script
4. Test inference speed
5. Deploy với Roboflow Inference

---

## CÁC FILE QUAN TRỌNG CẦN NGHIÊN CỨU

### 1. `rfdetr/main.py`
- **Chức năng**: Core training logic
- **Key classes**: `Model` class
- **Key methods**: 
  - `train()`: Main training function
  - `export()`: ONNX export
  - `reinitialize_detection_head()`: Adapt to new classes

### 2. `rfdetr/detr.py`
- **Chức năng**: High-level API
- **Key classes**: `RFDETR`, `RFDETRBase`, `RFDETRMedium`, etc.
- **Key methods**:
  - `predict()`: Inference
  - `train()`: Training wrapper
  - `optimize_for_inference()`: Speed optimization

### 3. `rfdetr/config.py`
- **Chức năng**: Model và training configurations
- **Key classes**: 
  - `ModelConfig`: Base config
  - `RFDETRMediumConfig`: Medium variant
  - `TrainConfig`: Training parameters

### 4. `rfdetr/models/lwdetr.py`
- **Chức năng**: Model architecture
- **Key class**: `LWDETR`
- **Components**: Backbone + Transformer + Detection Head

### 5. `rfdetr/engine.py`
- **Chức năng**: Training và evaluation functions
- **Key functions**:
  - `train_one_epoch()`: Training loop
  - `evaluate()`: Evaluation function

---

## BEST PRACTICES

### 1. Training Best Practices
- ✅ Luôn sử dụng pretrained weights
- ✅ Enable EMA (Exponential Moving Average)
- ✅ Use early stopping để tránh overfitting
- ✅ Monitor cả validation và training metrics
- ✅ Save checkpoints thường xuyên
- ✅ Use gradient accumulation cho small batch sizes

### 2. Data Best Practices
- ✅ Đảm bảo dataset balance
- ✅ Use proper data augmentation
- ✅ Validate annotations quality
- ✅ Split train/val/test properly

### 3. Model Selection
- ✅ Start với Medium variant (best accuracy/speed tradeoff)
- ✅ Use Nano/Small cho edge deployment
- ✅ Use Large cho maximum accuracy

### 4. Performance Optimization
- ✅ Use `optimize_for_inference()` cho production
- ✅ Export to ONNX cho cross-platform
- ✅ Benchmark trên target hardware
- ✅ Profile để tìm bottlenecks

---

## TROUBLESHOOTING GUIDE

### Vấn đề thường gặp:

1. **Out of Memory (OOM)**
   - Giảm batch_size
   - Tăng grad_accum_steps
   - Use gradient checkpointing
   - Giảm resolution

2. **Training không converge**
   - Kiểm tra learning rate
   - Verify dataset quality
   - Check pretrained weights loading
   - Adjust loss coefficients

3. **Low mAP**
   - Tăng số epochs
   - Improve dataset quality
   - Tune hyperparameters
   - Try different model variant

4. **Slow inference**
   - Use `optimize_for_inference()`
   - Export to ONNX
   - Reduce resolution
   - Use smaller model variant

---

## METRICS VÀ EVALUATION

### Metrics quan trọng:
- **mAP@50**: Mean Average Precision at IoU=0.5
- **mAP@50:95**: Mean AP averaged over IoU thresholds 0.5-0.95
- **Per-class AP**: AP cho từng class
- **Inference Latency**: Thời gian inference (ms)
- **FPS**: Frames per second

### Evaluation Script:
```python
from rfdetr import RFDETRMedium

model = RFDETRMedium()
model.load_checkpoint("output/checkpoint_best.pth")

# Evaluate trên test set
results = model.evaluate(test_dataset)
print(f"mAP@50: {results['mAP@50']}")
print(f"mAP@50:95: {results['mAP@50:95']}")
```

---

## TIMELINE TỔNG QUAN

| Phase | Thời gian | Mục tiêu |
|-------|-----------|----------|
| Phase 1 | Tuần 1-2 | Setup và hiểu codebase |
| Phase 2 | Tuần 3-4 | Training cơ bản |
| Phase 3 | Tuần 5-6 | Tùy chỉnh và tối ưu |
| Phase 4 | Tuần 7-8 | Segmentation (optional) |
| Phase 5 | Tuần 9-10 | Deployment |
| Phase 6 | Tuần 11-12 | Advanced features |

---

## TÀI LIỆU THAM KHẢO

1. **Official Documentation**: https://rfdetr.roboflow.com
2. **GitHub Repository**: https://github.com/roboflow/rf-detr
3. **Paper**: RF-DETR paper (nếu có)
4. **Related Papers**:
   - LW-DETR: https://arxiv.org/pdf/2406.03459
   - DINOv2: https://arxiv.org/pdf/2304.07193
   - Deformable DETR: https://arxiv.org/pdf/2010.04159

---

## NEXT STEPS

1. ✅ Đọc và hiểu plan này
2. ⬜ Setup môi trường development
3. ⬜ Download và test pretrained model
4. ⬜ Chuẩn bị dataset
5. ⬜ Bắt đầu Phase 1

---

**Lưu ý**: Plan này là flexible và có thể điều chỉnh dựa trên:
- Dataset cụ thể
- Hardware constraints
- Project requirements
- Timeline thực tế

**Updated**: 2025-01-XX

