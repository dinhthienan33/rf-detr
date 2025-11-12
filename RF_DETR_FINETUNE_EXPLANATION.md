# Phân tích Fine-tuning trong zaloaitask01_rfdetr.ipynb

## ❌ Kết luận: KHÔNG có Fine-tuning

### Tình trạng hiện tại:

**Notebook `zaloaitask01_rfdetr.ipynb` KHÔNG có fine-tuning RF-DETR.**

#### Evidence:
1. **Cell 11**: Chỉ load pre-trained model
   ```python
   det = RFDETRBase()  # Pre-trained trên COCO
   det.optimize_for_inference()
   ```

2. **Không có phần training**: 
   - Không có `model.train()` hoặc training loop
   - Không có dataset preparation cho training
   - Không có loss function hoặc optimizer

3. **Cell 12**: Comment rõ ràng
   ```markdown
   **Lưu ý:** RF-DETR không cần training riêng như YOLO, 
   có thể dùng pre-trained model trực tiếp.
   ```

---

## 📊 Vai trò của Training Dataset trong notebook hiện tại

### **Training dataset KHÔNG được sử dụng để train RF-DETR**

Training dataset chỉ được dùng cho:

#### 1. **Evaluation (Cell 18)**
```python
SAMPLES_ROOT = "/kaggle/input/zaloai2025-aeroeyes/observing/train/samples"
OUT_JSON = "/kaggle/working/train_pred.json"

process_all_videos(SAMPLES_ROOT, OUT_JSON, stabilize=False, train=True)
```
- Chạy inference trên training videos
- So sánh với ground truth annotations
- Tính metrics: Accuracy, IoU, etc.

#### 2. **Không có vai trò khác**
- ❌ Không được dùng để train RF-DETR
- ❌ Không được dùng để fine-tune model
- ❌ Chỉ để đánh giá performance của pre-trained model

---

## ⚠️ Vấn đề với cách tiếp cận hiện tại

### **RF-DETR Pre-trained trên COCO:**
- COCO có 80 classes: person, car, backpack, etc.
- RF-DETR detect được các objects này
- **NHƯNG**: Không biết object nào là target từ reference images

### **Ví dụ:**
- Video có nhiều backpacks → RF-DETR detect tất cả
- Chỉ có 1 backpack là target (giống với reference images)
- RF-DETR không thể phân biệt được backpack nào là target
- → Phải dùng template matching để rerank (như hiện tại)

### **Hạn chế:**
- Template matching chỉ là post-processing
- Nếu RF-DETR miss object → mất object
- Không học được cách match với reference images

---

## ✅ Giải pháp: Thêm Fine-tuning RF-DETR

### **Option 1: Fine-tune RF-DETR với Custom Dataset**

RF-DETR có thể fine-tune, nhưng cần dataset theo format COCO:

```python
# Chuẩn bị dataset theo COCO format
dataset/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── ...
│   └── _annotations.coco.json
├── valid/
│   ├── images/
│   └── _annotations.coco.json
└── test/
    ├── images/
    └── _annotations.coco.json
```

**Vấn đề**: 
- Dataset hiện tại không phải COCO format
- Cần convert từ format hiện tại sang COCO
- RF-DETR fine-tuning phức tạp hơn YOLO

### **Option 2: Giữ nguyên Pre-trained + Template Matching**

**Ưu điểm**:
- ✅ Đơn giản, không cần training
- ✅ Pre-trained model đã tốt cho general objects
- ✅ Template matching đã giải quyết được vấn đề reference matching

**Nhược điểm**:
- ❌ Phụ thuộc vào template matching
- ❌ Nếu RF-DETR miss → mất object
- ❌ Không tối ưu cho bài toán cụ thể

---

## 💡 Khuyến nghị

### **Ngắn hạn (Hiện tại):**
1. **Giữ nguyên pre-trained RF-DETR**
2. **Cải thiện template matching**:
   - Multi-template matching (thay vì mean)
   - Better similarity metrics
   - Adaptive thresholding

### **Dài hạn (Nếu cần cải thiện):**
1. **Fine-tune RF-DETR** trên dataset của bài toán
2. **Hoặc**: Dùng RF-DETR architecture từ `train_rf_detr.ipynb` (reference-based từ đầu)

---

## 📝 Code để thêm Fine-tuning (nếu muốn)

### **Bước 1: Convert dataset sang COCO format**

```python
import json
from pathlib import Path

def convert_to_coco_format(annotations_path, images_dir, output_path):
    """
    Convert annotations từ format hiện tại sang COCO format
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # COCO format structure
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    classes = sorted(set(...))  # Extract từ data
    for i, cls in enumerate(classes):
        coco_data["categories"].append({
            "id": i,
            "name": cls,
            "supercategory": "object"
        })
    
    # Convert images và annotations
    image_id = 0
    ann_id = 0
    
    for item in data.get("root", data):
        video_id = item["video_id"]
        video_path = os.path.join(images_dir, video_id, "drone_video.mp4")
        
        # Extract frames và convert
        # ... (chi tiết implementation)
    
    # Save COCO format
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
```

### **Bước 2: Fine-tune RF-DETR**

```python
from rfdetr import RFDETRBase

# Load pre-trained model
model = RFDETRBase()

# Fine-tune trên custom dataset
model.train(
    dataset_dir='path/to/coco/dataset',
    epochs=15,
    batch_size=16,
    lr=1e-4,
    output_dir='./checkpoints'
)

# Save fine-tuned model
model.save('fine_tuned_rfdetr.pt')
```

**Lưu ý**: 
- Cần GPU mạnh (16GB+ VRAM)
- Training time: vài giờ đến vài ngày
- Cần dataset lớn để fine-tune hiệu quả

---

## 🎯 Kết luận

### **Câu trả lời:**

1. **Có fine-tune không?**
   - ❌ **KHÔNG** - Notebook hiện tại chỉ dùng pre-trained model

2. **Training dataset có vai trò gì?**
   - ✅ **Chỉ để evaluation** - Test performance của pre-trained model
   - ❌ **KHÔNG được dùng để train** RF-DETR

3. **Có nên thêm fine-tuning không?**
   - ⚠️ **Tùy vào performance hiện tại**:
     - Nếu pre-trained + template matching đã tốt → Không cần
     - Nếu cần cải thiện → Nên thêm fine-tuning hoặc dùng reference-based architecture

---

**Cập nhật**: 2025-01-XX

