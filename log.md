# Implementation Log - Siamese DETR

## Step 1: AeroEyes Dataset Implementation ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Files Created

#### 1. `rfdetr/datasets/aeroeyes.py`
- **Purpose:** Dataset class for AeroEyes One-Shot Object Detection task
- **Features:**
  - Loads 3 reference images from `object_images/` directory
  - Extracts video frames from `drone_video.mp4`
  - Parses annotations from `annotations.json`
  - Supports positive and negative samples
  - Handles bbox normalization to [0, 1] range

**Key Components:**

1. **`AeroEyesDataset` class:**
   - `__init__()`: Initializes dataset, loads annotations, builds sample list
   - `load_annotations()`: Parses JSON annotations into dict structure
   - `build_samples()`: Creates list of samples with ref images, video paths, frame IDs, bboxes
   - `__getitem__()`: Returns (ref_imgs, frame, target) tuple
   - `load_frame()`: Extracts frame from video using OpenCV

2. **`aeroeyes_collate_fn()` function:**
   - Batches multiple samples
   - Handles variable number of reference images (3 per sample)
   - Creates NestedTensor for frames
   - Returns: `(ref_imgs_batch, frames_nested, targets_batch)`

**Data Format:**

- **Input:**
  - `ref_imgs`: List of 3 PIL Images → transformed to List[3 x Tensor [C, H, W]]
  - `frame`: PIL Image → transformed to Tensor [C, H, W]
  - `target`: Dict with:
    - `boxes`: Tensor [N, 4] - normalized [x1, y1, x2, y2] in [0, 1]
    - `labels`: Tensor [N] - 1.0 for positive, 0.0 for negative
    - `image_id`: Tensor [1]
    - `size`: Tensor [2] - (H, W) after transforms
    - `orig_size`: Tensor [2] - original (H, W)

- **Output from collate_fn:**
  - `ref_imgs_batch`: List[List[Tensor]] - [B, 3, C, H, W] structure
  - `frames_batch`: NestedTensor with tensors [B, C, H, W] and mask [B, H, W]
  - `targets_batch`: List[Dict] - one dict per sample

**Features Implemented:**

✅ Load 3 reference images per sample  
✅ Extract frames from video files  
✅ Parse JSON annotations  
✅ Handle positive samples (with bbox)  
✅ Handle negative samples (no object)  
✅ Bbox normalization to [0, 1]  
✅ Transform support for ref images and frames  
✅ Collate function for batching  
✅ Error handling for missing files  
✅ Fallback for missing ref images (repeat last)  

**Configuration Options:**

- `max_frames_per_video`: Limit frames per video (default: 200)
- `include_negatives`: Include negative samples (default: True)
- `negative_ratio`: Ratio of negatives to positives (default: 0.3)
- `transforms`: Transforms for video frames
- `ref_transforms`: Transforms for reference images (optional, defaults to transforms)

**Testing Notes:**

- Dataset assumes structure: `dataset/observing_train/samples/{video_id}/object_images/` and `drone_video.mp4`
- Annotations format: JSON with `video_id`, `annotations` array containing `frame_id` and `bboxes`
- Bbox format supported: `{"x1", "y1", "x2", "y2"}` or COCO format `{"bbox": [x, y, w, h]}`

**Next Steps:**

- [x] Export dataset in `__init__.py`
- [ ] Test dataset loading with actual data
- [ ] Verify transforms work correctly
- [ ] Test collate function with DataLoader
- [ ] Add unit tests
- [ ] Integrate with training pipeline (Step 7)

---

## Changes Summary

### Files Modified:
1. `rfdetr/datasets/__init__.py` - Added exports for AeroEyesDataset and aeroeyes_collate_fn

### Files Created:
1. `rfdetr/datasets/aeroeyes.py` (new, ~400 lines)
2. `test_aeroeyes_dataset.py` (test script for verification)

### Dependencies:
- `torch`, `torchvision`
- `PIL` (Pillow)
- `cv2` (OpenCV)
- `rfdetr.util.misc.NestedTensor`

---

## Implementation Checklist

- [x] Create `AeroEyesDataset` class
- [x] Implement `load_annotations()` method
- [x] Implement `build_samples()` method
- [x] Implement `__getitem__()` method
- [x] Implement `load_frame()` method
- [x] Create `aeroeyes_collate_fn()` function
- [x] Handle 3 reference images
- [x] Handle positive and negative samples
- [x] Bbox normalization
- [x] Error handling
- [ ] Unit tests
- [ ] Integration test with DataLoader

---

## Known Issues / TODOs

1. **Mask creation:** Currently creates all-False mask. Should handle variable image sizes properly.
2. **Bbox format:** Assumes single bbox per frame. May need to handle multiple bboxes.
3. **Video reading:** Uses OpenCV which may have issues with some video formats.
4. **Memory:** Loading entire video frames may be memory-intensive. Consider caching or lazy loading.

---

## Testing Commands

### Quick Test Script
A test script `test_aeroeyes_dataset.py` has been created to verify the implementation:

```bash
python test_aeroeyes_dataset.py
```

### Manual Testing
```python
# Test dataset loading
from rfdetr.datasets.aeroeyes import AeroEyesDataset, aeroeyes_collate_fn
from torch.utils.data import DataLoader
from torchvision import transforms

# Create transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = AeroEyesDataset(
    root_dir="dataset/observing_train",
    transforms=transform,
    max_frames_per_video=10
)

print(f"Dataset size: {len(dataset)}")

# Test single sample
ref_imgs, frame, target = dataset[0]
print(f"Ref imgs: {len(ref_imgs)}")
print(f"Frame shape: {frame.shape}")
print(f"Target keys: {target.keys()}")

# Test DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=aeroeyes_collate_fn,
    shuffle=False
)

ref_imgs_batch, frames_batch, targets_batch = next(iter(loader))
print(f"Batch ref imgs: {len(ref_imgs_batch)}")
print(f"Batch frames shape: {frames_batch.tensors.shape}")
print(f"Batch targets: {len(targets_batch)}")
```

---

## Step 2: SiameseDETR Model Implementation ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Files Created

#### 1. `rfdetr/models/siamese_detr.py`
- **Purpose:** Siamese DETR model wrapper for One-Shot Object Detection
- **Features:**
  - Wraps LWDETR with Siamese architecture
  - Processes 3 reference images → aggregates to reference vector
  - Processes target images → object embeddings
  - Weight sharing between reference and target branches
  - Compatible with existing LWDETR components

**Key Components:**

1. **`SiameseDETR` class:**
   - `__init__()`: Wraps LWDETR model, shares backbone/transformer, adds reference encoder
   - `forward()`: Processes ref_imgs and target_img, returns matching outputs
   - Reference encoder: Backbone → Pooling → Projection → v_ref
   - Target encoder: Standard DETR forward pass

**Architecture:**

```
Reference Branch:
  ref_imgs (3 images) → Backbone → Pooling → Aggregate (mean) → Projection → v_ref [B, D]

Target Branch:
  target_img → Backbone → Transformer → Object Embeddings [B, Q, D] → BBox Head → pred_boxes [B, Q, 4]
```

**Output Format:**

```python
{
    'v_ref': Tensor [B, D],              # Reference vector
    'object_embeddings': Tensor [B, Q, D],  # Object embeddings from decoder
    'pred_boxes': Tensor [B, Q, 4],      # Predicted boxes (cxcywh, normalized)
    'pred_logits': Tensor [B, Q, 1],     # Dummy logits (for compatibility)
    'aux_outputs': List[Dict] (optional) # Auxiliary outputs if aux_loss=True
}
```

**Features Implemented:**

✅ Weight sharing (backbone, transformer)  
✅ Reference encoder with 3-image aggregation  
✅ Target encoder (standard DETR forward)  
✅ Bbox prediction (supports bbox_reparam mode)  
✅ Auxiliary outputs support  
✅ Compatible with existing LWDETR structure  

**Configuration Options:**

- `use_siamese`: Enable Siamese mode (default: False)
- `match_loss_coef`: Weight for matching loss (default: 1.0)
- `match_margin`: Margin for contrastive loss (default: 0.5)
- `cost_match`: Cost weight for matching in matcher (default: 1.0)

**Implementation Details:**

1. **Reference Processing:**
   - Each of 3 ref images processed independently through backbone
   - Features pooled using AdaptiveAvgPool2d
   - 3 vectors aggregated via mean pooling
   - Projected through MLP to get v_ref

2. **Target Processing:**
   - Standard DETR forward pass
   - Uses shared backbone and transformer
   - Extracts object embeddings from last decoder layer
   - Predicts boxes using bbox_embed head

3. **Integration:**
   - Wraps LWDETR model in `build_model()` when `use_siamese=True`
   - Maintains compatibility with existing code
   - No changes needed to training loop (yet - Step 7)

**Next Steps:**

- [x] Create SiameseDETR class
- [x] Implement reference encoder
- [x] Implement target encoder
- [x] Integrate with build_model
- [x] Add config flags
- [ ] Test forward pass with dummy data
- [ ] Verify output shapes
- [ ] Test with actual dataset (Step 1)
- [ ] Modify matcher (Step 4)
- [ ] Modify criterion (Step 5)

---

## Changes Summary

### Files Modified:
1. `rfdetr/models/lwdetr.py` - Modified `build_model()` to wrap with SiameseDETR when `use_siamese=True`
2. `rfdetr/config.py` - Added Siamese flags: `use_siamese`, `match_loss_coef`, `match_margin`, `cost_match`
3. `rfdetr/models/__init__.py` - Added export for SiameseDETR

### Files Created:
1. `rfdetr/models/siamese_detr.py` (new, ~195 lines)

### Dependencies:
- `torch`, `torch.nn`
- `rfdetr.models.lwdetr.LWDETR`
- `rfdetr.util.misc.NestedTensor`, `nested_tensor_from_tensor_list`

---

## Known Issues / TODOs

1. **Reference aggregation:** Currently uses mean pooling. Could be improved with attention-based aggregation.
2. **Memory efficiency:** Processing 3 ref images separately may be memory-intensive. Consider batching.
3. **Testing:** Need to verify forward pass works correctly with actual data shapes.
4. **Integration:** Training loop needs modification to handle ref_imgs input (Step 7).

---

## Testing Commands

### Quick Test Script
```python
# Test SiameseDETR model
import torch
from rfdetr.main import Model
from rfdetr.util.misc import NestedTensor

# Create model with Siamese mode
model = Model(
    use_siamese=True,
    num_classes=1,
    resolution=640,
    encoder="dinov2_windowed_small",
    # ... other required args
)

# Dummy inputs
batch_size = 2
ref_imgs = [[torch.randn(3, 640, 640) for _ in range(3)] for _ in range(batch_size)]
target_tensors = torch.randn(batch_size, 3, 640, 640)
target_mask = torch.zeros(batch_size, 640, 640, dtype=torch.bool)
target_nested = NestedTensor(target_tensors, target_mask)

# Forward pass
model.model.eval()
with torch.no_grad():
    outputs = model.model(ref_imgs, target_nested)
    print(f"v_ref shape: {outputs['v_ref'].shape}")
    print(f"object_embeddings shape: {outputs['object_embeddings'].shape}")
    print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
    print("✅ Forward pass successful!")
```

---

## Step 3: Custom Collate Function Integration ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Overview

Step 3 integrates the custom collate function (`aeroeyes_collate_fn`) into the training pipeline and adds support for building the AeroEyes dataset.

### Files Modified

#### 1. `rfdetr/datasets/aeroeyes.py`
- **Added:** `build_aeroeyes()` function
  - Builds AeroEyes dataset for train/val/test splits
  - Creates appropriate transforms (square resize, normalization)
  - Handles dataset directory resolution
  - Supports configurable parameters (max_frames_per_video, include_negatives, etc.)

**Key Features:**
- Supports `square_resize_div_64` mode (default for AeroEyes)
- Creates transforms compatible with DETR training
- Handles train/val/test splits (currently uses same directory, can be extended)
- Configurable via args object

#### 2. `rfdetr/datasets/__init__.py`
- **Added:** Import for `build_aeroeyes`
- **Modified:** `build_dataset()` function to support `'aeroeyes'` dataset_file

#### 3. `rfdetr/main.py`
- **Modified:** Training loop to use `aeroeyes_collate_fn` when:
  - `use_siamese=True`, OR
  - `dataset_file == 'aeroeyes'`
- **Applied to:** train, val, and test DataLoaders

**Implementation:**
```python
# Select collate function based on mode
collate_fn_train = utils.collate_fn
collate_fn_val = utils.collate_fn
collate_fn_test = utils.collate_fn

if (hasattr(args, 'use_siamese') and args.use_siamese) or args.dataset_file == 'aeroeyes':
    from rfdetr.datasets.aeroeyes import aeroeyes_collate_fn
    collate_fn_train = aeroeyes_collate_fn
    collate_fn_val = aeroeyes_collate_fn
    collate_fn_test = aeroeyes_collate_fn
```

#### 4. `rfdetr/config.py`
- **Modified:** `TrainConfig.dataset_file` to include `"aeroeyes"` as valid option

### Features Implemented

✅ `build_aeroeyes()` function for dataset creation  
✅ Integration with `build_dataset()` pipeline  
✅ Automatic collate function selection  
✅ Support for train/val/test splits  
✅ Transform creation compatible with DETR  
✅ Configurable dataset parameters  

### Collate Function Details

The `aeroeyes_collate_fn` (already implemented in Step 1):
- Handles batching of 3 reference images per sample
- Creates NestedTensor for frames with proper masks
- Returns format: `(ref_imgs_batch, frames_nested, targets_batch)`
- Compatible with SiameseDETR forward signature

**Input Format:**
```python
batch: List[(ref_imgs, frame, target), ...]
# ref_imgs: List[3 x Tensor [C, H, W]]
# frame: Tensor [C, H, W]
# target: Dict with 'boxes', 'labels', etc.
```

**Output Format:**
```python
ref_imgs_batch: List[List[Tensor]]  # [B, 3, C, H, W] structure
frames_batch: NestedTensor          # [B, C, H, W] with mask
targets_batch: List[Dict]            # One dict per sample
```

### Usage

To use AeroEyes dataset with Siamese DETR:

```python
# In config or args
args.dataset_file = 'aeroeyes'
args.use_siamese = True
args.dataset_dir = 'path/to/dataset'  # Should contain 'observing_train' subdirectory

# Dataset will be automatically built and collate function selected
model.train(...)
```

### Next Steps

- [x] Implement `build_aeroeyes()` function
- [x] Integrate with `build_dataset()`
- [x] Modify training code to use custom collate function
- [x] Update config to support 'aeroeyes' dataset_file
- [ ] Test dataset building with actual data
- [ ] Test collate function in training loop
- [ ] Verify DataLoader outputs match model expectations
- [ ] Modify training loop to handle ref_imgs (Step 7)

---

## Changes Summary

### Files Modified:
1. `rfdetr/datasets/aeroeyes.py` - Added `build_aeroeyes()` function
2. `rfdetr/datasets/__init__.py` - Added import and integration
3. `rfdetr/main.py` - Modified DataLoader creation to use custom collate function
4. `rfdetr/config.py` - Added 'aeroeyes' to dataset_file Literal type

### Dependencies:
- `rfdetr.datasets.transforms.SquareResize`
- `torchvision.transforms`
- `pathlib.Path`

---

## Known Issues / TODOs

1. **Val/Test splits:** Currently uses same directory for val/test. Can be extended to support separate splits.
2. **Transform customization:** Reference images use same transforms as frames. Could add separate ref_transforms parameter.
3. **Testing:** Need to verify collate function works correctly in actual training loop.
4. **Memory:** Consider adding support for lazy frame loading if memory becomes an issue.

---

## Testing Commands

### Test Dataset Building
```python
from rfdetr.datasets import build_dataset
from rfdetr.config import TrainConfig

# Create args-like object
class Args:
    dataset_file = 'aeroeyes'
    dataset_dir = 'path/to/dataset'
    square_resize_div_64 = True
    multi_scale = False
    max_frames_per_video = 200
    include_negatives = True
    negative_ratio = 0.3

args = Args()
dataset = build_dataset('train', args, resolution=640)
print(f"Dataset size: {len(dataset)}")

# Test collate function
from rfdetr.datasets.aeroeyes import aeroeyes_collate_fn
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, collate_fn=aeroeyes_collate_fn)
ref_imgs, frames, targets = next(iter(loader))
print(f"Ref imgs batch: {len(ref_imgs)} samples")
print(f"Frames shape: {frames.tensors.shape}")
print(f"Targets: {len(targets)}")
```

---

## Step 4: Modify HungarianMatcher ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Overview

Step 4 modifies the HungarianMatcher to support Siamese mode, replacing classification cost with cosine similarity-based matching cost.

### Files Modified

#### 1. `rfdetr/models/matcher.py`
- **Modified:** `HungarianMatcher.__init__()` 
  - Added `use_siamese` parameter (default: False)
  - Added `cost_match` parameter (default: 1.0)
  - Updated assertion to handle Siamese mode

- **Modified:** `HungarianMatcher.forward()`
  - Routes to `forward_siamese()` when `use_siamese=True`
  - Maintains backward compatibility with standard mode

- **Added:** `HungarianMatcher.forward_siamese()`
  - Implements Siamese matching logic
  - Uses cosine similarity between `v_ref` and `object_embeddings`
  - Combines matching cost with bbox and GIoU costs
  - Supports group_detr for multi-group matching

- **Modified:** `build_matcher()`
  - Passes `use_siamese` and `cost_match` parameters from args
  - Works for both segmentation and standard modes

### Implementation Details

**Siamese Matching Logic:**

1. **Extract embeddings:**
   - `v_ref`: Reference vector [B, D]
   - `object_embeddings`: Object embeddings [B, Q, D]
   - `pred_boxes`: Predicted boxes [B, Q, 4]

2. **Compute cosine similarity:**
   ```python
   v_ref_norm = F.normalize(v_ref, p=2, dim=1)  # [B, D]
   obj_embeds_norm = F.normalize(obj_embeds, p=2, dim=2)  # [B, Q, D]
   sim_matrix = torch.einsum('bd,bqd->bq', v_ref_norm, obj_embeds_norm)  # [B, Q]
   cost_match = 1 - sim_matrix  # Lower similarity = higher cost
   ```

3. **Combine costs:**
   - `cost_bbox`: L1 distance between predicted and GT boxes
   - `cost_giou`: Negative GIoU (higher IoU = lower cost)
   - `cost_match`: 1 - cosine similarity (higher similarity = lower cost)
   - Combined: `C = cost_bbox * cost_bbox + cost_giou * cost_giou + cost_match * cost_match_expanded`

4. **Hungarian matching:**
   - Uses scipy's `linear_sum_assignment` to find optimal matching
   - Supports `group_detr` for multi-group matching
   - Returns indices matching predictions to targets

### Features Implemented

✅ `use_siamese` flag in `__init__`  
✅ `cost_match` parameter for matching cost weight  
✅ `forward_siamese()` method implementation  
✅ Cosine similarity computation  
✅ Cost matrix combination (bbox + GIoU + match)  
✅ Hungarian algorithm integration  
✅ Group DETR support  
✅ Backward compatibility with standard mode  
✅ Integration with `build_matcher()`  

### Key Differences from Standard Mode

| Aspect | Standard Mode | Siamese Mode |
|--------|--------------|--------------|
| Cost Type | Classification (focal loss) | Cosine similarity |
| Input | `pred_logits` | `v_ref`, `object_embeddings` |
| Cost Formula | `cost_class + cost_bbox + cost_giou` | `cost_match + cost_bbox + cost_giou` |
| Matching | Based on class predictions | Based on feature similarity |

### Usage

The matcher automatically uses Siamese mode when:
- `use_siamese=True` is set in args/config
- Model outputs contain `v_ref` and `object_embeddings` keys

```python
# In config or args
args.use_siamese = True
args.cost_match = 1.0  # Weight for matching cost

# Matcher will automatically use Siamese mode
matcher = build_matcher(args)
indices = matcher(outputs, targets)
```

### Next Steps

- [x] Add `use_siamese` flag
- [x] Implement `forward_siamese()`
- [x] Compute cosine similarity
- [x] Integrate with bbox cost
- [x] Update `build_matcher()`
- [ ] Test matching logic with actual data
- [ ] Verify matching quality
- [ ] Modify SetCriterion to use Siamese matcher (Step 5)

---

## Changes Summary

### Files Modified:
1. `rfdetr/models/matcher.py` - Added Siamese mode support

### Dependencies:
- `torch.nn.functional.normalize` (for cosine similarity)
- `torch.einsum` (for efficient similarity computation)
- `rfdetr.util.box_ops.generalized_box_iou`, `box_cxcywh_to_xyxy`

---

## Known Issues / TODOs

1. **Negative samples:** Currently matches all samples. May need special handling for negative samples (label=0).
2. **Cost balancing:** Need to tune `cost_match` weight relative to `cost_bbox` and `cost_giou`.
3. **Testing:** Need to verify matching works correctly with actual SiameseDETR outputs.
4. **Edge cases:** Handle empty targets, single query cases, etc.

---

## Testing Commands

### Test Siamese Matcher
```python
import torch
from rfdetr.models.matcher import HungarianMatcher

# Create matcher with Siamese mode
matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0,
    use_siamese=True,
    cost_match=1.0,
)

# Dummy outputs (from SiameseDETR)
batch_size = 2
num_queries = 300
hidden_dim = 256

outputs = {
    'v_ref': torch.randn(batch_size, hidden_dim),
    'object_embeddings': torch.randn(batch_size, num_queries, hidden_dim),
    'pred_boxes': torch.rand(batch_size, num_queries, 4),  # Normalized [0, 1]
}

# Dummy targets
targets = [
    {
        'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1]]),  # [cx, cy, w, h]
        'labels': torch.tensor([1.0]),
    },
    {
        'boxes': torch.tensor([[0.7, 0.7, 0.15, 0.15]]),
        'labels': torch.tensor([1.0]),
    },
]

# Test matching
matcher.eval()
with torch.no_grad():
    indices = matcher(outputs, targets)
    print(f"Matched {len(indices)} samples")
    for i, (idx_pred, idx_target) in enumerate(indices):
        print(f"Sample {i}: {len(idx_pred)} predictions matched to {len(idx_target)} targets")
        print(f"  Prediction indices: {idx_pred[:5]}...")  # Show first 5
        print(f"  Target indices: {idx_target[:5]}...")
```

---

## Step 5: Modify SetCriterion (Loss Function) ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Overview

Step 5 modifies the SetCriterion class to support Siamese mode, replacing classification loss with contrastive matching loss while keeping bbox and GIoU losses.

### Files Modified

#### 1. `rfdetr/models/lwdetr.py`
- **Modified:** `SetCriterion.__init__()`
  - Added `use_siamese` parameter (default: False)
  - Added `match_margin` parameter (default: 0.5)
  - Added `match_loss_coef` parameter (default: 1.0)

- **Added:** `SetCriterion.loss_match()`
  - Implements contrastive loss for matching
  - Positive pairs: Pull matched queries towards reference vector
  - Negative pairs: Push unmatched queries away (below margin)
  - Uses cosine similarity with L2 normalization

- **Modified:** `SetCriterion.forward()`
  - Routes to `forward_siamese()` when `use_siamese=True`
  - Maintains backward compatibility with standard mode

- **Added:** `SetCriterion.forward_siamese()`
  - Computes losses for Siamese mode
  - Uses `loss_bbox`, `loss_giou`, and `loss_match`
  - Skips classification loss (`loss_labels`)
  - Handles auxiliary outputs and encoder outputs

- **Modified:** `SetCriterion.get_loss()`
  - Added 'match' to loss_map

- **Modified:** `build_criterion_and_postprocessors()`
  - Builds weight_dict based on mode (Siamese vs standard)
  - Sets losses list based on mode
  - Passes Siamese parameters to SetCriterion

### Implementation Details

**Contrastive Loss Logic:**

1. **Normalize embeddings:**
   ```python
   v_ref_norm = F.normalize(v_ref, p=2, dim=1)  # [B, D]
   obj_embeds_norm = F.normalize(obj_embeds, p=2, dim=2)  # [B, Q, D]
   ```

2. **Positive pairs (matched queries):**
   - Extract matched embeddings: `obj_embeds[i, idx_pred]`
   - Compute similarity: `sim = cosine(v_ref[i], positive_embeds)`
   - Loss: `loss_positive = (1 - sim)^2` (pull towards similarity=1)

3. **Negative pairs (unmatched queries):**
   - Extract unmatched embeddings: `obj_embeds[i, ~idx_pred]`
   - Compute similarity: `sim = cosine(v_ref[i], negative_embeds)`
   - Loss: `loss_negative = max(0, sim - margin)^2` (push below margin)

4. **Total loss:**
   ```python
   loss_match = (loss_positive + loss_negative) / batch_size
   ```

**Loss Components in Siamese Mode:**

| Loss | Purpose | Formula |
|------|---------|---------|
| `loss_bbox` | L1 regression loss | `L1(pred_boxes, gt_boxes)` |
| `loss_giou` | Generalized IoU loss | `1 - GIoU(pred_boxes, gt_boxes)` |
| `loss_match` | Contrastive matching loss | `(1 - sim_pos)^2 + max(0, sim_neg - margin)^2` |

**Weight Dictionary:**

- Standard mode: `{'loss_ce': cls_coef, 'loss_bbox': bbox_coef, 'loss_giou': giou_coef}`
- Siamese mode: `{'loss_bbox': bbox_coef, 'loss_giou': giou_coef, 'loss_match': match_coef}`

### Features Implemented

✅ `use_siamese` flag in `__init__`  
✅ `match_margin` parameter for contrastive loss  
✅ `match_loss_coef` parameter for loss weight  
✅ `loss_match()` method implementation  
✅ Contrastive loss (positive + negative pairs)  
✅ `forward_siamese()` method  
✅ Integration with auxiliary outputs  
✅ Integration with encoder outputs (two_stage)  
✅ Updated `build_criterion_and_postprocessors()`  
✅ Dynamic weight_dict and losses list  

### Key Differences from Standard Mode

| Aspect | Standard Mode | Siamese Mode |
|--------|--------------|--------------|
| Losses | `loss_ce`, `loss_bbox`, `loss_giou`, `loss_cardinality` | `loss_bbox`, `loss_giou`, `loss_match` |
| Classification | Focal loss on class predictions | Contrastive loss on embeddings |
| Matching | Based on class + bbox | Based on feature similarity + bbox |
| Weight Dict | Includes `loss_ce` | Includes `loss_match` instead |

### Usage

The criterion automatically uses Siamese mode when:
- `use_siamese=True` is set in args/config
- Model outputs contain `v_ref` and `object_embeddings` keys

```python
# In config or args
args.use_siamese = True
args.match_loss_coef = 1.0  # Weight for matching loss
args.match_margin = 0.5     # Margin for contrastive loss

# Criterion will automatically use Siamese mode
criterion, postprocess = build_criterion_and_postprocessors(args)
losses = criterion(outputs, targets)
# Returns: {'loss_bbox': ..., 'loss_giou': ..., 'loss_match': ..., 'class_error': 0.0}
```

### Next Steps

- [x] Add `use_siamese` flag to SetCriterion
- [x] Implement `loss_match()`
- [x] Modify `forward()` to skip classification loss
- [x] Update `build_criterion_and_postprocessors()`
- [ ] Test loss computation with actual data
- [ ] Verify loss values are reasonable
- [ ] Modify training loop to handle ref_imgs (Step 7)

---

## Changes Summary

### Files Modified:
1. `rfdetr/models/lwdetr.py` - Added Siamese loss support to SetCriterion

### Dependencies:
- `torch.nn.functional.normalize` (for cosine similarity)
- `torch.einsum` (for efficient similarity computation)

---

## Known Issues / TODOs

1. **Loss balancing:** Need to tune `match_loss_coef` relative to `bbox_loss_coef` and `giou_loss_coef`.
2. **Margin tuning:** `match_margin` (default 0.5) may need adjustment based on training dynamics.
3. **Negative sampling:** Currently uses all unmatched queries. Could consider hard negative mining.
4. **Testing:** Need to verify loss computation works correctly with actual SiameseDETR outputs.
5. **Auxiliary losses:** Need to verify auxiliary outputs work correctly in Siamese mode.

---

## Testing Commands

### Test Siamese Criterion
```python
import torch
from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.matcher import HungarianMatcher

# Create matcher and criterion with Siamese mode
matcher = HungarianMatcher(
    cost_class=1.0,
    cost_bbox=5.0,
    cost_giou=2.0,
    use_siamese=True,
    cost_match=1.0,
)

weight_dict = {'loss_bbox': 5.0, 'loss_giou': 2.0, 'loss_match': 1.0}
criterion = SetCriterion(
    num_classes=1,
    matcher=matcher,
    weight_dict=weight_dict,
    focal_alpha=0.25,
    losses=['boxes', 'match'],
    use_siamese=True,
    match_margin=0.5,
    match_loss_coef=1.0,
)

# Dummy outputs (from SiameseDETR)
batch_size = 2
num_queries = 300
hidden_dim = 256

outputs = {
    'v_ref': torch.randn(batch_size, hidden_dim),
    'object_embeddings': torch.randn(batch_size, num_queries, hidden_dim),
    'pred_boxes': torch.rand(batch_size, num_queries, 4),  # Normalized [0, 1]
}

# Dummy targets
targets = [
    {
        'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1]]),  # [cx, cy, w, h]
        'labels': torch.tensor([1.0]),
    },
    {
        'boxes': torch.tensor([[0.7, 0.7, 0.15, 0.15]]),
        'labels': torch.tensor([1.0]),
    },
]

# Test loss computation
criterion.eval()
with torch.no_grad():
    losses = criterion(outputs, targets)
    print(f"Losses: {list(losses.keys())}")
    print(f"loss_bbox: {losses['loss_bbox'].item():.4f}")
    print(f"loss_giou: {losses['loss_giou'].item():.4f}")
    print(f"loss_match: {losses['loss_match'].item():.4f}")
    print(f"class_error: {losses['class_error'].item():.4f}")
    print("✅ Loss computation successful!")
```

---

## Step 6: Integrate vào Model Class ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Overview

Step 6 integrates SiameseDETR into the Model class by adding a forward method that routes between Siamese and standard modes based on configuration flags.

### Files Modified

#### 1. `rfdetr/main.py`
- **Added:** `Model.forward()` method
  - Handles Siamese mode routing
  - Accepts `ref_imgs` parameter for Siamese mode
  - Routes to appropriate model forward based on `use_siamese` flag
  - Validates ref_imgs presence in Siamese mode
  - Converts samples to NestedTensor if needed

### Implementation Details

**Forward Method Signature:**
```python
def forward(self, samples, targets=None, ref_imgs=None):
    """
    Forward pass for Model class
    
    Args:
        samples: NestedTensor or Tensor - target images/frames
        targets: Optional[List[Dict]] - ground truth targets (for training)
        ref_imgs: Optional[List[List[Tensor]]] - reference images for Siamese mode
    
    Returns:
        Dictionary with model outputs
    """
```

**Routing Logic:**

1. **Siamese Mode Detection:**
   ```python
   if self.args.use_siamese or self.args.dataset_file == 'aeroeyes':
   ```

2. **Validation:**
   - Checks if `ref_imgs` is provided (required for Siamese mode)
   - Raises ValueError if missing

3. **Input Conversion:**
   - Converts `samples` to NestedTensor if needed
   - Uses `nested_tensor_from_tensor_list` utility

4. **Model Forward:**
   - Siamese mode: `self.model(ref_imgs, samples, targets)`
   - Standard mode: `self.model(samples, targets)`

### Integration Points

**Already Completed (from Step 2):**
- ✅ `build_model()` wraps LWDETR with SiameseDETR when `use_siamese=True`
- ✅ Config flags (`use_siamese`, `match_loss_coef`, `match_margin`) already in ModelConfig
- ✅ Model initialization already handles Siamese wrapping

**New in Step 6:**
- ✅ Forward method routing
- ✅ ref_imgs parameter handling
- ✅ Input validation for Siamese mode

### Features Implemented

✅ Forward method added to Model class  
✅ Siamese mode detection (`use_siamese` or `dataset_file == 'aeroeyes'`)  
✅ ref_imgs parameter handling  
✅ Input validation (raises error if ref_imgs missing)  
✅ NestedTensor conversion for samples  
✅ Routing to correct model forward  
✅ Backward compatibility with standard mode  

### Usage

```python
# Initialize model with Siamese mode
model = Model(use_siamese=True, ...)

# Forward pass with ref_imgs
ref_imgs = [[ref1, ref2, ref3], ...]  # List[List[Tensor]]
samples = NestedTensor(...)  # Target frames
targets = [{...}, ...]  # Ground truth

outputs = model.forward(samples, targets, ref_imgs=ref_imgs)
# Returns: {'v_ref': ..., 'object_embeddings': ..., 'pred_boxes': ..., ...}

# Standard mode (backward compatible)
outputs = model.forward(samples, targets)
# Returns: {'pred_logits': ..., 'pred_boxes': ..., ...}
```

### Key Points

1. **Model Wrapping:** Already handled in `build_model()` (Step 2)
2. **Forward Routing:** New forward method handles both modes
3. **Validation:** Ensures ref_imgs provided in Siamese mode
4. **Compatibility:** Standard mode still works without changes

### Next Steps

- [x] Add forward method to Model class
- [x] Handle ref_imgs parameter
- [x] Verify SiameseDETR wrapping (already done in Step 2)
- [ ] Modify training loop to use Model.forward() with ref_imgs (Step 7)
- [ ] Test model initialization and forward pass

---

## Changes Summary

### Files Modified:
1. `rfdetr/main.py` - Added `Model.forward()` method

### Dependencies:
- `rfdetr.util.misc.NestedTensor` (already imported)
- `rfdetr.util.misc.nested_tensor_from_tensor_list` (imported in method)

---

## Known Issues / TODOs

1. **Training Loop Integration:** Step 7 needed to modify training loop to pass ref_imgs to Model.forward()
2. **Evaluation Loop:** Step 7 needed to modify evaluation loop similarly
3. **Testing:** Need to test forward pass with actual data

---

## Testing Commands

### Test Model Forward
```python
import torch
from rfdetr.main import Model
from rfdetr.util.misc import NestedTensor

# Initialize with Siamese mode
model = Model(
    use_siamese=True,
    encoder="dinov2_windowed_small",
    resolution=560,
    num_classes=1,
    # ... other config ...
)

# Create dummy inputs
batch_size = 2
ref_imgs = [
    [torch.randn(3, 224, 224) for _ in range(3)]  # 3 ref images
    for _ in range(batch_size)
]
samples = NestedTensor(
    torch.randn(batch_size, 3, 560, 560),
    torch.zeros(batch_size, 560, 560, dtype=torch.bool)
)
targets = [
    {'boxes': torch.tensor([[0.5, 0.5, 0.1, 0.1]]), 'labels': torch.tensor([1.0])}
    for _ in range(batch_size)
]

# Test forward pass
model.eval()
with torch.no_grad():
    outputs = model.forward(samples, targets, ref_imgs=ref_imgs)
    print(f"Output keys: {list(outputs.keys())}")
    print(f"v_ref shape: {outputs['v_ref'].shape}")
    print(f"object_embeddings shape: {outputs['object_embeddings'].shape}")
    print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
    print("✅ Model forward successful!")
```

---

## Step 7: Modify Training Loop ✅

**Date:** 2025-01-XX  
**Status:** ✅ Completed

### Overview

Step 7 modifies the training and evaluation loops to handle ref_imgs from the data loader and pass them to the model forward pass in Siamese mode.

### Files Modified

#### 1. `rfdetr/engine.py`
- **Modified:** `train_one_epoch()` function
  - Added Siamese mode detection
  - Unpack 3 items from data_loader when Siamese mode (ref_imgs_batch, samples, targets)
  - Extract ref_imgs for sub-batches in gradient accumulation
  - Pass ref_imgs to model forward in Siamese mode
  - Maintain backward compatibility with standard mode

- **Modified:** `evaluate()` function
  - Added Siamese mode detection
  - Unpack 3 items from data_loader when Siamese mode
  - Pass ref_imgs to model forward in Siamese mode
  - Maintain backward compatibility with standard mode

### Implementation Details

**Siamese Mode Detection:**
```python
use_siamese = getattr(args, 'use_siamese', False) or getattr(args, 'dataset_file', None) == 'aeroeyes'
```

**Training Loop Changes:**

1. **Data Unpacking:**
   ```python
   if use_siamese:
       ref_imgs_batch, samples, targets = batch_data
   else:
       samples, targets = batch_data
   ```

2. **Sub-batch Handling:**
   ```python
   if use_siamese:
       new_ref_imgs = ref_imgs_batch[start_idx:final_idx]
       new_ref_imgs = [[r.to(device) for r in refs] for refs in new_ref_imgs]
   ```

3. **Model Forward:**
   ```python
   if use_siamese:
       outputs = model(new_samples, new_targets, ref_imgs=new_ref_imgs)
   else:
       outputs = model(new_samples, new_targets)
   ```

**Evaluation Loop Changes:**

Similar changes as training loop:
- Unpack ref_imgs_batch from data_loader
- Move ref_imgs to device
- Pass ref_imgs to model forward

### Features Implemented

✅ Siamese mode detection in training loop  
✅ Data unpacking for 3-item batches  
✅ ref_imgs handling in gradient accumulation  
✅ Device movement for ref_imgs  
✅ Model forward with ref_imgs parameter  
✅ Evaluation loop modifications  
✅ Backward compatibility with standard mode  

### Key Points

1. **Data Loader Format:**
   - Siamese mode: Returns `(ref_imgs_batch, samples, targets)` - 3 items
   - Standard mode: Returns `(samples, targets)` - 2 items

2. **Gradient Accumulation:**
   - ref_imgs are extracted per sub-batch
   - Each ref image moved to device individually

3. **Model Forward:**
   - Uses Model.forward() which routes to SiameseDETR or LWDETR
   - ref_imgs passed as keyword argument

### Next Steps

- [x] Modify training loop data unpacking
- [x] Pass ref_imgs to model forward
- [x] Modify evaluation loop
- [ ] Test training loop with actual data
- [ ] Verify loss computation works correctly

---

## Changes Summary

### Files Modified:
1. `rfdetr/engine.py` - Modified `train_one_epoch()` and `evaluate()` functions

### Dependencies:
- No new dependencies added

---

## Known Issues / TODOs

1. **Testing:** Need to test training loop with actual AeroEyes dataset
2. **Performance:** Monitor memory usage with ref_imgs
3. **Validation:** Verify loss values are reasonable during training

---

## Testing Commands

### Test Training Loop
```python
# Training should now work with:
python -m rfdetr.main train \
    --use_siamese True \
    --dataset_file aeroeyes \
    --data_path /path/to/aeroeyes \
    --epochs 1 \
    --batch_size 2
```

---

## Inference Script: `inference_oneshot.py` ✅

**Date:** 2025-01-XX  
**Status:** ✅ Created

### Overview

Created a complete inference script for One-Shot Object Detection using Siamese DETR. The script loads a trained model, processes reference and target images, runs inference, and outputs predictions with similarity scores.

### Features

✅ Load model from checkpoint  
✅ Preprocess 3 reference images  
✅ Preprocess target image  
✅ Run inference with Siamese mode  
✅ Compute similarity scores  
✅ Filter by threshold  
✅ Postprocess boxes to original image coordinates  
✅ Save results as JSON  
✅ Optional visualization with bounding boxes  

### Usage

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

### Script Components

1. **Image Loading:**
   - `load_image()`: Load and convert to RGB
   - `preprocess_image()`: Apply SquareResize, ToTensor, Normalize
   - `preprocess_ref_images()`: Process 3 reference images

2. **Inference:**
   - Load model with Siamese mode enabled
   - Prepare ref_imgs and target as NestedTensor
   - Run forward pass
   - Extract outputs

3. **Postprocessing:**
   - Compute cosine similarity between v_ref and object_embeddings
   - Filter by threshold
   - Convert boxes from normalized cxcywh to xyxy
   - Scale to original image size

4. **Output:**
   - Save results as JSON
   - Optional visualization with OpenCV

### Output Format

```json
{
  "target_image": "path/to/target.jpg",
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

### Key Functions

- `postprocess_predictions()`: Convert model outputs to final predictions
- `draw_predictions()`: Visualize results with bounding boxes
- `main()`: Main inference pipeline

---

