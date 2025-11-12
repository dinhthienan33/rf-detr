# Implementation Log: Sửa 1 — Mở API để đưa reference vào pipeline

## Overview
This log tracks the implementation of adding reference image support to RF-DETR's predict() and train() APIs.

## Implementation Steps

### Step 1-3: Modify RFDETR.predict() API ✅
**Status**: Completed  
**Files Modified**: `rfdetr/detr.py`

**Changes**:
- [x] Add `ref_images` parameter to `predict()` signature
- [x] Add validation logic for `ref_images`
- [x] Add preprocessing for reference images
- [x] Modify model forward call to pass `ref_img`

**Notes**:
- Maintained backward compatibility (ref_images=None by default)
- Reference images validated to be same length as images list
- Preprocessing matches image preprocessing exactly
- Added warning for optimized inference model (ref_images not supported yet)

---

### Step 6: Update LWDETR.forward() Signature ✅
**Status**: Completed  
**Files Modified**: `rfdetr/models/lwdetr.py`

**Changes**:
- [x] Update forward signature to accept `ref_img=None`
- [x] Handle ref_img parameter (convert to NestedTensor if needed)
- [x] Added conversion logic for Tensor to NestedTensor
- [ ] Pass ref_img through forward pass (actual conditioning logic in Sửa 2 - will be implemented later)

---

### Step 4: Modify Dataset to Return Reference Images
**Status**: Pending  
**Files to Modify**: `rfdetr/datasets/coco.py`, `rfdetr/util/misc.py`

**Changes**:
- [ ] Modify dataset `__getitem__()` to return ref_images
- [ ] Update collate_fn to handle optional ref_images
- [ ] Add config for reference images directory

---

### Step 5: Update Training Loop ✅
**Status**: Completed  
**Files Modified**: `rfdetr/engine.py`, `rfdetr/util/misc.py`

**Changes**:
- [x] Modify data loader iteration to handle ref_samples
- [x] Process reference images in training loop
- [x] Pass ref_images to model forward call
- [x] Updated collate_fn to handle optional ref_images
- [x] Updated evaluation loop to handle ref_images
- [x] Handle multi-scale resizing for ref_images

---

### Step 7-8: Update Model.train() and train_from_config()
**Status**: Pending  
**Files to Modify**: `rfdetr/main.py`, `rfdetr/detr.py`

**Changes**:
- [ ] Update Model.train() to forward ref_images
- [ ] Add ref_images config to train_from_config()
- [ ] Pass ref_images config to dataset builder

---

### Step 9: Handle Optimized Inference Model
**Status**: Pending  
**Files to Modify**: `rfdetr/detr.py`

**Changes**:
- [ ] Update optimize_for_inference() for ref_images
- [ ] Update optimized inference path

---

### Step 10: Testing and Validation
**Status**: Pending

**Tests**:
- [ ] Test backward compatibility (no ref_images)
- [ ] Test new functionality with ref_images
- [ ] Validate preprocessing

---

## Progress Summary
- Started: Implementation in progress
- Current Step: Step 4 (Dataset modification) - Next
- Completed: 3/7 major steps
- Files Modified: 
  - `rfdetr/detr.py` - Added ref_images support to predict()
  - `rfdetr/models/lwdetr.py` - Updated forward() signature
  - `rfdetr/engine.py` - Updated training and evaluation loops
  - `rfdetr/util/misc.py` - Updated collate_fn

## Implementation Notes

### Completed Features:
1. ✅ **Predict API**: Can now accept `ref_images` parameter alongside `images`
2. ✅ **Model Forward**: Model accepts `ref_img` parameter (ready for conditioning logic)
3. ✅ **Training Loop**: Handles optional ref_images in batches
4. ✅ **Evaluation Loop**: Handles optional ref_images in batches
5. ✅ **Backward Compatibility**: All changes maintain backward compatibility

### Remaining Work:
- Step 4: Dataset modification (depends on specific dataset structure)
- Step 7-8: Config and dataset builder updates
- Step 9: Optimized inference support (currently warns)
- Step 10: Testing

### Key Design Decisions:
- Reference images are optional everywhere (backward compatible)
- Reference images follow same preprocessing as input images
- Reference images converted to NestedTensor format for consistency
- Multi-scale training resizes ref_images along with input images

---

# Implementation Log: Sửa 2 — Thêm "reference-conditioned queries" trong mô hình

## Overview
This log tracks the implementation of reference-conditioned queries in the RF-DETR model, allowing queries to be conditioned on reference images using cross-attention.

## Implementation Steps

### Step 1: Add Configuration Options ✅
**Status**: Completed  
**Files Modified**: `rfdetr/config.py`

**Changes**:
- [x] Add `ref_conditioning: Literal["none", "gap_xattn"] = "none"` to ModelConfig
- [x] Add `use_box_only: bool = False` to ModelConfig

**Notes**:
- `ref_conditioning="none"` maintains backward compatibility (default behavior)
- `ref_conditioning="gap_xattn"` enables reference-conditioned queries via GAP + cross-attention
- `use_box_only` flag added for future use (Sửa 3 - box-only loss)

---

### Step 2: Add Reference Adapter Modules ✅
**Status**: Completed  
**Files Modified**: `rfdetr/models/lwdetr.py`

**Changes**:
- [x] Add `ref_conditioning` and `use_box_only` parameters to `LWDETR.__init__()`
- [x] Add reference adapter modules when `ref_conditioning == "gap_xattn"`:
  - `self.ref_pool`: AdaptiveAvgPool2d for global average pooling
  - `self.ref_proj`: Sequential(Linear + LayerNorm) to project ref features to hidden_dim
  - `self.ref_cross_attn`: MultiheadAttention for cross-attention conditioning

**Notes**:
- Modules only created when `ref_conditioning != "none"` (backward compatible)
- Uses shared backbone to encode reference images (shares representation)
- Reference features projected to match query feature dimension (hidden_dim)

---

### Step 3: Implement Reference-Conditioned Query Logic ✅
**Status**: Completed  
**Files Modified**: `rfdetr/models/lwdetr.py`, `rfdetr/models/transformer.py`

**Changes**:
- [x] Implement reference conditioning in `LWDETR.forward()`:
  - Encode reference image using shared backbone
  - Extract last feature map and apply global average pooling
  - Project reference vector to hidden_dim
  - Condition learned queries via cross-attention
  - Apply residual connection: `q_conditioned = q_learned + cross_attn(q_learned, q_ref)`
- [x] Update `Transformer.forward()` to handle batched query_feat:
  - Support both 2D `[Q, D]` (non-batched) and 3D `[B, Q, D]` (batched) query_feat
  - Maintain backward compatibility with existing code

**Notes**:
- Reference conditioning only applied when `ref_img is not None` AND `ref_conditioning == "gap_xattn"`
- Uses same backbone as input images to share learned representations
- Cross-attention allows queries to "attend" to reference semantics
- Residual connection preserves learned query initialization

---

### Step 4: Update Model Builder ✅
**Status**: Completed  
**Files Modified**: `rfdetr/models/lwdetr.py`

**Changes**:
- [x] Update `build_model()` to pass `ref_conditioning` and `use_box_only` to LWDETR
- [x] Use `getattr()` with defaults for backward compatibility

**Notes**:
- Uses `getattr(args, 'ref_conditioning', 'none')` to handle missing config
- Uses `getattr(args, 'use_box_only', False)` for safe defaults

---

## Progress Summary
- Started: Sửa 2 implementation
- Current Step: Completed ✅
- Completed: 4/4 major steps
- Files Modified:
  - `rfdetr/config.py` - Added ref_conditioning and use_box_only config
  - `rfdetr/models/lwdetr.py` - Added adapter modules and conditioning logic
  - `rfdetr/models/transformer.py` - Updated to handle batched query_feat

## Implementation Notes

### Completed Features:
1. ✅ **Configuration**: Added `ref_conditioning` and `use_box_only` options
2. ✅ **Adapter Modules**: Reference pooling, projection, and cross-attention modules
3. ✅ **Conditioning Logic**: Reference-conditioned queries via GAP + cross-attention
4. ✅ **Transformer Support**: Updated transformer to handle batched query features
5. ✅ **Backward Compatibility**: All changes maintain backward compatibility

### Key Design Decisions:
- **Shared Backbone**: Reference images use same backbone as input (shares learned features)
- **GAP + Cross-Attention**: Global Average Pooling extracts reference vector, cross-attention conditions queries
- **Residual Connection**: Preserves learned query initialization while adding reference conditioning
- **Optional Conditioning**: Only activates when `ref_img` provided AND `ref_conditioning != "none"`
- **Batched Support**: Transformer handles both batched and non-batched query features

### Architecture Details:
```
Reference Image → Backbone → Feature Map → GAP → Ref Vector → Projection → [B, 1, D]
                                                                                ↓
Learned Queries [Q, D] → Expand [B, Q, D] → Cross-Attention ←──────────────────┘
                                                                                ↓
Conditioned Queries [B, Q, D] → Transformer Decoder → Predictions
```

### Next Steps (Sửa 3):
- Implement box-only loss (disable classification loss when `use_box_only=True`)
- Update matcher to remove `cost_class` when `use_box_only=True`
- Update criterion to skip classification loss computation

