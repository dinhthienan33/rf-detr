# 🗺️ Roadmap Implementation: Siamese DETR cho One-Shot Object Detection

## 📋 Tổng quan

Roadmap này mô tả các bước cụ thể để implement Siamese DETR trên RF-DETR codebase, tích hợp với dataset AeroEyes (3 reference images + video frames).

**Mục tiêu:** Biến RF-DETR từ Classifier thành Matcher cho bài toán One-Shot Object Detection.

---

## 🏗️ Kiến trúc Tổng thể

```
Reference Images (3) ─┐
                      │  ┌────────────────────────┐
                      └─▶│ Reference Encoder      │──┐
                         │ (Backbone + Pooling)   │  │
                         └────────────────────────┘  │
                                                    ▼
                                              Reference Query [B, D]
                                                    │
Video Frame ─────────────────────────────────┐     │
                                              ▼     │
                                      ┌────────────────────┐
                                      │ Frame Encoder      │
                                      │ (Backbone)         │
                                      └────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Transformer Decoder    │
                                  │ (Cross-Attention)      │
                                  └────────────────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Object Embeddings      │
                                  │ [B, Q, D]              │
                                  └────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
        ┌────────────────────────┐                        ┌────────────────────────┐
        │ BBox Head             │                        │ Matching Score         │
        │ (Regression)          │                        │ (Cosine Similarity)    │
        └────────────────────────┘                        └────────────────────────┘
```

---

## 📦 Dataset Structure (AeroEyes)

```
dataset/
└── observing_train/
    ├── annotations/
    │   └── annotations.json
    └── samples/
        ├── Backpack_0/
        │   ├── object_images/
        │   │   ├── img_1.jpg  ← Reference 1
        │   │   ├── img_2.jpg  ← Reference 2
        │   │   └── img_3.jpg  ← Reference 3
        │   └── drone_video.mp4
        └── ...
```

**Format output từ Dataset:**
```python
(
    [ref_img_1, ref_img_2, ref_img_3],  # List[Tensor] - 3 reference images
    frame_tensor,                        # Tensor [C, H, W] - video frame
    bbox_tensor,                        # Tensor [4] - [x1, y1, x2, y2]
    conf_tensor                          # Tensor [1] - 1.0 if positive, 0.0 if negative
)
```

---

## 🎯 Implementation Steps

### **STEP 1: Tạo Dataset cho AeroEyes** ✅ Priority: HIGH

**File:** `rfdetr/datasets/aeroeyes.py`

**Nhiệm vụ:**
- Load 3 reference images từ `object_images/`
- Load video frame từ `drone_video.mp4` tại frame_id
- Parse annotations.json để lấy bbox
- Apply transforms độc lập cho ref và frame

**Code structure:**
```python
class AeroEyesDataset(Dataset):
    def __init__(self, root_dir, transform=None, ref_transform=None):
        # Load annotations
        # Build sample list: [(video_id, frame_id, bbox, ref_paths), ...]
        
    def __getitem__(self, idx):
        # Load 3 ref images
        # Load video frame
        # Apply transforms
        # Return: (ref_imgs, frame, bbox, conf)
```

**Checklist:**
- [ ] Implement `load_annotations()` để parse JSON
- [ ] Implement `build_samples()` để tạo sample list
- [ ] Implement `load_frame()` để extract frame từ video
- [ ] Handle negative samples (conf=0)
- [ ] Test với 1 sample

**Estimated time:** 4-6 hours

---

### **STEP 2: Tạo SiameseDETR Model** ✅ Priority: HIGH

**File:** `rfdetr/models/siamese_detr.py`

**Nhiệm vụ:**
- Wrap LWDETR với Siamese architecture
- Implement reference encoder (backbone + pooling)
- Implement target encoder (backbone)
- Combine features qua decoder

**Code structure:**
```python
class SiameseDETR(nn.Module):
    def __init__(self, lwdetr_model: LWDETR):
        super().__init__()
        # Share backbone
        self.backbone = lwdetr_model.backbone
        self.transformer = lwdetr_model.transformer
        self.bbox_embed = lwdetr_model.bbox_embed
        
        # Reference encoder
        self.ref_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ref_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
    def forward(self, ref_imgs, target_img):
        # Reference branch: ref_imgs → backbone → pool → v_ref
        # Target branch: target_img → backbone → encoder → decoder
        # Return: {v_ref, object_embeddings, pred_boxes}
```

**Chi tiết implementation:**

1. **Reference Encoder:**
   ```python
   # Process 3 ref images
   ref_features_list = []
   for ref_img in ref_imgs:  # ref_imgs: [B, 3, C, H, W] → iterate over 3
       feats, _, _ = self.backbone(ref_img)
       ref_feat = feats[-1]  # Last stage [B, C, H, W]
       ref_vec = self.ref_pool(ref_feat).flatten(1)  # [B, C]
       ref_features_list.append(ref_vec)
   
   # Aggregate 3 ref images (mean or attention)
   v_ref = torch.stack(ref_features_list, dim=1).mean(dim=1)  # [B, C]
   v_ref = self.ref_proj(v_ref)  # [B, D]
   ```

2. **Target Encoder:**
   ```python
   # Standard DETR forward
   feats, pos, mask = self.backbone(target_img)
   srcs, masks = [], []
   for feat in feats:
       src, m = feat.decompose()
       srcs.append(src)
       masks.append(m)
   
   # Decoder forward
   hs, ref_unsigmoid, _, _ = self.transformer(
       srcs, masks, pos, 
       refpoint_embed_weight, 
       query_feat_weight
   )
   
   object_embeddings = hs[-1]  # [B, Q, D]
   pred_boxes = self.bbox_embed(hs[-1])  # [B, Q, 4]
   ```

**Checklist:**
- [ ] Create `siamese_detr.py`
- [ ] Implement `__init__` với weight sharing
- [ ] Implement reference encoder (handle 3 ref images)
- [ ] Implement target encoder
- [ ] Test forward pass với dummy data
- [ ] Verify output shapes

**Estimated time:** 6-8 hours

---

### **STEP 3: Tạo Custom Collate Function** ✅ Priority: MEDIUM

**File:** `rfdetr/datasets/aeroeyes.py` (thêm vào)

**Nhiệm vụ:**
- Batch multiple samples với format đặc biệt
- Handle variable number of ref images (3)

**Code:**
```python
def aeroeyes_collate_fn(batch):
    """
    batch: List[(ref_imgs, frame, bbox, conf), ...]
    Returns:
        ref_imgs_batch: List[List[Tensor]] - [B, 3, C, H, W]
        frames_batch: NestedTensor
        targets_batch: List[Dict]
    """
    ref_imgs_list = []
    frames_list = []
    targets_list = []
    
    for ref_imgs, frame, bbox, conf in batch:
        ref_imgs_list.append(ref_imgs)  # List of 3 tensors
        frames_list.append(frame)
        targets_list.append({
            'boxes': bbox.unsqueeze(0),  # [1, 4]
            'labels': torch.tensor([1.0 if conf > 0 else 0.0]),  # For matching
        })
    
    # Stack frames
    frames_tensor = torch.stack(frames_list)
    frames_mask = None  # TODO: create mask if needed
    
    return ref_imgs_list, NestedTensor(frames_tensor, frames_mask), targets_list
```

**Checklist:**
- [ ] Implement collate function
- [ ] Handle batching ref images
- [ ] Create NestedTensor for frames
- [ ] Test với DataLoader

**Estimated time:** 2-3 hours

---

### **STEP 4: Modify HungarianMatcher** ✅ Priority: HIGH

**File:** `rfdetr/models/matcher.py`

**Nhiệm vụ:**
- Thay `cost_class` bằng `cost_match` (cosine similarity)
- Tính similarity giữa `v_ref` và `object_embeddings`

**Modifications:**

1. **Thêm parameter:**
   ```python
   def __init__(self, ..., cost_match: float = 1.0, use_siamese: bool = False):
       self.cost_match = cost_match
       self.use_siamese = use_siamese
   ```

2. **Modify forward:**
   ```python
   def forward(self, outputs, targets, group_detr=1):
       if self.use_siamese:
           return self.forward_siamese(outputs, targets, group_detr)
       else:
           return self.forward_standard(outputs, targets, group_detr)
   
   def forward_siamese(self, outputs, targets, group_detr=1):
       # Extract v_ref and object_embeddings
       v_ref = outputs["v_ref"]  # [B, D]
       obj_embeds = outputs["object_embeddings"]  # [B, Q, D]
       pred_bboxes = outputs["pred_boxes"]  # [B, Q, 4]
       
       # Normalize for cosine similarity
       v_ref_norm = F.normalize(v_ref, p=2, dim=1)  # [B, D]
       obj_embeds_norm = F.normalize(obj_embeds, p=2, dim=2)  # [B, Q, D]
       
       # Compute similarity matrix: [B, Q]
       # For each query, compute similarity with v_ref
       sim_matrix = torch.einsum('bd,bqd->bq', v_ref_norm, obj_embeds_norm)  # [B, Q]
       
       # Cost: 1 - similarity (lower similarity = higher cost)
       cost_match = 1 - sim_matrix  # [B, Q]
       
       # Get GT boxes
       tgt_bbox = torch.cat([v["boxes"] for v in targets])
       tgt_labels = torch.cat([v["labels"] for v in targets])
       
       # Only match positive samples (label > 0)
       # For negative samples, use bbox cost only
       # TODO: Implement proper matching logic
       
       # Bbox costs
       cost_bbox = torch.cdist(pred_bboxes.flatten(0, 1), tgt_bbox, p=1)
       giou = generalized_box_iou(...)
       cost_giou = -giou
       
       # Combined cost
       C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
       if self.cost_match > 0:
           # Expand cost_match to match GT boxes
           # For now, simple approach: use similarity for all GT
           C = C + self.cost_match * cost_match.unsqueeze(-1)  # [B*Q, T]
       
       # Hungarian matching
       # ... (rest of matching logic)
   ```

**Checklist:**
- [ ] Add `use_siamese` flag
- [ ] Implement `forward_siamese()`
- [ ] Compute cosine similarity
- [ ] Integrate với bbox cost
- [ ] Test matching logic

**Estimated time:** 4-6 hours

---

### **STEP 5: Modify SetCriterion (Loss Function)** ✅ Priority: HIGH

**File:** `rfdetr/models/lwdetr.py` (class SetCriterion)

**Nhiệm vụ:**
- Bỏ `loss_labels` (classification loss)
- Thêm `loss_match` (contrastive loss)
- Giữ `loss_bbox` và `loss_giou`

**Modifications:**

1. **Add parameters:**
   ```python
   def __init__(self, ..., use_siamese: bool = False, match_margin: float = 0.5):
       self.use_siamese = use_siamese
       self.match_margin = match_margin
   ```

2. **Add loss_match method:**
   ```python
   def loss_match(self, outputs, targets, indices, num_boxes):
       """
       Contrastive loss for matching
       """
       v_ref = F.normalize(outputs["v_ref"], p=2, dim=1)  # [B, D]
       obj_embeds = F.normalize(outputs["object_embeddings"], p=2, dim=2)  # [B, Q, D]
       
       loss_total = 0
       for i, (idx_pred, idx_gt) in enumerate(indices):
           if len(idx_pred) == 0:
               continue
           
           # Positive pairs (matched)
           positive_embeds = obj_embeds[i, idx_pred]  # [M, D]
           positive_sim = torch.einsum('d,md->m', v_ref[i], positive_embeds)  # [M]
           loss_positive = (1 - positive_sim).pow(2).mean()
           
           # Negative pairs (unmatched queries)
           mask_negative = torch.ones(obj_embeds.shape[1], dtype=torch.bool, device=obj_embeds.device)
           mask_negative[idx_pred] = False
           if mask_negative.sum() > 0:
               negative_embeds = obj_embeds[i, mask_negative]  # [N, D]
               negative_sim = torch.einsum('d,nd->n', v_ref[i], negative_embeds)  # [N]
               # Push negatives away (below margin)
               loss_negative = F.relu(negative_sim - self.match_margin).pow(2).mean()
           else:
               loss_negative = torch.tensor(0.0, device=v_ref.device)
           
           loss_total += loss_positive + loss_negative
       
       return loss_total / max(len(indices), 1)
   ```

3. **Modify forward:**
   ```python
   def forward(self, outputs, targets):
       if self.use_siamese:
           # Remove classification loss
           indices = self.matcher(outputs, targets)
           num_boxes = sum(len(t["boxes"]) for t in targets)
           
           losses = {}
           losses['loss_bbox'] = self.loss_boxes(outputs, targets, indices, num_boxes)
           losses['loss_giou'] = self.loss_giou(outputs, targets, indices, num_boxes)
           losses['loss_match'] = self.loss_match(outputs, targets, indices, num_boxes)
           
           # Dummy class_error for logging
           losses['class_error'] = torch.tensor(0.0, device=outputs["pred_boxes"].device)
           
           return losses
       else:
           # Standard DETR loss
           # ... (existing code)
   ```

**Checklist:**
- [ ] Add `use_siamese` flag to SetCriterion
- [ ] Implement `loss_match()`
- [ ] Modify `forward()` to skip classification loss
- [ ] Test loss computation
- [ ] Verify gradients flow correctly

**Estimated time:** 4-6 hours

---

### **STEP 6: Integrate vào Model Class** ✅ Priority: HIGH

**File:** `rfdetr/main.py` (class Model)

**Nhiệm vụ:**
- Thêm flag `use_siamese` vào ModelConfig
- Wrap LWDETR với SiameseDETR khi flag enabled
- Handle forward pass với ref_imgs

**Modifications:**

1. **In `rfdetr/config.py`:**
   ```python
   class ModelConfig(BaseModel):
       # ... existing fields ...
       use_siamese: bool = False
       match_loss_coef: float = 1.0
       match_margin: float = 0.5
   ```

2. **In `rfdetr/main.py` (Model.__init__):**
   ```python
   def __init__(self, **kwargs):
       args = populate_args(**kwargs)
       self.args = args
       
       # Build base model
       self.model = build_model(args)
       
       # Wrap with SiameseDETR if needed
       if args.use_siamese:
           from rfdetr.models.siamese_detr import SiameseDETR
           self.model = SiameseDETR(self.model)
   ```

3. **Modify forward trong Model:**
   ```python
   def forward(self, samples, targets=None, ref_imgs=None):
       if self.args.use_siamese:
           if ref_imgs is None:
               raise ValueError("ref_imgs required for Siamese mode")
           return self.model(ref_imgs, samples.tensors)
       else:
           return self.model(samples, targets)
   ```

**Checklist:**
- [ ] Add config flags
- [ ] Wrap model in SiameseDETR
- [ ] Modify forward signature
- [ ] Test model initialization

**Estimated time:** 2-3 hours

---

### **STEP 7: Modify Training Loop** ✅ Priority: HIGH

**File:** `rfdetr/engine.py` (train_one_epoch)

**Nhiệm vụ:**
- Handle ref_imgs trong data_loader
- Pass ref_imgs vào model forward
- Update loss weights

**Modifications:**

1. **In `train_one_epoch()`:**
   ```python
   for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
       # Handle Siamese mode
       if args.use_siamese:
           # Unpack ref_imgs if present
           if isinstance(samples, tuple):
               ref_imgs, samples, targets = samples
           else:
               # Assume ref_imgs in targets or separate
               ref_imgs = [t.get('ref_imgs') for t in targets]
           
           # Move to device
           ref_imgs = [[r.to(device) for r in refs] for refs in ref_imgs]
           
           # Forward pass
           outputs = model(samples, targets, ref_imgs=ref_imgs)
       else:
           outputs = model(samples, targets)
   ```

2. **Update weight_dict in SetCriterion:**
   ```python
   # In build_criterion_and_postprocessors()
   if args.use_siamese:
       weight_dict = {
           'loss_bbox': args.bbox_loss_coef,
           'loss_giou': args.giou_loss_coef,
           'loss_match': args.match_loss_coef,
       }
   ```

**Checklist:**
- [ ] Modify data loading to handle ref_imgs
- [ ] Update forward call
- [ ] Update loss weights
- [ ] Test training loop

**Estimated time:** 3-4 hours

---

### **STEP 8: Modify Inference/Predict** ✅ Priority: MEDIUM

**File:** `rfdetr/detr.py` (RFDETR.predict)

**Nhiệm vụ:**
- Add `ref_images` parameter
- Compute similarity scores
- Filter by threshold

**Modifications:**

```python
def predict(self, images, ref_images=None, similarity_threshold=0.7, **kwargs):
    if self.model_config.use_siamese:
        if ref_images is None:
            raise ValueError("ref_images required for Siamese mode")
        
        # Process ref_images
        ref_tensors = self._preprocess_ref_images(ref_images)
        
        # Process target images
        outputs = []
        for img in images:
            img_tensor = self._preprocess_image(img)
            
            # Forward pass
            model_outputs = self.model.model(ref_tensors, img_tensor)
            
            # Compute similarity scores
            v_ref = F.normalize(model_outputs["v_ref"], p=2, dim=1)
            obj_embeds = F.normalize(model_outputs["object_embeddings"], p=2, dim=2)
            sim_scores = torch.einsum('bd,bqd->bq', v_ref, obj_embeds)  # [B, Q]
            
            # Filter by threshold
            valid_mask = sim_scores > similarity_threshold
            pred_boxes = model_outputs["pred_boxes"][valid_mask]
            scores = sim_scores[valid_mask]
            
            outputs.append({
                'boxes': pred_boxes,
                'scores': scores,
            })
        
        return outputs
    else:
        # Standard DETR prediction
        # ... (existing code)
```

**Checklist:**
- [ ] Add ref_images parameter
- [ ] Implement preprocessing for ref images
- [ ] Compute similarity scores
- [ ] Filter by threshold
- [ ] Test inference

**Estimated time:** 3-4 hours

---

### **STEP 9: Testing & Validation** ✅ Priority: HIGH

**Files:** Create `tests/test_siamese_detr.py`

**Test cases:**

1. **Dataset Tests:**
   - [ ] Load sample from AeroEyes dataset
   - [ ] Verify ref_imgs shape (3 images)
   - [ ] Verify frame shape
   - [ ] Verify bbox format

2. **Model Tests:**
   - [ ] Forward pass với dummy data
   - [ ] Verify output shapes
   - [ ] Verify gradients flow
   - [ ] Test với batch_size > 1

3. **Matcher Tests:**
   - [ ] Test matching với positive samples
   - [ ] Test matching với negative samples
   - [ ] Verify cost matrix computation

4. **Loss Tests:**
   - [ ] Test loss_match computation
   - [ ] Verify loss decreases during training
   - [ ] Test loss weights

5. **Integration Tests:**
   - [ ] End-to-end training với 1 epoch
   - [ ] Inference với sample images
   - [ ] Verify predictions format

**Estimated time:** 6-8 hours

---

### **STEP 10: Hyperparameter Tuning** ✅ Priority: MEDIUM

**Parameters to tune:**

1. **Loss weights:**
   - `match_loss_coef`: 0.5, 1.0, 2.0
   - `bbox_loss_coef`: 5.0 (keep standard)
   - `giou_loss_coef`: 2.0 (keep standard)

2. **Matching:**
   - `match_margin`: 0.3, 0.5, 0.7
   - `cost_match`: 0.5, 1.0, 2.0

3. **Reference aggregation:**
   - Mean pooling (current)
   - Attention pooling (future)
   - Max pooling (alternative)

4. **Similarity threshold:**
   - Inference: 0.6, 0.7, 0.8

**Estimated time:** 8-12 hours (multiple training runs)

---

## 📊 Implementation Timeline

| Step | Priority | Estimated Time | Dependencies |
|------|----------|----------------|--------------|
| Step 1: Dataset | HIGH | 4-6h | None |
| Step 2: SiameseDETR | HIGH | 6-8h | Step 1 |
| Step 3: Collate Function | MEDIUM | 2-3h | Step 1 |
| Step 4: Matcher | HIGH | 4-6h | Step 2 |
| Step 5: Criterion | HIGH | 4-6h | Step 4 |
| Step 6: Model Integration | HIGH | 2-3h | Step 2, 5 |
| Step 7: Training Loop | HIGH | 3-4h | Step 6 |
| Step 8: Inference | MEDIUM | 3-4h | Step 6 |
| Step 9: Testing | HIGH | 6-8h | All steps |
| Step 10: Tuning | MEDIUM | 8-12h | Step 9 |

**Total estimated time:** 42-60 hours (~1-2 weeks full-time)

---

## 🚨 Critical Issues & Solutions

### Issue 1: Handling 3 Reference Images
**Problem:** Dataset có 3 ref images, cần aggregate thành 1 vector.

**Solution:** 
- Option A: Mean pooling (simple, fast)
- Option B: Attention pooling (learnable, better)
- Option C: Use all 3 separately (more complex)

**Recommendation:** Start with Option A, upgrade to B later.

### Issue 2: Memory Usage
**Problem:** Siamese forward cần 2x backbone passes → 2x memory.

**Solution:**
- Use gradient checkpointing
- Reduce batch size
- Use mixed precision (AMP)

### Issue 3: Loss Balancing
**Problem:** 3 loss terms cần cân bằng.

**Solution:**
- Start với weights từ DETR: `bbox=5, giou=2, match=1`
- Monitor loss values, adjust ratios
- Use learning rate scheduling

### Issue 4: Negative Samples
**Problem:** Dataset có negative samples (conf=0), cần handle trong matcher.

**Solution:**
- Skip matching cho negative samples
- Use contrastive loss để push away
- Consider hard negative mining

---

## ✅ Success Criteria

1. **Training:**
   - [ ] Model trains without errors
   - [ ] Loss decreases over epochs
   - [ ] No NaN/Inf values

2. **Matching:**
   - [ ] Positive samples match correctly
   - [ ] Negative samples have low similarity
   - [ ] Bbox predictions are reasonable

3. **Inference:**
   - [ ] Can predict on new images với ref images
   - [ ] Similarity scores are meaningful
   - [ ] Predictions align with ground truth

---

## 📝 Notes

- **Start simple:** Implement basic version first, optimize later
- **Test frequently:** Test after each step
- **Keep compatibility:** Maintain backward compatibility với standard DETR
- **Documentation:** Document all changes và decisions
- **Version control:** Commit after each working step

---

## 🔗 References

- RF-DETR codebase structure
- DETR paper: https://arxiv.org/abs/2005.12872
- Siamese Networks: https://arxiv.org/abs/1506.02142
- Contrastive Loss: https://arxiv.org/abs/2004.11362

