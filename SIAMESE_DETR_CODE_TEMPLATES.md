# 💻 Code Templates cho Siamese DETR Implementation

File này chứa các code templates cụ thể để implement từng component.

---

## 📁 File 1: `rfdetr/datasets/aeroeyes.py`

```python
"""
AeroEyes Dataset for One-Shot Object Detection
Handles 3 reference images + video frames
"""
import os
import json
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Dict, Optional
from rfdetr.util.misc import NestedTensor


class AeroEyesDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        ref_transform: Optional[transforms.Compose] = None,
        max_frames_per_video: int = 200,
        include_negatives: bool = True,
    ):
        """
        Args:
            root_dir: Path to dataset/observing_train
            transform: Transform for video frames
            ref_transform: Transform for reference images (can be different)
            max_frames_per_video: Max frames to extract per video
            include_negatives: Whether to include negative samples
        """
        self.root_dir = root_dir
        self.samples_dir = os.path.join(root_dir, "samples")
        self.ann_path = os.path.join(root_dir, "annotations", "annotations.json")
        self.transform = transform or self._default_transform()
        self.ref_transform = ref_transform or self.transform
        self.max_frames = max_frames_per_video
        self.include_negatives = include_negatives

        # Load annotations
        self.annotations = self.load_annotations(self.ann_path)
        
        # Build sample list
        self.sample_list = self.build_samples()

    def _default_transform(self):
        """Default transform: resize + normalize"""
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_annotations(self, json_path: str) -> Dict:
        """Load annotations.json into dict structure"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Map: video_id -> {frame_id -> [bboxes]}
        ann_dict = {}
        for item in data:
            vid = item["video_id"]
            boxes = {}
            for ann_block in item.get("annotations", []):
                frame_id = ann_block["frame_id"]
                bboxes = ann_block.get("bboxes", [])
                boxes[frame_id] = bboxes
            ann_dict[vid] = boxes
        
        return ann_dict

    def build_samples(self) -> List[Dict]:
        """
        Build list of samples:
        [
            {
                'video_id': str,
                'ref_imgs': List[str],  # paths to 3 ref images
                'video_path': str,
                'frame_id': int,
                'bbox': List[float],  # [x1, y1, x2, y2]
                'conf': float  # 1.0 for positive, 0.0 for negative
            },
            ...
        ]
        """
        samples = []
        
        for video_name in sorted(os.listdir(self.samples_dir)):
            video_dir = os.path.join(self.samples_dir, video_name)
            video_path = os.path.join(video_dir, "drone_video.mp4")
            ref_dir = os.path.join(video_dir, "object_images")

            if not os.path.exists(video_path) or not os.path.exists(ref_dir):
                continue

            # Get reference images
            ref_imgs = sorted([
                os.path.join(ref_dir, f)
                for f in os.listdir(ref_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            
            if len(ref_imgs) == 0:
                continue

            # Get annotated frames
            ann_frames = list(self.annotations.get(video_name, {}).keys())
            ann_frames = sorted([int(f) for f in ann_frames])

            # Add positive samples
            for frame_id in ann_frames:
                bboxes = self.annotations[video_name][frame_id]
                if len(bboxes) == 0:
                    continue
                
                # Take first bbox (assuming single object)
                bbox = bboxes[0]
                bbox_coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                
                samples.append({
                    "video_id": video_name,
                    "ref_imgs": ref_imgs,
                    "video_path": video_path,
                    "frame_id": frame_id,
                    "bbox": bbox_coords,
                    "conf": 1.0
                })

            # Add negative samples (optional)
            if self.include_negatives and len(ann_frames) > 5:
                # Sample some frames that don't have annotations
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Find frames without annotations
                negative_frames = [
                    f for f in range(total_frames) 
                    if f not in ann_frames
                ]
                
                # Sample a few negative frames
                num_negatives = min(3, len(negative_frames))
                if num_negatives > 0:
                    sampled_negatives = random.sample(negative_frames, num_negatives)
                    for neg_frame_id in sampled_negatives:
                        samples.append({
                            "video_id": video_name,
                            "ref_imgs": ref_imgs,
                            "video_path": video_path,
                            "frame_id": neg_frame_id,
                            "bbox": [0.0, 0.0, 0.0, 0.0],  # Dummy bbox
                            "conf": 0.0
                        })

        return samples

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ref_imgs: List of 3 reference image tensors [C, H, W]
            frame: Video frame tensor [C, H, W]
            bbox: Bounding box tensor [4] - [x1, y1, x2, y2]
            conf: Confidence tensor [1] - 1.0 or 0.0
        """
        sample = self.sample_list[idx]

        # Load 3 reference images
        ref_imgs = []
        for ref_path in sample["ref_imgs"]:
            ref_img = Image.open(ref_path).convert("RGB")
            ref_img = self.ref_transform(ref_img)
            ref_imgs.append(ref_img)

        # Load video frame
        frame = self.load_frame(sample["video_path"], sample["frame_id"])
        frame = self.transform(frame)

        # Convert bbox to tensor (normalize to [0, 1] if needed)
        bbox = torch.tensor(sample["bbox"], dtype=torch.float32)
        conf = torch.tensor([sample["conf"]], dtype=torch.float32)

        return ref_imgs, frame, bbox, conf

    def load_frame(self, video_path: str, frame_id: int) -> Image.Image:
        """Extract frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_id} from {video_path}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)


def aeroeyes_collate_fn(batch: List[Tuple]) -> Tuple[List[List[torch.Tensor]], NestedTensor, List[Dict]]:
    """
    Collate function for AeroEyes dataset
    
    Args:
        batch: List of (ref_imgs, frame, bbox, conf) tuples
    
    Returns:
        ref_imgs_batch: List[List[Tensor]] - [B, 3, C, H, W]
        frames_batch: NestedTensor
        targets_batch: List[Dict] with 'boxes' and 'labels'
    """
    ref_imgs_list = []
    frames_list = []
    targets_list = []
    
    for ref_imgs, frame, bbox, conf in batch:
        ref_imgs_list.append(ref_imgs)  # List of 3 tensors
        frames_list.append(frame)
        
        # Convert bbox to center format [x_center, y_center, w, h] if needed
        # For now, keep as [x1, y1, x2, y2]
        bbox_cxcywh = [
            (bbox[0] + bbox[2]) / 2,  # x_center (normalize later)
            (bbox[1] + bbox[3]) / 2,  # y_center
            bbox[2] - bbox[0],         # width
            bbox[3] - bbox[1],        # height
        ]
        
        targets_list.append({
            'boxes': torch.tensor([bbox_cxcywh], dtype=torch.float32),  # [1, 4]
            'labels': torch.tensor([1.0 if conf.item() > 0 else 0.0], dtype=torch.float32),
        })
    
    # Stack frames
    frames_tensor = torch.stack(frames_list)  # [B, C, H, W]
    
    # Create mask (all False for now, can be improved)
    frames_mask = torch.zeros(frames_tensor.shape[0], frames_tensor.shape[2], frames_tensor.shape[3], dtype=torch.bool)
    
    return ref_imgs_list, NestedTensor(frames_tensor, frames_mask), targets_list
```

---

## 📁 File 2: `rfdetr/models/siamese_detr.py`

```python
"""
Siamese DETR Model for One-Shot Object Detection
Wraps LWDETR with Siamese architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from rfdetr.models.lwdetr import LWDETR


class SiameseDETR(nn.Module):
    """
    Siamese DETR model that processes reference images and target images
    """
    def __init__(self, lwdetr_model: LWDETR):
        super().__init__()
        
        # Share backbone and transformer
        self.backbone = lwdetr_model.backbone
        self.transformer = lwdetr_model.transformer
        self.bbox_embed = lwdetr_model.bbox_embed
        
        # Reference encoder components
        hidden_dim = self.transformer.d_model
        
        # Pooling for reference features
        self.ref_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection layer for reference vector
        self.ref_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Store original model for compatibility
        self.lwdetr_model = lwdetr_model
        self.num_queries = lwdetr_model.num_queries
        
    def forward(
        self,
        ref_imgs: List[List[torch.Tensor]],
        target_img: torch.Tensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Siamese DETR
        
        Args:
            ref_imgs: List[List[Tensor]] - [B, 3, C, H, W] - 3 ref images per sample
            target_img: Tensor or NestedTensor - [B, C, H, W] - target frames
            targets: Optional list of target dicts (for training)
        
        Returns:
            Dictionary with:
                - v_ref: Reference vector [B, D]
                - object_embeddings: Decoder outputs [B, Q, D]
                - pred_boxes: Predicted boxes [B, Q, 4]
                - pred_logits: Dummy logits [B, Q, 1] (for compatibility)
        """
        batch_size = len(ref_imgs)
        device = target_img.tensors.device if hasattr(target_img, 'tensors') else target_img.device
        
        # ========== Reference Branch ==========
        ref_features_list = []
        
        for b in range(batch_size):
            batch_ref_features = []
            
            # Process each of the 3 reference images
            for ref_img in ref_imgs[b]:  # ref_img: [C, H, W]
                ref_img_batch = ref_img.unsqueeze(0)  # [1, C, H, W]
                
                # Backbone forward
                ref_feats, _, _ = self.backbone(ref_img_batch)
                
                # Get last stage feature map
                ref_feat = ref_feats[-1]  # NestedTensor
                ref_src, _ = ref_feat.decompose()  # [1, C, H, W]
                
                # Global Average Pooling
                ref_vec = self.ref_pool(ref_src).flatten(1)  # [1, C]
                batch_ref_features.append(ref_vec)
            
            # Aggregate 3 reference images (mean pooling)
            batch_ref_vec = torch.stack(batch_ref_features, dim=0).mean(dim=0)  # [1, C]
            ref_features_list.append(batch_ref_vec)
        
        # Stack and project
        v_ref = torch.cat(ref_features_list, dim=0)  # [B, C]
        v_ref = self.ref_proj(v_ref)  # [B, D]
        
        # ========== Target Branch ==========
        # Standard DETR forward
        if hasattr(target_img, 'tensors'):
            # NestedTensor
            feats, pos, mask = self.backbone(target_img)
        else:
            # Regular tensor - convert to NestedTensor
            from rfdetr.util.misc import NestedTensor
            mask = torch.zeros(target_img.shape[0], target_img.shape[2], target_img.shape[3], dtype=torch.bool, device=device)
            nested_target = NestedTensor(target_img, mask)
            feats, pos, mask = self.backbone(nested_target)
        
        # Decompose features
        srcs = []
        masks = []
        for feat in feats:
            src, m = feat.decompose()
            srcs.append(src)
            masks.append(m)
        
        # Get query embeddings
        if self.training:
            refpoint_embed_weight = self.lwdetr_model.refpoint_embed.weight
            query_feat_weight = self.lwdetr_model.query_feat.weight
        else:
            refpoint_embed_weight = self.lwdetr_model.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.lwdetr_model.query_feat.weight[:self.num_queries]
        
        # Transformer forward
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, pos, refpoint_embed_weight, query_feat_weight
        )
        
        if hs is None:
            raise ValueError("Transformer returned None")
        
        # Get object embeddings from last decoder layer
        object_embeddings = hs[-1]  # [B, Q, D]
        
        # Predict boxes
        if self.lwdetr_model.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(object_embeddings)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            pred_boxes = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            pred_boxes = (self.bbox_embed(object_embeddings) + ref_unsigmoid[-1]).sigmoid()
        
        # Dummy logits for compatibility (not used in loss)
        pred_logits = torch.zeros(batch_size, self.num_queries, 1, device=device)
        
        # Build output dict
        outputs = {
            'v_ref': v_ref,  # [B, D]
            'object_embeddings': object_embeddings,  # [B, Q, D]
            'pred_boxes': pred_boxes,  # [B, Q, 4]
            'pred_logits': pred_logits,  # [B, Q, 1] - dummy
        }
        
        return outputs
```

---

## 📁 File 3: Modifications to `rfdetr/models/matcher.py`

Add this method to `HungarianMatcher` class:

```python
def forward_siamese(self, outputs: Dict, targets: List[Dict], group_detr: int = 1):
    """
    Forward pass for Siamese matching mode
    Uses cosine similarity instead of classification cost
    """
    from rfdetr.util.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
    
    bs, num_queries = outputs["pred_boxes"].shape[:2]
    
    # Extract reference vector and object embeddings
    v_ref = outputs["v_ref"]  # [B, D]
    obj_embeds = outputs["object_embeddings"]  # [B, Q, D]
    pred_bboxes = outputs["pred_boxes"]  # [B, Q, 4]
    
    # Normalize for cosine similarity
    v_ref_norm = F.normalize(v_ref, p=2, dim=1)  # [B, D]
    obj_embeds_norm = F.normalize(obj_embeds, p=2, dim=2)  # [B, Q, D]
    
    # Compute similarity: [B, Q]
    sim_matrix = torch.einsum('bd,bqd->bq', v_ref_norm, obj_embeds_norm)
    
    # Cost: 1 - similarity (lower similarity = higher cost)
    cost_match = 1 - sim_matrix  # [B, Q]
    
    # Get GT boxes
    tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [T, 4]
    tgt_labels = torch.cat([v["labels"] for v in targets])  # [T]
    
    # Flatten predictions
    out_bbox = pred_bboxes.flatten(0, 1)  # [B*Q, 4]
    
    # Compute bbox costs
    cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*Q, T]
    
    # Compute GIoU cost
    giou = generalized_box_iou(
        box_cxcywh_to_xyxy(out_bbox),
        box_cxcywh_to_xyxy(tgt_bbox)
    )
    cost_giou = -giou  # [B*Q, T]
    
    # Expand cost_match to match GT boxes
    # For each GT, use the same matching cost
    cost_match_expanded = cost_match.unsqueeze(-1).expand(-1, -1, len(tgt_bbox))  # [B, Q, T]
    cost_match_expanded = cost_match_expanded.flatten(0, 1)  # [B*Q, T]
    
    # Combined cost
    C = (
        self.cost_bbox * cost_bbox +
        self.cost_giou * cost_giou +
        self.cost_match * cost_match_expanded
    )
    
    C = C.view(bs, num_queries, -1).float().cpu()
    
    # Handle NaN/Inf
    max_cost = C.max() if C.numel() > 0 else 0
    C[C.isinf() | C.isnan()] = max_cost * 2
    
    # Hungarian matching
    sizes = [len(v["boxes"]) for v in targets]
    indices = []
    g_num_queries = num_queries // group_detr
    C_list = C.split(g_num_queries, dim=1)
    
    for g_i in range(group_detr):
        C_g = C_list[g_i]
        indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
        if g_i == 0:
            indices = indices_g
        else:
            indices = [
                (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), 
                 np.concatenate([indice1[1], indice2[1]]))
                for indice1, indice2 in zip(indices, indices_g)
            ]
    
    return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
```

Modify `__init__`:
```python
def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, 
             focal_alpha: float = 0.25, use_pos_only: bool = False,
             use_position_modulated_cost: bool = False, mask_point_sample_ratio: int = 16, 
             cost_mask_ce: float = 1, cost_mask_dice: float = 1,
             use_siamese: bool = False, cost_match: float = 1.0):
    # ... existing code ...
    self.use_siamese = use_siamese
    self.cost_match = cost_match
```

Modify `forward`:
```python
def forward(self, outputs, targets, group_detr=1):
    if self.use_siamese:
        return self.forward_siamese(outputs, targets, group_detr)
    else:
        return self.forward_standard(outputs, targets, group_detr)  # existing code
```

---

## 📁 File 4: Modifications to `rfdetr/models/lwdetr.py` (SetCriterion)

Add to `SetCriterion.__init__`:
```python
def __init__(self, ..., use_siamese: bool = False, match_margin: float = 0.5, match_loss_coef: float = 1.0):
    # ... existing code ...
    self.use_siamese = use_siamese
    self.match_margin = match_margin
    self.match_loss_coef = match_loss_coef
```

Add method:
```python
def loss_match(self, outputs, targets, indices, num_boxes):
    """Contrastive loss for matching reference and object embeddings"""
    v_ref = F.normalize(outputs["v_ref"], p=2, dim=1)  # [B, D]
    obj_embeds = F.normalize(outputs["object_embeddings"], p=2, dim=2)  # [B, Q, D]
    
    loss_total = 0
    for i, (idx_pred, idx_gt) in enumerate(indices):
        if len(idx_pred) == 0:
            continue
        
        # Positive pairs (matched queries)
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

Modify `forward`:
```python
def forward(self, outputs, targets):
    if self.use_siamese:
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["boxes"]) for t in targets)
        
        losses = {}
        losses['loss_bbox'] = self.loss_boxes(outputs, targets, indices, num_boxes)
        losses['loss_giou'] = self.loss_giou(outputs, targets, indices, num_boxes)
        losses['loss_match'] = self.loss_match(outputs, targets, indices, num_boxes)
        losses['class_error'] = torch.tensor(0.0, device=outputs["pred_boxes"].device)
        
        return losses
    else:
        # ... existing code ...
```

---

## 📁 File 5: Modifications to `rfdetr/config.py`

Add to `ModelConfig`:
```python
class ModelConfig(BaseModel):
    # ... existing fields ...
    use_siamese: bool = False
    match_loss_coef: float = 1.0
    match_margin: float = 0.5
    cost_match: float = 1.0
```

---

## 📁 File 6: Modifications to `rfdetr/main.py`

In `Model.__init__`:
```python
def __init__(self, **kwargs):
    args = populate_args(**kwargs)
    self.args = args
    # ... existing code ...
    
    # Wrap with SiameseDETR if needed
    if args.use_siamese:
        from rfdetr.models.siamese_detr import SiameseDETR
        self.model = SiameseDETR(self.model)
```

---

## 📁 File 7: Test Script `test_siamese_detr.py`

```python
"""Test script for Siamese DETR"""
import torch
from rfdetr.datasets.aeroeyes import AeroEyesDataset, aeroeyes_collate_fn
from torch.utils.data import DataLoader
from rfdetr.main import Model

# Test dataset
dataset = AeroEyesDataset(
    root_dir="dataset/observing_train",
    max_frames_per_video=10
)

print(f"Dataset size: {len(dataset)}")

# Test dataloader
loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=aeroeyes_collate_fn,
    shuffle=False
)

# Get one batch
ref_imgs, frames, targets = next(iter(loader))
print(f"Ref imgs: {len(ref_imgs)} batches, each with {len(ref_imgs[0])} images")
print(f"Frames shape: {frames.tensors.shape}")
print(f"Targets: {len(targets)}")

# Test model
model = Model(
    use_siamese=True,
    num_classes=1,  # Dummy
    resolution=640,
)

model.model.eval()
with torch.no_grad():
    outputs = model.model(ref_imgs, frames)
    print(f"v_ref shape: {outputs['v_ref'].shape}")
    print(f"object_embeddings shape: {outputs['object_embeddings'].shape}")
    print(f"pred_boxes shape: {outputs['pred_boxes'].shape}")
    print("✅ Forward pass successful!")
```

---

## 🚀 Quick Start

1. **Create dataset file:** Copy File 1 to `rfdetr/datasets/aeroeyes.py`
2. **Create model file:** Copy File 2 to `rfdetr/models/siamese_detr.py`
3. **Modify matcher:** Add File 3 code to `rfdetr/models/matcher.py`
4. **Modify criterion:** Add File 4 code to `rfdetr/models/lwdetr.py`
5. **Update config:** Add File 5 fields to `rfdetr/config.py`
6. **Update Model:** Add File 6 code to `rfdetr/main.py`
7. **Test:** Run File 7 test script

---

## ⚠️ Notes

- These are templates - adjust based on your actual codebase structure
- Test each component individually before integrating
- Handle edge cases (empty batches, missing files, etc.)
- Add proper error handling and logging

