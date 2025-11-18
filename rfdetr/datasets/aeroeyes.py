# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Custom Dataset for AeroEyes One-Shot Object Detection
# Handles 3 reference images + video frames
# ------------------------------------------------------------------------

"""
AeroEyes Dataset for One-Shot Object Detection
Handles 3 reference images + video frames with annotations
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
    """
    Dataset for AeroEyes One-Shot Object Detection task.
    
    Structure:
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
    
    Returns:
        ref_imgs: List of 3 reference image tensors [C, H, W]
        frame: Video frame tensor [C, H, W]
        target: Dict with 'boxes', 'labels', 'image_id', 'size', 'orig_size'
    """
    
    def __init__(
        self,
        root_dir: str,
        transforms: Optional[transforms.Compose] = None,
        ref_transforms: Optional[transforms.Compose] = None,
        max_frames_per_video: int = 200,
        include_negatives: bool = True,
        negative_ratio: float = 0.3,
    ):
        """
        Args:
            root_dir: Path to dataset/observing_train
            transforms: Transform for video frames (should include ToTensor, Normalize)
            ref_transforms: Transform for reference images (can be different, optional)
            max_frames_per_video: Max frames to extract per video (for memory)
            include_negatives: Whether to include negative samples (no object)
            negative_ratio: Ratio of negative samples to positive samples
        """
        self.root_dir = root_dir
        self.samples_dir = os.path.join(root_dir, "samples")
        self.ann_path = os.path.join(root_dir, "annotations", "annotations.json")
        self.transforms = transforms
        self.ref_transforms = ref_transforms or transforms
        self.max_frames = max_frames_per_video
        self.include_negatives = include_negatives
        self.negative_ratio = negative_ratio

        # Load annotations
        if not os.path.exists(self.ann_path):
            raise FileNotFoundError(f"Annotations file not found: {self.ann_path}")
        
        self.annotations = self.load_annotations(self.ann_path)
        
        # Build sample list
        self.sample_list = self.build_samples()
        
        if len(self.sample_list) == 0:
            raise ValueError(f"No samples found in {self.root_dir}")

    def load_annotations(self, json_path: str) -> Dict:
        """
        Load annotations.json into dict structure
        
        Returns:
            Dict mapping video_id -> {frame_id -> [bboxes]}
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Map: video_id -> {frame_id -> [bboxes]}
        ann_dict = {}
        for item in data:
            vid = item.get("video_id", "")
            if not vid:
                continue
                
            boxes = {}
            for ann_block in item.get("annotations", []):
                frame_id = ann_block.get("frame_id")
                if frame_id is None:
                    continue
                    
                bboxes = ann_block.get("bboxes", [])
                if len(bboxes) > 0:
                    boxes[str(frame_id)] = bboxes
            
            if len(boxes) > 0:
                ann_dict[vid] = boxes
        
        return ann_dict

    def build_samples(self) -> List[Dict]:
        """
        Build list of samples from dataset directory
        
        Returns:
            List of sample dicts with:
                - video_id: str
                - ref_imgs: List[str] - paths to 3 ref images
                - video_path: str
                - frame_id: int
                - bbox: List[float] - [x1, y1, x2, y2] or None for negatives
                - conf: float - 1.0 for positive, 0.0 for negative
        """
        samples = []
        
        if not os.path.exists(self.samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
        
        for video_name in sorted(os.listdir(self.samples_dir)):
            video_dir = os.path.join(self.samples_dir, video_name)
            video_path = os.path.join(video_dir, "drone_video.mp4")
            ref_dir = os.path.join(video_dir, "object_images")

            # Skip if required files don't exist
            if not os.path.exists(video_path) or not os.path.exists(ref_dir):
                continue

            # Get reference images (should be 3)
            ref_imgs = sorted([
                os.path.join(ref_dir, f)
                for f in os.listdir(ref_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            
            if len(ref_imgs) == 0:
                continue
            
            # Ensure we have exactly 3 reference images (pad or truncate if needed)
            if len(ref_imgs) < 3:
                # Repeat last image if less than 3
                while len(ref_imgs) < 3:
                    ref_imgs.append(ref_imgs[-1])
            elif len(ref_imgs) > 3:
                # Take first 3 if more than 3
                ref_imgs = ref_imgs[:3]

            # Get annotated frames for this video
            ann_frames_dict = self.annotations.get(video_name, {})
            ann_frames = sorted([int(f) for f in ann_frames_dict.keys()])

            # Add positive samples (with annotations)
            for frame_id in ann_frames[:self.max_frames]:
                bboxes = ann_frames_dict[str(frame_id)]
                if len(bboxes) == 0:
                    continue
                
                # Take first bbox (assuming single object per frame for now)
                bbox = bboxes[0]
                # Handle different bbox formats
                if "x1" in bbox and "y1" in bbox:
                    bbox_coords = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                elif "bbox" in bbox:
                    # COCO format [x, y, w, h] -> convert to [x1, y1, x2, y2]
                    x, y, w, h = bbox["bbox"]
                    bbox_coords = [x, y, x + w, y + h]
                else:
                    continue
                
                samples.append({
                    "video_id": video_name,
                    "ref_imgs": ref_imgs,
                    "video_path": video_path,
                    "frame_id": frame_id,
                    "bbox": bbox_coords,
                    "conf": 1.0
                })

            # Add negative samples (optional)
            if self.include_negatives:
                try:
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if total_frames > 0:
                        # Find frames without annotations
                        negative_frames = [
                            f for f in range(total_frames) 
                            if f not in ann_frames
                        ]
                        
                        # Sample negative frames (ratio of positive samples)
                        num_positives = len(ann_frames[:self.max_frames])
                        num_negatives = max(1, int(num_positives * self.negative_ratio))
                        num_negatives = min(num_negatives, len(negative_frames), self.max_frames)
                        
                        if num_negatives > 0:
                            sampled_negatives = random.sample(negative_frames, num_negatives)
                            for neg_frame_id in sampled_negatives:
                                samples.append({
                                    "video_id": video_name,
                                    "ref_imgs": ref_imgs,
                                    "video_path": video_path,
                                    "frame_id": neg_frame_id,
                                    "bbox": None,  # No bbox for negatives
                                    "conf": 0.0
                                })
                except Exception as e:
                    # Skip negative sampling if video reading fails
                    print(f"Warning: Could not sample negatives for {video_name}: {e}")

        return samples

    def __len__(self) -> int:
        return len(self.sample_list)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, Dict]:
        """
        Get a sample from the dataset
        
        Returns:
            ref_imgs: List of 3 reference image tensors [C, H, W]
            frame: Video frame tensor [C, H, W]
            target: Dict with:
                - 'boxes': Tensor [N, 4] - [x1, y1, x2, y2] format
                - 'labels': Tensor [N] - 1.0 for positive, 0.0 for negative
                - 'image_id': Tensor [1]
                - 'size': Tensor [2] - (H, W)
                - 'orig_size': Tensor [2] - original (H, W)
        """
        sample = self.sample_list[idx]

        # Load 3 reference images
        ref_imgs = []
        for ref_path in sample["ref_imgs"]:
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                if self.ref_transforms:
                    ref_img = self.ref_transforms(ref_img)
                ref_imgs.append(ref_img)
            except Exception as e:
                # Fallback: use first ref image if loading fails
                if len(ref_imgs) > 0:
                    ref_imgs.append(ref_imgs[0].clone() if isinstance(ref_imgs[0], torch.Tensor) else ref_imgs[0])
                else:
                    raise ValueError(f"Cannot load reference image {ref_path}: {e}")

        # Load video frame
        frame = self.load_frame(sample["video_path"], sample["frame_id"])
        orig_size = frame.size  # (W, H)
        
        # Apply transforms to frame
        if self.transforms:
            frame = self.transforms(frame)
        else:
            # Default transform if none provided
            frame = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])(frame)

        # Build target dict
        image_id = torch.tensor([idx], dtype=torch.int64)
        
        if sample["conf"] > 0 and sample["bbox"] is not None:
            # Positive sample: has bbox
            bbox = sample["bbox"]
            # Normalize bbox coordinates to [0, 1] based on original image size
            w, h = orig_size
            bbox_normalized = [
                bbox[0] / w,  # x1
                bbox[1] / h,  # y1
                bbox[2] / w,  # x2
                bbox[3] / h,  # y2
            ]
            boxes = torch.tensor([bbox_normalized], dtype=torch.float32)
            labels = torch.tensor([1.0], dtype=torch.float32)
        else:
            # Negative sample: no bbox
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.float32)

        # Get current frame size (after transforms)
        if isinstance(frame, torch.Tensor):
            _, frame_h, frame_w = frame.shape
        else:
            frame_h, frame_w = frame.size[1], frame.size[0]

        target = {
            'boxes': boxes,  # [N, 4] in normalized [0, 1] coordinates
            'labels': labels,  # [N] - 1.0 for positive, 0.0 for negative
            'image_id': image_id,
            'size': torch.tensor([frame_h, frame_w], dtype=torch.int64),
            'orig_size': torch.tensor([orig_size[1], orig_size[0]], dtype=torch.int64),  # (H, W)
        }

        return ref_imgs, frame, target

    def load_frame(self, video_path: str, frame_id: int) -> Image.Image:
        """Extract frame from video at given frame_id"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_id} from {video_path}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)


def aeroeyes_collate_fn(batch: List[Tuple]) -> Tuple[List[List[torch.Tensor]], NestedTensor, List[Dict]]:
    """
    Collate function for AeroEyes dataset
    
    Args:
        batch: List of (ref_imgs, frame, target) tuples
    
    Returns:
        ref_imgs_batch: List[List[Tensor]] - List of batches, each containing 3 ref images
        frames_batch: NestedTensor - Batched frames with mask
        targets_batch: List[Dict] - List of target dicts
    """
    ref_imgs_list = []
    frames_list = []
    targets_list = []
    
    for ref_imgs, frame, target in batch:
        # ref_imgs is already a list of 3 tensors
        ref_imgs_list.append(ref_imgs)
        
        # frame is already a tensor [C, H, W]
        frames_list.append(frame)
        
        # target is already a dict
        targets_list.append(target)
    
    # Stack frames into batch
    frames_tensor = torch.stack(frames_list)  # [B, C, H, W]
    
    # Create mask for NestedTensor (all False for now - can be improved for variable sizes)
    # Mask indicates padding: True = padding, False = valid
    B, C, H, W = frames_tensor.shape
    frames_mask = torch.zeros(B, H, W, dtype=torch.bool)
    
    # Create NestedTensor
    frames_nested = NestedTensor(frames_tensor, frames_mask)
    
    return ref_imgs_list, frames_nested, targets_list


def build_aeroeyes(image_set, args, resolution):
    """
    Build AeroEyes dataset for training/validation
    
    Args:
        image_set: 'train', 'val', or 'test'
        args: Arguments object with dataset configuration
        resolution: Target resolution for images
    
    Returns:
        AeroEyesDataset instance
    """
    from pathlib import Path
    from rfdetr.datasets.transforms import SquareResize
    import torchvision.transforms as T
    
    # Get dataset directory
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided AeroEyes path {root} does not exist'
    
    # Determine which split to use
    if image_set == 'train':
        dataset_dir = root / "observing_train"
    elif image_set in ['val', 'test']:
        # For now, use same directory for val/test
        # Can be extended to support separate val/test splits
        dataset_dir = root / "observing_train"  # or "observing_val" if available
    else:
        raise ValueError(f'Unknown image_set: {image_set}')
    
    assert dataset_dir.exists(), f'AeroEyes {image_set} directory not found: {dataset_dir}'
    
    # Create transforms
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Use square resize for AeroEyes (similar to square_resize_div_64)
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = True  # Default to True for AeroEyes
    
    try:
        multi_scale = args.multi_scale
    except:
        multi_scale = False
    
    if image_set == 'train':
        if square_resize_div_64:
            transforms = T.Compose([
                T.RandomHorizontalFlip(),
                SquareResize([resolution]),
                normalize,
            ])
        else:
            transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomResize([resolution], max_size=1333),
                normalize,
            ])
    else:  # val or test
        if square_resize_div_64:
            transforms = T.Compose([
                SquareResize([resolution]),
                normalize,
            ])
        else:
            transforms = T.Compose([
                T.RandomResize([resolution], max_size=1333),
                normalize,
            ])
    
    # Reference images use same transforms (can be customized)
    ref_transforms = transforms
    
    # Create dataset
    dataset = AeroEyesDataset(
        root_dir=str(dataset_dir),
        transforms=transforms,
        ref_transforms=ref_transforms,
        max_frames_per_video=getattr(args, 'max_frames_per_video', 200),
        include_negatives=getattr(args, 'include_negatives', True),
        negative_ratio=getattr(args, 'negative_ratio', 0.3),
    )
    
    return dataset
