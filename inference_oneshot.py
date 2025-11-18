#!/usr/bin/env python3
"""
Inference script for One-Shot Object Detection using Siamese DETR

Usage:
    python inference_oneshot.py \
        --checkpoint path/to/checkpoint.pth \
        --ref_images ref1.jpg ref2.jpg ref3.jpg \
        --target_image target.jpg \
        --output_dir output \
        --similarity_threshold 0.7 \
        --device cuda
"""

import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import json
from typing import List, Union, Tuple, Dict
import cv2

from rfdetr.main import Model
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from rfdetr.util.box_ops import box_cxcywh_to_xyxy
from rfdetr.datasets.transforms import SquareResize, Normalize, ToTensor, Compose


def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Load and convert image to RGB"""
    img = Image.open(image_path).convert("RGB")
    return img


def preprocess_image(image: Image.Image, resolution: int, device: torch.device) -> torch.Tensor:
    """
    Preprocess image for inference
    
    Args:
        image: PIL Image
        resolution: Target resolution
        device: Device to move tensor to
    
    Returns:
        Preprocessed tensor [C, H, W]
    """
    transform = Compose([
        SquareResize([resolution]),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor, _ = transform(image, None)
    return img_tensor.to(device)


def preprocess_ref_images(ref_image_paths: List[Union[str, Path]], resolution: int, device: torch.device) -> List[torch.Tensor]:
    """
    Preprocess reference images
    
    Args:
        ref_image_paths: List of paths to reference images (should be 3)
        resolution: Target resolution
        device: Device to move tensors to
    
    Returns:
        List of preprocessed tensors [C, H, W]
    """
    ref_images = []
    for ref_path in ref_image_paths:
        ref_img = load_image(ref_path)
        ref_tensor = preprocess_image(ref_img, resolution, device)
        ref_images.append(ref_tensor)
    
    return ref_images


def postprocess_predictions(
    outputs: Dict[str, torch.Tensor],
    orig_size: Tuple[int, int],
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Postprocess model outputs to get final predictions
    
    Args:
        outputs: Model outputs dict with 'v_ref', 'object_embeddings', 'pred_boxes'
        orig_size: Original image size (H, W)
        similarity_threshold: Threshold for filtering predictions
    
    Returns:
        List of prediction dicts with 'boxes', 'scores', 'similarities'
    """
    v_ref = outputs["v_ref"]  # [B, D]
    obj_embeds = outputs["object_embeddings"]  # [B, Q, D]
    pred_boxes = outputs["pred_boxes"]  # [B, Q, 4] in normalized cxcywh format
    
    # Normalize embeddings for cosine similarity
    v_ref_norm = F.normalize(v_ref, p=2, dim=1)  # [B, D]
    obj_embeds_norm = F.normalize(obj_embeds, p=2, dim=2)  # [B, Q, D]
    
    # Compute similarity scores
    sim_scores = torch.einsum('bd,bqd->bq', v_ref_norm, obj_embeds_norm)  # [B, Q]
    
    batch_size = sim_scores.shape[0]
    orig_h, orig_w = orig_size
    
    all_predictions = []
    
    for b in range(batch_size):
        batch_sim = sim_scores[b]  # [Q]
        batch_boxes = pred_boxes[b]  # [Q, 4]
        
        # Filter by similarity threshold
        valid_mask = batch_sim > similarity_threshold
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            all_predictions.append({
                'boxes': [],
                'scores': [],
                'similarities': []
            })
            continue
        
        valid_sim = batch_sim[valid_indices].cpu().numpy()
        valid_boxes = batch_boxes[valid_indices]  # [N, 4]
        
        # Convert from normalized cxcywh to xyxy in original image coordinates
        # pred_boxes are in [0, 1] normalized format
        valid_boxes_xyxy = box_cxcywh_to_xyxy(valid_boxes)  # [N, 4] in [0, 1]
        
        # Scale to original image size
        boxes_scaled = valid_boxes_xyxy * torch.tensor([orig_w, orig_h, orig_w, orig_h], device=valid_boxes.device)
        boxes_scaled = boxes_scaled.cpu().numpy()
        
        # Convert to list format [x1, y1, x2, y2]
        boxes_list = boxes_scaled.tolist()
        
        all_predictions.append({
            'boxes': boxes_list,
            'scores': valid_sim.tolist(),
            'similarities': valid_sim.tolist()
        })
    
    return all_predictions


def draw_predictions(
    image: Image.Image,
    predictions: Dict,
    output_path: Union[str, Path],
    show_scores: bool = True
):
    """
    Draw bounding boxes on image and save
    
    Args:
        image: PIL Image
        predictions: Prediction dict with 'boxes' and 'scores'
        output_path: Path to save output image
        show_scores: Whether to show similarity scores
    """
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    boxes = predictions['boxes']
    scores = predictions['scores']
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw score label
        if show_scores:
            label = f"{score:.3f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_cv, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Convert back to RGB and save
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(img_rgb)
    output_img.save(output_path)
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="One-Shot Object Detection Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--ref_images", type=str, nargs=3, required=True,
                       help="Paths to 3 reference images")
    parser.add_argument("--target_image", type=str, required=True,
                       help="Path to target image")
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for results")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                       help="Similarity threshold for filtering predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")
    parser.add_argument("--resolution", type=int, default=560,
                       help="Input resolution")
    parser.add_argument("--save_visualization", action="store_true",
                       help="Save visualization with bounding boxes")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (optional)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load device with CUDA availability check
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            print("Note: PyTorch may not be compiled with CUDA support, or no GPU is available.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if checkpoint is a hosted model name
    HOSTED_MODELS = {
        "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "rf-detr-base-o365.pth": "https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth",
        "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
        "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
        "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    }
    
    checkpoint_path = args.checkpoint
    
    # If checkpoint is a hosted model name, download it
    if checkpoint_path in HOSTED_MODELS:
        if not os.path.exists(checkpoint_path):
            print(f"Downloading pretrained weights: {checkpoint_path}")
            from rfdetr.util.files import download_file
            download_file(HOSTED_MODELS[checkpoint_path], checkpoint_path)
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"\nERROR: Checkpoint file not found: {checkpoint_path}")
        print("\nAvailable pretrained checkpoints:")
        for name in HOSTED_MODELS.keys():
            print(f"  - {name}")
        print("\nYou can use one of these names directly, e.g.:")
        print("  --checkpoint rf-detr-base.pth")
        print("\nOr provide a path to your trained Siamese DETR checkpoint.")
        exit(1)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Start with default config to ensure all required attributes are present
    from rfdetr.config import RFDETRBaseConfig
    model_config = RFDETRBaseConfig()
    
    # Convert config to dict - this ensures all required attributes are present
    if hasattr(model_config, 'model_dump'):
        model_kwargs = model_config.model_dump()
    else:
        model_kwargs = vars(model_config)
    
    # Override with checkpoint args if available
    if 'args' in checkpoint:
        checkpoint_args = checkpoint['args']
        # Convert Namespace to dict if needed
        if hasattr(checkpoint_args, '__dict__'):
            checkpoint_dict = vars(checkpoint_args)
        elif isinstance(checkpoint_args, dict):
            checkpoint_dict = checkpoint_args
        else:
            checkpoint_dict = {}
        
        # Update model_kwargs with checkpoint values
        # This preserves default values from config if checkpoint doesn't have them
        for key, value in checkpoint_dict.items():
            # Only update if value is not None and key exists in model_kwargs
            if value is not None and key in model_kwargs:
                model_kwargs[key] = value
        
        # Update resolution from checkpoint if available
        if 'resolution' in checkpoint_dict and checkpoint_dict['resolution'] is not None:
            args.resolution = checkpoint_dict['resolution']
    
    # Set Siamese-specific config (override any existing values)
    model_kwargs['use_siamese'] = True
    model_kwargs['dataset_file'] = 'aeroeyes'
    model_kwargs['resolution'] = args.resolution
    
    # Don't set pretrain_weights to avoid Model.__init__ auto-loading
    # We'll load the checkpoint manually after initialization
    model_kwargs['pretrain_weights'] = None
    
    # Identify classification layers to exclude
    classification_keys_to_exclude = [
        'class_embed.weight',
        'class_embed.bias',
    ]
    # Also exclude enc_out_class_embed layers (for two_stage models)
    if 'model' in checkpoint:
        for key in list(checkpoint['model'].keys()):
            if 'enc_out_class_embed' in key:
                classification_keys_to_exclude.append(key)
    
    # Initialize model - Model.__init__ will call populate_args(**kwargs)
    # populate_args accepts **extra_kwargs, so any additional keys will be passed through
    model = Model(**model_kwargs)
    
    # Load checkpoint manually, filtering out classification layers
    if 'model' in checkpoint:
        # Filter out classification layers before loading
        filtered_state_dict = {}
        model_state_dict = model.model.state_dict()
        
        for key, value in checkpoint['model'].items():
            # Skip classification layers
            if any(exclude_key in key for exclude_key in classification_keys_to_exclude):
                continue
            
            # Only load if key exists in model and shape matches
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Skipping {key} due to shape mismatch: checkpoint {value.shape} vs model {model_state_dict[key].shape}")
            else:
                # Key not in model (e.g., SiameseDETR-specific keys), skip it
                pass
        
        model.model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded model weights from checkpoint (excluded {len(classification_keys_to_exclude)} classification layers)")
    else:
        print("Warning: No 'model' key in checkpoint, using pretrained weights")
    
    model.model.eval()
    model.model.to(device)
    
    # Load and preprocess images
    print("Loading images...")
    target_image = load_image(args.target_image)
    orig_size = target_image.size[::-1]  # (H, W) from (W, H)
    
    ref_images_tensors = preprocess_ref_images(args.ref_images, args.resolution, device)
    target_tensor = preprocess_image(target_image, args.resolution, device)
    
    # Prepare inputs
    # ref_imgs: List[List[Tensor]] - batch of 1, with 3 ref images
    ref_imgs_batch = [ref_images_tensors]
    
    # target: NestedTensor
    target_tensor_batch = target_tensor.unsqueeze(0)  # [1, C, H, W]
    target_mask = torch.zeros(1, target_tensor_batch.shape[2], target_tensor_batch.shape[3], 
                             dtype=torch.bool, device=device)
    target_nested = NestedTensor(target_tensor_batch, target_mask)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model.forward(target_nested, ref_imgs=ref_imgs_batch)
    
    # Postprocess predictions
    print("Postprocessing predictions...")
    predictions = postprocess_predictions(
        outputs,
        orig_size,
        similarity_threshold=args.similarity_threshold
    )
    
    # Get predictions for first (and only) batch item
    pred = predictions[0]
    
    # Print results
    print(f"\nFound {len(pred['boxes'])} detections (threshold={args.similarity_threshold})")
    for i, (box, score) in enumerate(zip(pred['boxes'], pred['scores'])):
        x1, y1, x2, y2 = box
        print(f"  Detection {i+1}: Box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], Similarity={score:.4f}")
    
    # Save results
    results = {
        'target_image': str(args.target_image),
        'reference_images': args.ref_images,
        'similarity_threshold': args.similarity_threshold,
        'num_detections': len(pred['boxes']),
        'detections': [
            {
                'box': box,
                'similarity_score': float(score)
            }
            for box, score in zip(pred['boxes'], pred['scores'])
        ]
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Save visualization if requested
    if args.save_visualization:
        vis_path = output_dir / "visualization.jpg"
        draw_predictions(target_image, pred, vis_path)
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()

