# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Siamese DETR Model for One-Shot Object Detection
# Wraps LWDETR with Siamese architecture
# ------------------------------------------------------------------------

"""
Siamese DETR Model for One-Shot Object Detection
Wraps LWDETR with Siamese architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from rfdetr.models.lwdetr import LWDETR
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list


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
        self.group_detr = lwdetr_model.group_detr
        self.bbox_reparam = lwdetr_model.bbox_reparam
        self.aux_loss = lwdetr_model.aux_loss
        self.two_stage = lwdetr_model.two_stage
    
    def reinitialize_detection_head(self, num_classes):
        """
        Reinitialize detection head for different number of classes
        Delegates to wrapped LWDETR model
        """
        self.lwdetr_model.reinitialize_detection_head(num_classes)
        
    def forward(
        self,
        ref_imgs: List[List[torch.Tensor]],
        target_img: NestedTensor,
        targets: Optional[List[Dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Siamese DETR
        
        Args:
            ref_imgs: List[List[Tensor]] - [B, 3, C, H, W] - 3 ref images per sample
            target_img: NestedTensor or Tensor - [B, C, H, W] - target frames
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
                
                # Create NestedTensor for ref image
                ref_mask = torch.zeros(1, ref_img_batch.shape[2], ref_img_batch.shape[3], 
                                     dtype=torch.bool, device=device)
                ref_nested = NestedTensor(ref_img_batch, ref_mask)
                
                # Backbone forward
                ref_feats, _ = self.backbone(ref_nested)
                
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
        if isinstance(target_img, (list, torch.Tensor)):
            target_img = nested_tensor_from_tensor_list(target_img)
        
        feats, poss = self.backbone(target_img)
        
        # Decompose features
        srcs = []
        masks = []
        for feat in feats:
            src, m = feat.decompose()
            srcs.append(src)
            masks.append(m)
            assert m is not None
        
        # Get query embeddings
        if self.training:
            refpoint_embed_weight = self.lwdetr_model.refpoint_embed.weight
            query_feat_weight = self.lwdetr_model.query_feat.weight
        else:
            refpoint_embed_weight = self.lwdetr_model.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.lwdetr_model.query_feat.weight[:self.num_queries]
        
        # Transformer forward
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, poss, refpoint_embed_weight, query_feat_weight
        )
        
        if hs is None:
            raise ValueError("Transformer returned None")
        
        # Get object embeddings from last decoder layer
        # hs is stacked tensor [num_layers, B, Q, D] or list
        object_embeddings = hs[-1]  # [B, Q, D]
        
        # Predict boxes (ref_unsigmoid is stacked tensor [num_layers, B, Q, 4])
        # Get last layer reference points
        ref_unsigmoid_last = ref_unsigmoid[-1]  # [B, Q, 4]
        
        if self.bbox_reparam:
            outputs_coord_delta = self.bbox_embed(object_embeddings)  # [B, Q, 4]
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid_last[..., 2:] + ref_unsigmoid_last[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid_last[..., 2:]
            pred_boxes = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            pred_boxes = (self.bbox_embed(object_embeddings) + ref_unsigmoid_last).sigmoid()
        
        # Dummy logits for compatibility (not used in loss)
        pred_logits = torch.zeros(batch_size, self.num_queries, 1, device=device)
        
        # Build output dict
        outputs = {
            'v_ref': v_ref,  # [B, D]
            'object_embeddings': object_embeddings,  # [B, Q, D]
            'pred_boxes': pred_boxes,  # [B, Q, 4]
            'pred_logits': pred_logits,  # [B, Q, 1] - dummy
        }
        
        # Add aux outputs if needed
        if self.aux_loss and hs is not None:
            aux_outputs = []
            num_layers = len(hs) if isinstance(hs, (list, tuple)) else hs.shape[0]
            for i in range(num_layers - 1):
                aux_obj_embeds = hs[i] if isinstance(hs, (list, tuple)) else hs[i]
                ref_unsigmoid_i = ref_unsigmoid[i] if isinstance(ref_unsigmoid, (list, tuple)) else ref_unsigmoid[i]
                if self.bbox_reparam:
                    aux_coord_delta = self.bbox_embed(aux_obj_embeds)
                    aux_coord_cxcy = aux_coord_delta[..., :2] * ref_unsigmoid_i[..., 2:] + ref_unsigmoid_i[..., :2]
                    aux_coord_wh = aux_coord_delta[..., 2:].exp() * ref_unsigmoid_i[..., 2:]
                    aux_coord = torch.cat([aux_coord_cxcy, aux_coord_wh], dim=-1)
                else:
                    aux_coord = (self.bbox_embed(aux_obj_embeds) + ref_unsigmoid_i).sigmoid()
                aux_logits = torch.zeros(batch_size, self.num_queries, 1, device=device)
                aux_outputs.append({
                    'v_ref': v_ref,
                    'object_embeddings': aux_obj_embeds,
                    'pred_boxes': aux_coord,
                    'pred_logits': aux_logits,
                })
            outputs['aux_outputs'] = aux_outputs
        
        return outputs

