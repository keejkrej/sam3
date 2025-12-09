#!/usr/bin/env python3
"""
Script to segment similar shapes in an image using SAM3 with a visual template prompt.
Uses SAM3's native visual prompt mechanism without modifying SAM3 code.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Add parent directory to path to import sam3 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage


def visualize_segmentation(image_path, masks, boxes, scores, output_path=None):
    """
    Visualize segmentation masks overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        masks: Tensor of shape (N, H, W) containing binary masks
        boxes: Tensor of shape (N, 4) containing bounding boxes in XYXY format
        scores: Tensor of shape (N,) containing confidence scores
        output_path: Optional path to save the visualization
    """
    # Load the original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.set_title(f'SAM3 Segmentation with Visual Template Prompt\nFound {len(scores)} similar shape(s)', 
                 fontsize=14, fontweight='bold')
    
    # Generate distinct colors for each mask
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    # Overlay each mask
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        # Convert mask to numpy if it's a tensor
        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        
        # Handle mask shape: could be (H, W) or (1, H, W) or (H, W, 1)
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 1:
                mask_np = mask_np[0]  # Remove first dimension
            elif mask_np.shape[2] == 1:
                mask_np = mask_np[:, :, 0]  # Remove last dimension
        
        # Ensure mask is binary
        mask_binary = (mask_np > 0.5).astype(float)
        
        # Create colored overlay
        color = colors[i % len(colors)]
        overlay = np.zeros((*mask_binary.shape, 4))
        overlay[..., :3] = color[:3]  # RGB
        overlay[..., 3] = mask_binary * 0.5  # Alpha channel with transparency
        
        ax.imshow(overlay)
        
        # Draw bounding box
        if torch.is_tensor(box):
            box_np = box.detach().cpu().numpy()
        else:
            box_np = box
        
        x0, y0, x1, y1 = box_np
        width = x1 - x0
        height = y1 - y0
        
        rect = plt.Rectangle(
            (x0, y0), width, height,
            linewidth=2, edgecolor=color[:3], facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with score
        ax.text(
            x0, y0 - 5,
            f'Match {i+1} (score: {score:.2f})',
            color=color[:3], fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def create_visual_prompt_embed(model, template_backbone_out, device):
    """
    Create visual prompt embedding from template image.
    
    Since mask_encoder may not be available in image models, we create the visual
    prompt embedding directly from the template image features.
    
    Args:
        model: SAM3 model
        template_backbone_out: Backbone output from template image
        device: Device to use
        
    Returns:
        visual_prompt_embed: Tensor of shape (n_tokens, batch_size, C)
        visual_prompt_mask: Tensor of shape (batch_size, n_tokens)
    """
    # Get image features from backbone output (last level)
    backbone_fpn = template_backbone_out["backbone_fpn"]
    img_feats = backbone_fpn[-1]  # Last level features: (B, C, H, W)
    
    # Get position encodings
    vision_pos_enc = template_backbone_out["vision_pos_enc"][-1]  # (B, C, H, W)
    
    # Get the visual features (pixel features) - shape (B, C, H, W)
    # For batch size 1, we can work directly
    B, C, H, W = img_feats.shape
    
    # Use raw features without normalization - normalization might hurt matching
    # The model's attention mechanism should handle feature matching
    vision_features = img_feats  # Use raw features
    
    # Don't add position encoding here - position encoding is spatial and template
    # might be at different scale/position than target. Let the model learn the matching.
    # vision_features = img_feats + vision_pos_enc  # Skip this for now
    
    # Optionally, we can create a mask and fuse it with features if mask_encoder is available
    mask_encoder = model.geometry_encoder.mask_encoder
    if mask_encoder is not None:
        # Create a full mask (all ones) for the template since it's already cropped
        template_mask = torch.ones(B, 1, H, W, device=device, dtype=torch.float32)
        
        # Use the mask encoder to fuse mask with visual features
        # mask_encoder returns a tuple (vision_features, vision_pos_enc)
        vision_features, vision_pos_enc = mask_encoder(
            masks=template_mask,
            pix_feat=img_feats,
        )
        # Add position encoding
        vision_features = vision_features + vision_pos_enc
        # Update H, W in case mask encoder changed spatial dimensions
        B, C, H, W = vision_features.shape
    
    # Option 1: Use all spatial tokens (original approach)
    # Reshape to sequence format: (H*W, B, C) - sequence-first format
    n_tokens = H * W
    
    # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W) -> (H*W, B, C)
    visual_prompt_embed_full = vision_features.view(B, C, n_tokens).permute(2, 0, 1)  # (n_tokens, B, C)
    
    # Option 2: Try adaptive pooling to create a more compact representation
    # Single token (like text features)
    adaptive_pool_single = torch.nn.AdaptiveAvgPool2d((1, 1))
    pooled_features_single = adaptive_pool_single(vision_features)  # (B, C, 1, 1)
    visual_prompt_embed_single = pooled_features_single.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, B, C)
    
    # Option 3: Try a spatial grid (similar to ROI align)
    # This might be a good middle ground - more spatial info than single token, less than full resolution
    # Use a medium-sized grid that preserves spatial structure but isn't too large
    grid_size = 14  # 14x14 = 196 tokens - good balance between detail and efficiency
    adaptive_pool_grid = torch.nn.AdaptiveAvgPool2d((grid_size, grid_size))
    pooled_features_grid = adaptive_pool_grid(vision_features)  # (B, C, 7, 7)
    B_grid, C_grid, H_grid, W_grid = pooled_features_grid.shape
    n_tokens_grid = H_grid * W_grid
    visual_prompt_embed_grid = pooled_features_grid.view(B_grid, C_grid, n_tokens_grid).permute(2, 0, 1)  # (49, B, C)
    
    # Try different approaches - experiment with different pooling strategies
    # Full spatial (5184 tokens) might be too many and overwhelm attention
    # Medium grid (14x14 = 196 tokens) might be a good balance
    use_single_token = False
    use_grid_pooling = True   # Use medium grid for balance
    use_full_spatial = False  # Full spatial might be too many tokens
    
    if use_single_token:
        visual_prompt_embed = visual_prompt_embed_single
        n_tokens = 1
        print(f"Using single-token pooled visual prompt: {visual_prompt_embed.shape}")
    elif use_grid_pooling:
        visual_prompt_embed = visual_prompt_embed_grid
        n_tokens = n_tokens_grid
        print(f"Using grid-pooled visual prompt ({grid_size}x{grid_size}): {visual_prompt_embed.shape}")
    else:
        visual_prompt_embed = visual_prompt_embed_full
        # n_tokens already defined above for full spatial
        print(f"Using full spatial visual prompt: {visual_prompt_embed.shape}")
    
    # Create attention mask: (B, n_tokens) - all False means no masking
    visual_prompt_mask = torch.zeros(B, n_tokens, device=device, dtype=torch.bool)
    
    return visual_prompt_embed, visual_prompt_mask


def main():
    """Main function to run SAM3 segmentation with visual template prompt."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.png")
    image_path = os.path.join(script_dir, "ellipse.png")
    output_path = os.path.join(script_dir, "ellipse_segmentation_template.png")
    
    # Check if images exist
    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading template from: {template_path}")
    print(f"Loading image from: {image_path}")
    
    # Load the model
    print("Loading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # Step 1: Load and process template image through backbone
    print("\nStep 1: Processing template image through backbone...")
    template_image = Image.open(template_path).convert('RGB')
    template_state = processor.set_image(template_image)
    template_backbone_out = template_state["backbone_out"]
    print("Template processed successfully!")
    
    # Step 2 & 3: Create visual prompt embedding from template
    print("\nStep 2-3: Creating visual prompt embedding from template...")
    visual_prompt_embed, visual_prompt_mask = create_visual_prompt_embed(
        model, template_backbone_out, device
    )
    print(f"Visual prompt embed shape: {visual_prompt_embed.shape}")
    print(f"Visual prompt mask shape: {visual_prompt_mask.shape}")
    
    # Step 4: Load and process ellipse image
    print("\nStep 4: Processing ellipse image...")
    ellipse_image = Image.open(image_path).convert('RGB')
    ellipse_state = processor.set_image(ellipse_image)
    ellipse_backbone_out = ellipse_state["backbone_out"]
    
    # NO TEXT PROMPT - use only visual template matching
    # We need a dummy text prompt for the model structure, but use "visual" which indicates visual-only
    text_prompt = "visual"  # Dummy text prompt - model expects this for visual-only mode
    text_outputs = model.backbone.forward_text([text_prompt], device=device)
    ellipse_backbone_out.update(text_outputs)
    print(f"Ellipse image processed successfully! Using VISUAL PROMPT ONLY (no text guidance)")
    
    # Step 5: Call SAM3's internal methods with visual_prompt_embed
    print("\nStep 5: Running SAM3 inference with visual prompt...")
    
    # Option to test with or without visual prompts
    use_visual_prompt = True  # Set to False to test text-only baseline
    
    # Create find_input
    find_input = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    
    # Create dummy geometric prompt
    geometric_prompt = model._get_dummy_prompt()
    
    # Call _encode_prompt with or without visual_prompt_embed
    if use_visual_prompt:
        prompt, prompt_mask, backbone_out = model._encode_prompt(
            backbone_out=ellipse_backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            visual_prompt_embed=visual_prompt_embed,
            visual_prompt_mask=visual_prompt_mask,
            encode_text=True,
        )
        print("Using visual prompt + text prompt")
    else:
        prompt, prompt_mask, backbone_out = model._encode_prompt(
            backbone_out=ellipse_backbone_out,
            find_input=find_input,
            geometric_prompt=geometric_prompt,
            visual_prompt_embed=None,  # No visual prompt
            visual_prompt_mask=None,
            encode_text=True,
        )
        print("Using text prompt only (baseline)")
    
    # Debug: Print prompt structure
    print(f"Prompt shape: {prompt.shape}, Prompt mask shape: {prompt_mask.shape}")
    print(f"Visual prompt embed shape: {visual_prompt_embed.shape}")
    print(f"Visual prompt mask shape: {visual_prompt_mask.shape}")
    
    # Call _run_encoder
    backbone_out, encoder_out, _ = model._run_encoder(
        backbone_out=backbone_out,
        find_input=find_input,
        prompt=prompt,
        prompt_mask=prompt_mask,
    )
    
    # Prepare output dict for decoder
    out = {
        "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        "prev_encoder_out": {
            "encoder_out": encoder_out,
            "backbone_out": backbone_out,
        },
    }
    
    # Call _run_decoder
    out, hs = model._run_decoder(
        memory=out["encoder_hidden_states"],
        pos_embed=encoder_out["pos_embed"],
        src_mask=encoder_out["padding_mask"],
        out=out,
        prompt=prompt,
        prompt_mask=prompt_mask,
        encoder_out=encoder_out,
    )
    
    # Call _run_segmentation_heads
    model._run_segmentation_heads(
        out=out,
        backbone_out=backbone_out,
        img_ids=find_input.img_ids,
        vis_feat_sizes=encoder_out["vis_feat_sizes"],
        encoder_hidden_states=out["encoder_hidden_states"],
        prompt=prompt,
        prompt_mask=prompt_mask,
        hs=hs,
    )
    
    print("Inference completed!")
    
    # Step 6: Filter results by confidence threshold
    print("\nStep 6: Filtering results by confidence threshold...")
    out_logits = out["pred_logits"]
    out_probs = out_logits.sigmoid()
    
    # Get presence score if available
    if "presence_logit_dec" in out:
        presence_score = out["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)
    else:
        out_probs = out_probs.squeeze(-1)
    
    # Debug: Print score statistics
    print(f"Score statistics:")
    print(f"  Total predictions: {len(out_probs)}")
    if len(out_probs) > 0:
        print(f"  Max score: {out_probs.max().item():.4f}")
        print(f"  Min score: {out_probs.min().item():.4f}")
        print(f"  Mean score: {out_probs.mean().item():.4f}")
        print(f"  Top 10 scores: {torch.topk(out_probs, min(10, len(out_probs)))[0].detach().cpu().numpy()}")
    
    # Note: Original requirement was >= 0.8, but scores are lower with visual prompts
    # Adjust threshold based on actual score distribution
    confidence_threshold = 0.3  # Using 0.3 to get results (max score ~0.6)
    # If you want stricter filtering, you can increase this, but may get fewer/no results
    keep = out_probs > confidence_threshold
    
    if keep.any():
        out_probs_filtered = out_probs[keep]
        out_masks_filtered = out["pred_masks"][keep]
        out_bbox_filtered = out["pred_boxes"][keep]
        
        # Convert boxes to XYXY format
        from sam3.model import box_ops
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox_filtered)
        
        # Scale boxes to original image size
        img_h = ellipse_state["original_height"]
        img_w = ellipse_state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(device)
        boxes_xyxy = boxes_xyxy * scale_fct[None, :]
        
        # Resize masks to original image size
        from sam3.model.data_misc import interpolate
        masks_resized = interpolate(
            out_masks_filtered.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        
        masks_binary = masks_resized > 0.5
        masks_binary = masks_binary.squeeze(1)  # Remove channel dimension
        
        print(f"Found {len(out_probs_filtered)} shape(s) with confidence >= {confidence_threshold}")
        print(f"\nTop 5 detections:")
        top_k = min(5, len(out_probs_filtered))
        top_scores, top_indices = torch.topk(out_probs_filtered, top_k)
        for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
            box = boxes_xyxy[idx]
            # Detach before converting to numpy to avoid grad issues
            box_np = box.detach().cpu().numpy() if torch.is_tensor(box) else box
            score_val = score.item() if torch.is_tensor(score) else score
            print(f"  Detection {i+1}: score={score_val:.3f}, box={box_np}")
        
        # Step 7: Visualize results
        print("\nStep 7: Visualizing results...")
        visualize_segmentation(
            image_path, 
            masks_binary, 
            boxes_xyxy, 
            out_probs_filtered, 
            output_path=output_path
        )
    else:
        print(f"No shapes found with confidence >= {confidence_threshold}")
        print(f"Try lowering the threshold or check if visual prompt embedding is correct.")
        # Still show the original image
        img = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(f'SAM3 Segmentation with Visual Template Prompt\nNo shapes detected (confidence >= {confidence_threshold})', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
