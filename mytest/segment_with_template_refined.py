#!/usr/bin/env python3
"""
Script to segment similar shapes using SAM3 with template matching and refinement.
1. First pass: Embed template in image, get initial detections
2. Second pass: Use top detection(s) as box prompts on original image to find more matches
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add parent directory to path to import sam3 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def visualize_segmentation(image_path, masks, boxes, scores, output_path=None, title_suffix=""):
    """
    Visualize segmentation masks overlaid on the original image.
    """
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.set_title(f'SAM3 Template Matching{title_suffix}\nFound {len(scores)} similar shape(s)', 
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        if torch.is_tensor(mask):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            elif mask_np.shape[2] == 1:
                mask_np = mask_np[:, :, 0]
        
        mask_binary = (mask_np > 0.5).astype(float)
        
        color = colors[i % len(colors)]
        overlay = np.zeros((*mask_binary.shape, 4))
        overlay[..., :3] = color[:3]
        overlay[..., 3] = mask_binary * 0.5
        
        ax.imshow(overlay)
        
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


def create_composite_image(target_image, template_image):
    """Create composite image with template in top-left corner."""
    target_array = np.array(target_image)
    template_array = np.array(template_image)
    
    target_h, target_w = target_array.shape[:2]
    template_h, template_w = template_array.shape[:2]
    
    composite = target_array.copy()
    composite[0:template_h, 0:template_w] = template_array
    
    template_bbox_xyxy = np.array([
        0.0,
        0.0,
        template_w / target_w,
        template_h / target_h
    ])
    
    composite_image = Image.fromarray(composite)
    return composite_image, template_bbox_xyxy


def main():
    """Main function with two-pass detection."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.png")
    image_path = os.path.join(script_dir, "ellipse.png")
    output_path = os.path.join(script_dir, "ellipse_segmentation_refined.png")
    
    # Check if images exist
    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading template from: {template_path}")
    print(f"Loading image from: {image_path}")
    
    # Load images
    template_image = Image.open(template_path).convert('RGB')
    target_image = Image.open(image_path).convert('RGB')
    
    original_w, original_h = target_image.size
    template_w, template_h = template_image.size
    print(f"Original image size: {original_w}x{original_h}")
    print(f"Template size: {template_w}x{template_h}")
    
    # Load model
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # ===== FIRST PASS: Template embedding =====
    print("\n" + "="*60)
    print("FIRST PASS: Template Embedding")
    print("="*60)
    
    # Create composite image
    composite_image, template_bbox_xyxy = create_composite_image(target_image, template_image)
    
    # Process composite image
    print("\nProcessing composite image...")
    composite_state = processor.set_image(composite_image)
    
    # Add box prompt around template
    cx = (template_bbox_xyxy[0] + template_bbox_xyxy[2]) / 2
    cy = (template_bbox_xyxy[1] + template_bbox_xyxy[3]) / 2
    w = template_bbox_xyxy[2] - template_bbox_xyxy[0]
    h = template_bbox_xyxy[3] - template_bbox_xyxy[1]
    box_prompt = [cx, cy, w, h]
    
    text_outputs = model.backbone.forward_text(["visual"], device=device)
    composite_state["backbone_out"].update(text_outputs)
    
    output_state = processor.add_geometric_prompt(
        box=box_prompt,
        label=True,
        state=composite_state
    )
    
    # Extract first pass results
    masks_first = output_state["masks"]
    boxes_first = output_state["boxes"]
    scores_first = output_state["scores"]
    
    print(f"Found {len(scores_first)} detection(s) in first pass")
    
    # Filter by size
    size_ratio_min = 0.8
    size_ratio_max = 1.2
    
    if torch.is_tensor(boxes_first):
        boxes_np = boxes_first.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes_first)
    
    detection_widths = boxes_np[:, 2] - boxes_np[:, 0]
    detection_heights = boxes_np[:, 3] - boxes_np[:, 1]
    width_ratios = detection_widths / template_w
    height_ratios = detection_heights / template_h
    
    width_mask = (width_ratios >= size_ratio_min) & (width_ratios <= size_ratio_max)
    height_mask = (height_ratios >= size_ratio_min) & (height_ratios <= size_ratio_max)
    size_mask = width_mask & height_mask
    size_filtered_indices = np.where(size_mask)[0]
    
    print(f"After size filtering: {len(size_filtered_indices)}/{len(scores_first)} detections remain")
    
    if len(size_filtered_indices) == 0:
        print("No detections found in first pass. Cannot proceed to second pass.")
        return
    
    # Get top detection(s) for second pass
    # Use top 1 or top 2 detections as prompts
    num_prompts = min(2, len(size_filtered_indices))  # Use up to 2 top detections
    
    # Sort by score (if available) or just take first N
    if torch.is_tensor(scores_first):
        scores_np = scores_first.detach().cpu().numpy()
    else:
        scores_np = np.array(scores_first)
    
    # Get top N by score
    top_indices = np.argsort(scores_np[size_filtered_indices])[-num_prompts:][::-1]
    prompt_indices = size_filtered_indices[top_indices]
    
    boxes_for_prompt = boxes_np[prompt_indices]  # These are in composite image coordinates
    
    # Convert to original image coordinates (for extend methods, but for replace_patch they're already correct)
    # Since we use replace_patch, boxes are already in original coordinates
    boxes_prompt_original = boxes_for_prompt.copy()
    
    print(f"\nUsing top {num_prompts} detection(s) as prompt(s) for second pass:")
    for i, idx in enumerate(prompt_indices):
        score_val = scores_np[idx]
        box = boxes_prompt_original[i]
        print(f"  Prompt {i+1}: score={score_val:.3f}, box={box}")
    
    # ===== SECOND PASS: Use detections as prompts on original image =====
    print("\n" + "="*60)
    print("SECOND PASS: Refinement on Original Image")
    print("="*60)
    
    # Process original image
    print("\nProcessing original image...")
    original_state = processor.set_image(target_image)
    
    # Use "visual" text prompt
    text_outputs = model.backbone.forward_text(["visual"], device=device)
    original_state["backbone_out"].update(text_outputs)
    
    # Add box prompts from first pass detections
    print(f"\nAdding {num_prompts} box prompt(s) from first pass detections...")
    for i, box_xyxy in enumerate(boxes_prompt_original):
        # Convert XYXY to normalized [center_x, center_y, width, height]
        x0, y0, x1, y1 = box_xyxy
        cx = (x0 + x1) / 2 / original_w
        cy = (y0 + y1) / 2 / original_h
        w = (x1 - x0) / original_w
        h = (y1 - y0) / original_h
        box_prompt_normalized = [cx, cy, w, h]
        
        print(f"  Box prompt {i+1}: center=({cx:.3f}, {cy:.3f}), size=({w:.3f}, {h:.3f})")
        
        original_state = processor.add_geometric_prompt(
            box=box_prompt_normalized,
            label=True,
            state=original_state
        )
    
    # Extract second pass results
    masks_second = original_state["masks"]
    boxes_second = original_state["boxes"]
    scores_second = original_state["scores"]
    
    print(f"\nFound {len(scores_second)} detection(s) in second pass")
    
    # Filter second pass results by size
    if torch.is_tensor(boxes_second):
        boxes_second_np = boxes_second.detach().cpu().numpy()
    else:
        boxes_second_np = np.array(boxes_second)
    
    detection_widths_2 = boxes_second_np[:, 2] - boxes_second_np[:, 0]
    detection_heights_2 = boxes_second_np[:, 3] - boxes_second_np[:, 1]
    width_ratios_2 = detection_widths_2 / template_w
    height_ratios_2 = detection_heights_2 / template_h
    
    width_mask_2 = (width_ratios_2 >= size_ratio_min) & (width_ratios_2 <= size_ratio_max)
    height_mask_2 = (height_ratios_2 >= size_ratio_min) & (height_ratios_2 <= size_ratio_max)
    size_mask_2 = width_mask_2 & height_mask_2
    size_filtered_indices_2 = np.where(size_mask_2)[0]
    
    print(f"After size filtering: {len(size_filtered_indices_2)}/{len(scores_second)} detections remain")
    
    if len(size_filtered_indices_2) == 0:
        print("No detections found in second pass matching size criteria")
        # Show original image
        img = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title('SAM3 Template Matching (Refined)\nNo shapes detected in second pass', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    # Prepare final results
    masks_final = masks_second[size_filtered_indices_2] if torch.is_tensor(masks_second) else [masks_second[i] for i in size_filtered_indices_2]
    boxes_final = boxes_second_np[size_filtered_indices_2]
    scores_final = scores_second[size_filtered_indices_2] if torch.is_tensor(scores_second) else scores_second[size_filtered_indices_2]
    
    # Convert masks to tensor
    if torch.is_tensor(masks_final):
        masks_tensor = masks_final.float()
    else:
        masks_list = []
        for m in masks_final:
            if isinstance(m, np.ndarray):
                masks_list.append(torch.tensor(m, dtype=torch.float32))
            elif torch.is_tensor(m):
                masks_list.append(m.float())
            else:
                masks_list.append(torch.tensor(m, dtype=torch.float32))
        masks_tensor = torch.stack(masks_list)
    
    if masks_tensor.dim() == 2:
        masks_tensor = masks_tensor.unsqueeze(0)
    
    if masks_tensor.dtype != torch.bool:
        masks_binary = masks_tensor > 0.5
    else:
        masks_binary = masks_tensor
    
    boxes_tensor = torch.tensor(boxes_final, device=device)
    
    print(f"\nFinal results: {len(size_filtered_indices_2)} shape(s)")
    print(f"\nTop detections:")
    filtered_width_ratios = width_ratios_2[size_filtered_indices_2]
    filtered_height_ratios = height_ratios_2[size_filtered_indices_2]
    for i, (box, score, w_ratio, h_ratio) in enumerate(zip(boxes_final, scores_final, filtered_width_ratios, filtered_height_ratios)):
        score_val = score.item() if torch.is_tensor(score) else score
        print(f"  Detection {i+1}: score={score_val:.3f}, width_ratio={w_ratio:.2f}x, height_ratio={h_ratio:.2f}x")
    
    # Visualize
    print("\nVisualizing results...")
    visualize_segmentation(
        image_path,
        masks_binary,
        boxes_tensor,
        scores_final,
        output_path=output_path,
        title_suffix=" (Refined - Two Pass)"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
