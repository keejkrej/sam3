#!/usr/bin/env python3
"""
Script to segment similar shapes using SAM3 by embedding the template into the target image.
This approach:
1. Embeds the template image into the target image (extends or replaces a patch)
2. Uses SAM3's box prompt on the template region
3. Finds similar shapes in the composite image
4. Converts results back to original image coordinates
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path to import sam3 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def visualize_segmentation(image_path, masks, boxes, scores, output_path=None, title_suffix=""):
    """
    Visualize segmentation masks overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        masks: Tensor of shape (N, H, W) containing binary masks
        boxes: Tensor of shape (N, 4) containing bounding boxes in XYXY format
        scores: Tensor of shape (N,) containing confidence scores
        output_path: Optional path to save the visualization
        title_suffix: Additional text for the title
    """
    # Load the original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    ax.set_title(f'SAM3 Template Matching{title_suffix}\nFound {len(scores)} similar shape(s)', 
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
                mask_np = mask_np[0]
            elif mask_np.shape[2] == 1:
                mask_np = mask_np[:, :, 0]
        
        # Ensure mask is binary
        mask_binary = (mask_np > 0.5).astype(float)
        
        # Create colored overlay
        color = colors[i % len(colors)]
        overlay = np.zeros((*mask_binary.shape, 4))
        overlay[..., :3] = color[:3]
        overlay[..., 3] = mask_binary * 0.5
        
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


def create_composite_image(target_image, template_image, method='extend_right', template_scale=1.0):
    """
    Create a composite image by embedding the template into the target image.
    
    Args:
        target_image: PIL Image of the target image
        template_image: PIL Image of the template
        method: 'extend_right', 'extend_bottom', or 'replace_patch'
        template_scale: Scale factor for template (1.0 = original size)
    
    Returns:
        composite_image: PIL Image with template embedded
        template_bbox_xyxy: Bounding box of template in composite image (x0, y0, x1, y1)
        offset_x, offset_y: Offset to convert from composite to original coordinates
    """
    target_array = np.array(target_image)
    template_array = np.array(template_image)
    
    target_h, target_w = target_array.shape[:2]
    template_h, template_w = template_array.shape[:2]
    
    # Scale template if needed
    if template_scale != 1.0:
        new_w = int(template_w * template_scale)
        new_h = int(template_h * template_scale)
        template_image_scaled = template_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        template_array = np.array(template_image_scaled)
        template_h, template_w = template_array.shape[:2]
    
    if method == 'extend_right':
        # Extend image to the right
        composite_w = target_w + template_w
        composite_h = max(target_h, template_h)
        composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        
        # Place target image on the left
        composite[:target_h, :target_w] = target_array
        
        # Place template on the right
        y_offset = (composite_h - template_h) // 2
        composite[y_offset:y_offset+template_h, target_w:target_w+template_w] = template_array
        
        # Template bbox in composite image (normalized)
        template_bbox_xyxy = np.array([
            target_w / composite_w,  # x0 (normalized)
            y_offset / composite_h,  # y0 (normalized)
            (target_w + template_w) / composite_w,  # x1 (normalized)
            (y_offset + template_h) / composite_h   # y1 (normalized)
        ])
        
        offset_x, offset_y = 0, 0  # No offset needed for target region
        
    elif method == 'extend_bottom':
        # Extend image to the bottom
        composite_w = max(target_w, template_w)
        composite_h = target_h + template_h
        composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
        
        # Place target image on top
        x_offset = (composite_w - target_w) // 2
        composite[:target_h, x_offset:x_offset+target_w] = target_array
        
        # Place template on bottom
        x_offset_template = (composite_w - template_w) // 2
        composite[target_h:target_h+template_h, x_offset_template:x_offset_template+template_w] = template_array
        
        # Template bbox in composite image (normalized)
        template_bbox_xyxy = np.array([
            x_offset_template / composite_w,  # x0 (normalized)
            target_h / composite_h,  # y0 (normalized)
            (x_offset_template + template_w) / composite_w,  # x1 (normalized)
            (target_h + template_h) / composite_h   # y1 (normalized)
        ])
        
        offset_x, offset_y = 0, 0  # No offset needed for target region
        
    elif method == 'replace_patch':
        # Replace a patch in the target image - simple: place template in top-left corner
        composite = target_array.copy()
        composite[0:template_h, 0:template_w] = template_array
        
        # Template bbox in composite image (normalized)
        template_bbox_xyxy = np.array([
            0.0,  # x0 (normalized) - top-left corner
            0.0,  # y0 (normalized)
            template_w / target_w,  # x1 (normalized)
            template_h / target_h   # y1 (normalized)
        ])
        
        offset_x, offset_y = 0, 0  # No offset needed
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    composite_image = Image.fromarray(composite)
    return composite_image, template_bbox_xyxy, offset_x, offset_y


def convert_bboxes_to_original(bboxes_xyxy, composite_w, composite_h, 
                               original_w, original_h, offset_x=0, offset_y=0):
    """
    Convert bounding boxes from composite image coordinates to original image coordinates.
    
    Args:
        bboxes_xyxy: Bounding boxes in composite image (normalized or pixel coordinates)
        composite_w, composite_h: Size of composite image
        original_w, original_h: Size of original image
        offset_x, offset_y: Offset to subtract
    
    Returns:
        bboxes_original: Bounding boxes in original image coordinates
        valid_mask: Boolean mask indicating which boxes are within original image bounds
    """
    # Convert from normalized to pixel coordinates if needed
    if bboxes_xyxy.max() <= 1.0:
        bboxes_pixel = bboxes_xyxy.copy()
        bboxes_pixel[:, [0, 2]] *= composite_w
        bboxes_pixel[:, [1, 3]] *= composite_h
    else:
        bboxes_pixel = bboxes_xyxy.copy()
    
    # Convert to original image coordinates
    bboxes_original = bboxes_pixel.copy()
    bboxes_original[:, [0, 2]] -= offset_x
    bboxes_original[:, [1, 3]] -= offset_y
    
    # Check which boxes are within original image bounds
    valid_mask = (
        (bboxes_original[:, 0] >= 0) & (bboxes_original[:, 0] < original_w) &
        (bboxes_original[:, 1] >= 0) & (bboxes_original[:, 1] < original_h) &
        (bboxes_original[:, 2] > 0) & (bboxes_original[:, 2] <= original_w) &
        (bboxes_original[:, 3] > 0) & (bboxes_original[:, 3] <= original_h)
    )
    
    # Clip boxes to original image bounds
    bboxes_original[:, [0, 2]] = np.clip(bboxes_original[:, [0, 2]], 0, original_w)
    bboxes_original[:, [1, 3]] = np.clip(bboxes_original[:, [1, 3]], 0, original_h)
    
    return bboxes_original, valid_mask


def process_with_template_embedding(target_image, template_image, model, processor, device,
                                    size_ratio_min=0.7, size_ratio_max=1.3, method='replace_patch'):
    """
    Process an image with template embedding approach.
    
    Args:
        target_image: PIL Image of target
        template_image: PIL Image of template
        model: SAM3 model
        processor: SAM3 processor
        device: Device to use
        size_ratio_min: Minimum size ratio (default 0.7)
        size_ratio_max: Maximum size ratio (default 1.3)
        method: Embedding method ('replace_patch', 'extend_right', 'extend_bottom')
    
    Returns:
        masks: Filtered masks
        boxes: Filtered boxes (in original image coordinates)
        scores: Filtered scores
        first_pass_boxes: All boxes from first pass (for refinement)
        first_pass_scores: All scores from first pass (for refinement)
    """
    original_w, original_h = target_image.size
    template_w, template_h = template_image.size
    
    # Create composite image
    composite_image, template_bbox_xyxy, offset_x, offset_y = create_composite_image(
        target_image, template_image, method=method, template_scale=1.0
    )
    composite_w, composite_h = composite_image.size
    
    # Process composite image
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
    
    # Extract results
    masks_composite = output_state["masks"]
    boxes_composite = output_state["boxes"]
    scores = output_state["scores"]
    
    # Filter by size
    masks_filtered, boxes_filtered, scores_filtered = filter_by_size(
        masks_composite, boxes_composite, scores,
        template_w, template_h, size_ratio_min, size_ratio_max,
        method, composite_w, composite_h, original_w, original_h, offset_x, offset_y
    )
    
    return masks_filtered, boxes_filtered, scores_filtered, boxes_composite, scores


def filter_by_size(masks, boxes, scores, template_w, template_h, 
                   size_ratio_min, size_ratio_max, method='replace_patch',
                   composite_w=None, composite_h=None, original_w=None, original_h=None,
                   offset_x=0, offset_y=0, aspect_ratio_min=0.9, aspect_ratio_max=1.1):
    """
    Filter detections by size (width and height ratios) and aspect ratio.
    
    Args:
        aspect_ratio_min: Minimum aspect ratio ratio (default 0.9)
        aspect_ratio_max: Maximum aspect ratio ratio (default 1.1)
    
    Returns:
        masks_filtered, boxes_filtered, scores_filtered
    """
    # Convert boxes to numpy
    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)
    
    # Handle coordinate conversion for different methods
    if method == 'replace_patch':
        boxes_candidates = boxes_np
        all_indices = np.arange(len(boxes_np))
    else:
        boxes_original, valid_mask = convert_bboxes_to_original(
            boxes_np, composite_w, composite_h,
            original_w, original_h, offset_x, offset_y
        )
        all_indices = np.where(valid_mask)[0]
        boxes_candidates = boxes_original[all_indices]
    
    if len(all_indices) == 0:
        return [], np.array([]), np.array([])
    
    # Extract masks and scores
    masks_candidates = masks[all_indices] if torch.is_tensor(masks) else [masks[i] for i in all_indices]
    scores_candidates = scores[all_indices] if torch.is_tensor(scores) else scores[all_indices]
    
    # Filter by size
    detection_widths = boxes_candidates[:, 2] - boxes_candidates[:, 0]
    detection_heights = boxes_candidates[:, 3] - boxes_candidates[:, 1]
    width_ratios = detection_widths / template_w
    height_ratios = detection_heights / template_h
    
    width_mask = (width_ratios >= size_ratio_min) & (width_ratios <= size_ratio_max)
    height_mask = (height_ratios >= size_ratio_min) & (height_ratios <= size_ratio_max)
    size_mask = width_mask & height_mask
    size_filtered_indices = np.where(size_mask)[0]
    
    if len(size_filtered_indices) == 0:
        return [], np.array([]), np.array([])
    
    # Filter by aspect ratio
    template_aspect_ratio = template_w / template_h
    detection_aspect_ratios = detection_widths[size_filtered_indices] / detection_heights[size_filtered_indices]
    aspect_ratio_ratios = detection_aspect_ratios / template_aspect_ratio
    
    aspect_mask = (aspect_ratio_ratios >= aspect_ratio_min) & (aspect_ratio_ratios <= aspect_ratio_max)
    final_filtered_indices = size_filtered_indices[np.where(aspect_mask)[0]]
    
    if len(size_filtered_indices) > 0:
        print(f"After size filtering: {len(size_filtered_indices)}/{len(all_indices)} detections remain")
        if len(final_filtered_indices) < len(size_filtered_indices):
            print(f"After aspect ratio filtering ({aspect_ratio_min:.1f}x-{aspect_ratio_max:.1f}x): {len(final_filtered_indices)}/{len(size_filtered_indices)} detections remain")
    
    if len(final_filtered_indices) == 0:
        return [], np.array([]), np.array([])
    
    # Apply filter
    masks_filtered = masks_candidates[final_filtered_indices] if torch.is_tensor(masks_candidates) else [masks_candidates[i] for i in final_filtered_indices]
    boxes_filtered = boxes_candidates[final_filtered_indices]
    scores_filtered = scores_candidates[final_filtered_indices] if torch.is_tensor(scores_candidates) else scores_candidates[final_filtered_indices]
    
    return masks_filtered, boxes_filtered, scores_filtered


def is_box_in_embedded_region(box_xyxy, template_w, template_h, overlap_threshold=0.5):
    """
    Check if a bounding box overlaps significantly with the embedded template region.
    The embedded template is at [0, 0, template_w, template_h] in the top-left corner.
    
    Args:
        box_xyxy: Bounding box in [x0, y0, x1, y1] format
        template_w, template_h: Template dimensions
        overlap_threshold: Minimum IoU/overlap ratio to consider as "in embedded region"
    
    Returns:
        bool: True if box overlaps significantly with embedded region
    """
    x0, y0, x1, y1 = box_xyxy
    embedded_region = np.array([0, 0, template_w, template_h])
    
    # Calculate intersection
    inter_x0 = max(x0, embedded_region[0])
    inter_y0 = max(y0, embedded_region[1])
    inter_x1 = min(x1, embedded_region[2])
    inter_y1 = min(y1, embedded_region[3])
    
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return False  # No overlap
    
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    box_area = (x1 - x0) * (y1 - y0)
    
    # Check if intersection is significant relative to box area
    overlap_ratio = inter_area / box_area if box_area > 0 else 0
    
    return overlap_ratio >= overlap_threshold


def refine_with_detections(target_image, boxes_prompt, scores_prompt, model, processor, device,
                          template_w, template_h, size_ratio_min=0.7, size_ratio_max=1.3,
                          num_prompts=2):
    """
    Refine detections by using first pass results as prompts on original image.
    Excludes bounding boxes that overlap with the embedded template region.
    
    Args:
        target_image: PIL Image of original target
        boxes_prompt: Boxes from first pass to use as prompts
        scores_prompt: Scores from first pass
        model: SAM3 model
        processor: SAM3 processor
        device: Device
        template_w, template_h: Template dimensions
        size_ratio_min, size_ratio_max: Size filtering parameters
        num_prompts: Number of top detections to use as prompts
    
    Returns:
        masks_refined, boxes_refined, scores_refined
    """
    original_w, original_h = target_image.size
    
    # Convert boxes to numpy if needed
    if torch.is_tensor(boxes_prompt):
        boxes_np = boxes_prompt.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes_prompt)
    
    if torch.is_tensor(scores_prompt):
        scores_np = scores_prompt.detach().cpu().numpy()
    else:
        scores_np = np.array(scores_prompt)
    
    if len(scores_np) == 0:
        return [], np.array([]), np.array([])
    
    # Filter out boxes that overlap with embedded template region
    valid_indices = []
    for i, box in enumerate(boxes_np):
        if not is_box_in_embedded_region(box, template_w, template_h):
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("Warning: All detections overlap with embedded template region. Skipping second pass.")
        return [], np.array([]), np.array([])
    
    print(f"Filtering out {len(boxes_np) - len(valid_indices)} detection(s) in embedded template region")
    print(f"Using {len(valid_indices)} valid detection(s) for second pass")
    
    # Get top N detections from valid boxes
    valid_scores = scores_np[valid_indices]
    valid_boxes = boxes_np[valid_indices]
    
    num_prompts = min(num_prompts, len(valid_scores))
    top_indices = np.argsort(valid_scores)[-num_prompts:][::-1]
    boxes_for_prompt = valid_boxes[top_indices]
    
    # Process original image
    original_state = processor.set_image(target_image)
    text_outputs = model.backbone.forward_text(["visual"], device=device)
    original_state["backbone_out"].update(text_outputs)
    
    # Add box prompts
    for box_xyxy in boxes_for_prompt:
        x0, y0, x1, y1 = box_xyxy
        cx = (x0 + x1) / 2 / original_w
        cy = (y0 + y1) / 2 / original_h
        w = (x1 - x0) / original_w
        h = (y1 - y0) / original_h
        box_prompt_normalized = [cx, cy, w, h]
        
        original_state = processor.add_geometric_prompt(
            box=box_prompt_normalized,
            label=True,
            state=original_state
        )
    
    # Extract results
    masks_second = original_state["masks"]
    boxes_second = original_state["boxes"]
    scores_second = original_state["scores"]
    
    # Filter by size
    masks_refined, boxes_refined, scores_refined = filter_by_size(
        masks_second, boxes_second, scores_second,
        template_w, template_h, size_ratio_min, size_ratio_max,
        method='replace_patch'  # Original image, no conversion needed
    )
    
    return masks_refined, boxes_refined, scores_refined


def process_image_with_template(target_image, template_image, model, processor, device,
                                size_ratio_min=0.7, size_ratio_max=1.3, 
                                use_refinement=True, num_refinement_prompts=2,
                                output_path=None, image_path=None):
    """
    Complete pipeline: Process an image with template embedding and optional refinement.
    
    Args:
        target_image: PIL Image of target
        template_image: PIL Image of template
        model: SAM3 model
        processor: SAM3 processor
        device: Device to use
        size_ratio_min: Minimum size ratio (default 0.7)
        size_ratio_max: Maximum size ratio (default 1.3)
        use_refinement: Whether to use second pass refinement (default True)
        num_refinement_prompts: Number of top detections to use as prompts (default 2)
        output_path: Optional path to save visualization
        image_path: Optional path to image (for visualization)
    
    Returns:
        masks_final, boxes_final, scores_final
    """
    template_w, template_h = template_image.size
    
    # First pass: Template embedding
    masks_first, boxes_first, scores_first, boxes_all, scores_all = process_with_template_embedding(
        target_image, template_image, model, processor, device,
        size_ratio_min, size_ratio_max, method='replace_patch'
    )
    
    print(f"First pass: {len(boxes_first)} shape(s) matching size criteria")
    
    if len(boxes_first) == 0:
        print("No detections matching size criteria")
        return [], np.array([]), np.array([])
    
    # Second pass: Refinement (optional)
    if use_refinement:
        print(f"\n--- Second Pass: Refinement on Original Image ---")
        masks_refined, boxes_refined, scores_refined = refine_with_detections(
            target_image, boxes_all, scores_all, model, processor, device,
            template_w, template_h, size_ratio_min, size_ratio_max, num_refinement_prompts
        )
        
        if len(boxes_refined) > 0:
            print(f"Second pass: {len(boxes_refined)} shape(s) found")
            masks_final = masks_refined
            boxes_final = boxes_refined
            scores_final = scores_refined
        else:
            print("No detections in second pass, using first pass results")
            masks_final = masks_first
            boxes_final = boxes_first
            scores_final = scores_first
    else:
        masks_final = masks_first
        boxes_final = boxes_first
        scores_final = scores_first
    
    # Convert masks to tensor format for visualization
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
    
    print(f"\nFinal: {len(boxes_final)} shape(s)")
    print(f"Top detections:")
    for i, (box, score) in enumerate(zip(boxes_final, scores_final)):
        score_val = score.item() if torch.is_tensor(score) else score
        print(f"  Detection {i+1}: score={score_val:.3f}, box={box}")
    
    # Visualize if output path provided
    if output_path and image_path:
        visualize_segmentation(
            image_path,
            masks_binary,
            boxes_tensor,
            scores_final,
            output_path=output_path,
            title_suffix=" (Embedded Template Method)" + (" - Refined" if use_refinement else "")
        )
    elif output_path:
        # If no image_path, use target_image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            target_image.save(tmp.name)
            visualize_segmentation(
                tmp.name,
                masks_binary,
                boxes_tensor,
                scores_final,
                output_path=output_path,
                title_suffix=" (Embedded Template Method)" + (" - Refined" if use_refinement else "")
            )
            os.unlink(tmp.name)
    
    return masks_binary, boxes_tensor, scores_final


def main():
    """Main function to run SAM3 segmentation with embedded template."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.png")
    image_path = os.path.join(script_dir, "ellipse.png")
    output_path = os.path.join(script_dir, "ellipse_segmentation_embedded.png")
    
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
    
    template_w, template_h = template_image.size
    print(f"Original image size: {target_image.size[0]}x{target_image.size[1]}")
    print(f"Template size: {template_w}x{template_h}")
    
    # Load the model
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # Process with full pipeline (including second pass)
    size_ratio_min = 0.7
    size_ratio_max = 1.3
    use_refinement = True
    num_refinement_prompts = 2
    
    print(f"\nProcessing parameters:")
    print(f"  Size ratio range: {size_ratio_min:.1f}x - {size_ratio_max:.1f}x (width & height)")
    print(f"  Two-pass refinement: {'Enabled' if use_refinement else 'Disabled'}")
    print(f"  Number of refinement prompts: {num_refinement_prompts}")
    
    masks, boxes, scores = process_image_with_template(
        target_image, template_image, model, processor, device,
        size_ratio_min, size_ratio_max, use_refinement, num_refinement_prompts,
        output_path=output_path, image_path=image_path
    )
    
    if len(boxes) == 0:
        print("\nNo detections found - showing original image")
        img = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title(f'SAM3 Template Matching\nNo shapes detected matching size criteria ({size_ratio_min:.1f}x-{size_ratio_max:.1f}x template)', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
