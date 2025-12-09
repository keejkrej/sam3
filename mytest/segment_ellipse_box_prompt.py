#!/usr/bin/env python3
"""
Script to segment elliptical shapes using a bounding box as geometric prompt (no text prompt).
Uses one of the detected boxes from the previous text-prompt run as a reference.
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


def xyxy_to_cxcywh_normalized(xyxy_box, img_width, img_height):
    """
    Convert bounding box from XYXY format to normalized [center_x, center_y, width, height] format.
    
    Args:
        xyxy_box: List or array [x0, y0, x1, y1] in absolute pixel coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List [center_x, center_y, width, height] normalized to [0, 1]
    """
    x0, y0, x1, y1 = xyxy_box
    
    # Calculate center and dimensions
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    width = x1 - x0
    height = y1 - y0
    
    # Normalize to [0, 1]
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return [center_x_norm, center_y_norm, width_norm, height_norm]


def visualize_segmentation(image_path, masks, boxes, scores, prompt_box=None, output_path=None):
    """
    Visualize segmentation masks overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        masks: Tensor of shape (N, H, W) containing binary masks
        boxes: Tensor of shape (N, 4) containing bounding boxes in XYXY format
        scores: Tensor of shape (N,) containing confidence scores
        prompt_box: Optional prompt box in XYXY format to highlight
        output_path: Optional path to save the visualization
    """
    # Load the original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_array)
    title = f'SAM3 Segmentation with Box Prompt (No Text)\nFound {len(scores)} elliptical shape(s)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Draw prompt box if provided
    if prompt_box is not None:
        x0, y0, x1, y1 = prompt_box
        width = x1 - x0
        height = y1 - y0
        rect = plt.Rectangle(
            (x0, y0), width, height,
            linewidth=3, edgecolor='yellow', facecolor='none', linestyle='--',
            label='Prompt Box'
        )
        ax.add_patch(rect)
        ax.text(
            x0, y0 - 5,
            'PROMPT BOX',
            color='yellow', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
    
    # Generate distinct colors for each mask
    colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
    
    # Overlay each mask
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        # Convert mask to numpy if it's a tensor
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
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
            box_np = box.cpu().numpy()
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
            f'Oval {i+1} (score: {score:.2f})',
            color=color[:3], fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    if prompt_box is not None:
        ax.legend(loc='upper right')
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def main():
    """Main function to run SAM3 segmentation with box prompt."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "ellipse.png")
    output_path = os.path.join(script_dir, "ellipse_segmentation_box_prompt.png")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading image from: {image_path}")
    
    # Load image to get dimensions
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    print(f"Image dimensions: {img_width} x {img_height}")
    
    # Hardcoded bounding box from previous results (Shape 8 with high confidence)
    # Box format: [x0, y0, x1, y1] in absolute pixel coordinates
    # This is Shape 8 from the previous run: score=0.915
    prompt_box_xyxy = [775.77277, 1590.7705, 1028.8243, 1730.3026]
    print(f"\nUsing prompt box (XYXY): {prompt_box_xyxy}")
    
    # Convert to normalized [center_x, center_y, width, height] format
    prompt_box_normalized = xyxy_to_cxcywh_normalized(
        prompt_box_xyxy, img_width, img_height
    )
    print(f"Normalized box (cx, cy, w, h): {prompt_box_normalized}")
    
    # Load the model
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # Set image in processor
    print("Setting image in processor...")
    inference_state = processor.set_image(image)
    
    # Add geometric prompt (box) - no text prompt
    print(f"Adding geometric box prompt (positive)...")
    output = processor.add_geometric_prompt(
        box=prompt_box_normalized,
        label=True,  # True = positive prompt
        state=inference_state
    )
    
    # Extract results
    masks = output["masks"]  # Boolean masks
    boxes = output["boxes"]  # Bounding boxes in XYXY format
    scores = output["scores"]  # Confidence scores
    
    # Filter out low confidence results (< 0.8)
    confidence_threshold = 0.8
    original_count = len(scores)
    
    if torch.is_tensor(scores):
        keep_mask = scores >= confidence_threshold
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    else:
        keep_mask = scores >= confidence_threshold
        keep_indices = np.where(keep_mask)[0]
    
    if len(keep_indices) > 0:
        # Filter masks, boxes, and scores
        if torch.is_tensor(masks):
            masks = masks[keep_indices]
        else:
            masks = [masks[i] for i in keep_indices]
        
        if torch.is_tensor(boxes):
            boxes = boxes[keep_indices]
        else:
            boxes = [boxes[i] for i in keep_indices]
        
        if torch.is_tensor(scores):
            scores = scores[keep_indices]
        else:
            scores = scores[keep_indices]
        
        print(f"\nFiltered Results (confidence >= {confidence_threshold}):")
        print(f"  Found {len(scores)} elliptical shape(s) after filtering (from {original_count} total)")
    else:
        # Set to empty results - preserve tensor types
        if torch.is_tensor(masks) and len(masks) > 0:
            # Create empty tensor with same shape (except first dimension) and dtype/device
            masks = masks[:0]
            boxes = boxes[:0]
            scores = scores[:0]
        elif torch.is_tensor(masks):
            # Already empty, keep as is
            pass
        else:
            masks = []
            boxes = []
            scores = np.array([])
        print(f"\nFiltered Results (confidence >= {confidence_threshold}):")
        print(f"  No shapes found with confidence >= {confidence_threshold} (from {original_count} total)")
    
    print(f"\nSegmentation Results:")
    print(f"  Found {len(scores)} elliptical shape(s)")
    
    if len(scores) == 0:
        print("  Warning: No elliptical shapes detected. Try adjusting the confidence threshold.")
        # Still show the original image with prompt box
        img_array = np.array(image)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_array)
        ax.set_title('SAM3 Segmentation with Box Prompt\nNo shapes detected', 
                     fontsize=14, fontweight='bold')
        
        # Draw prompt box
        x0, y0, x1, y1 = prompt_box_xyxy
        width = x1 - x0
        height = y1 - y0
        rect = plt.Rectangle(
            (x0, y0), width, height,
            linewidth=3, edgecolor='yellow', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)
        ax.axis('off')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.show()
        return
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        print(f"  Shape {i+1}: score={score:.3f}, box={box.cpu().numpy()}")
    
    # Visualize results
    print("\nGenerating visualization...")
    visualize_segmentation(
        image_path, masks, boxes, scores, 
        prompt_box=prompt_box_xyxy,
        output_path=output_path
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
