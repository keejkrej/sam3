#!/usr/bin/env python3
"""
Script to segment elliptical shapes in an image using SAM3 with text prompt "oval".
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
    ax.set_title(f'SAM3 Segmentation with Text Prompt "oval"\nFound {len(scores)} elliptical shape(s)', 
                 fontsize=14, fontweight='bold')
    
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
    
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def main():
    """Main function to run SAM3 segmentation with text prompt."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "ellipse.png")
    output_path = os.path.join(script_dir, "ellipse_segmentation.png")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading image from: {image_path}")
    
    # Load the model
    print("Loading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # Load image
    print("Loading image...")
    image = Image.open(image_path).convert('RGB')
    
    # Set image in processor
    print("Setting image in processor...")
    inference_state = processor.set_image(image)
    
    # Set text prompt
    text_prompt = "oval"
    print(f"Setting text prompt: '{text_prompt}'")
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
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
        # Still show the original image
        img = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img)
        ax.set_title('SAM3 Segmentation with Text Prompt "oval"\nNo shapes detected', 
                     fontsize=14, fontweight='bold')
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
    visualize_segmentation(image_path, masks, boxes, scores, output_path=output_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
