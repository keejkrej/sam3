#!/usr/bin/env python3
"""
Script to create an oval template by cropping a high-confidence bounding box from the image.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent directory to path to import sam3 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def crop_box_from_image(image, box_xyxy, padding=10):
    """
    Crop a region from an image using a bounding box in XYXY format.
    
    Args:
        image: PIL Image
        box_xyxy: List or array [x0, y0, x1, y1] in absolute pixel coordinates
        padding: Additional padding in pixels around the box
    
    Returns:
        Cropped PIL Image
    """
    x0, y0, x1, y1 = box_xyxy
    
    # Add padding
    x0 = max(0, int(x0 - padding))
    y0 = max(0, int(y0 - padding))
    x1 = min(image.width, int(x1 + padding))
    y1 = min(image.height, int(y1 + padding))
    
    # Crop the image
    cropped = image.crop((x0, y0, x1, y1))
    return cropped


def main():
    """Main function to create template from high-confidence bounding box."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "ellipse.png")
    template_path = os.path.join(script_dir, "template.png")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"Loading image from: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size
    print(f"Image dimensions: {img_width} x {img_height}")
    
    # Load the model
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # Set image in processor
    print("Setting image in processor...")
    inference_state = processor.set_image(image)
    
    # Set text prompt to find ovals
    text_prompt = "oval"
    print(f"Setting text prompt: '{text_prompt}'")
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    # Extract results
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    # Filter for high confidence results (>= 0.8)
    confidence_threshold = 0.8
    if torch.is_tensor(scores):
        keep_mask = scores >= confidence_threshold
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    else:
        keep_mask = scores >= confidence_threshold
        keep_indices = np.where(keep_mask)[0]
    
    if len(keep_indices) == 0:
        print(f"\nError: No high-confidence results found (>= {confidence_threshold})")
        print("Available scores:", scores.cpu().numpy() if torch.is_tensor(scores) else scores)
        return
    
    # Filter to high confidence results
    if torch.is_tensor(masks):
        masks = masks[keep_indices]
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
    else:
        masks = [masks[i] for i in keep_indices]
        boxes = [boxes[i] for i in keep_indices]
        scores = scores[keep_indices]
    
    print(f"\nFound {len(scores)} high-confidence elliptical shape(s) (>= {confidence_threshold})")
    
    # Use the specified high-confidence box (Shape 8: score=0.915)
    best_box = [775.77277, 1590.7705, 1028.8243, 1730.3026]
    best_score = 0.915
    
    print(f"\nUsing specified high-confidence box:")
    print(f"  Score: {best_score:.3f}")
    print(f"  Box (XYXY): {best_box}")
    
    # Crop the template
    print(f"\nCropping template from image...")
    template = crop_box_from_image(image, best_box, padding=10)
    
    print(f"Template dimensions: {template.width} x {template.height}")
    
    # Save template
    print(f"Saving template to: {template_path}")
    template.save(template_path)
    print("Template saved successfully!")
    
    print(f"\nDone! Template saved to: {template_path}")


if __name__ == "__main__":
    main()
