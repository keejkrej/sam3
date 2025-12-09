#!/usr/bin/env python3
"""
Batch processing script to segment ellipses in multiple images using template matching.
Processes all images in the ellipses/ folder and creates visualizations.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

# Add parent directory to path to import sam3 modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Import functions from embedded script
from segment_with_template_embedded import (
    visualize_segmentation,
    process_image_with_template
)


def process_single_image(image_path, template_image, model, processor, device, output_dir, 
                        size_ratio_min=0.8, size_ratio_max=1.2, use_refinement=True, num_refinement_prompts=2):
    """
    Process a single image using the complete pipeline from segment_with_template_embedded.py
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    try:
        # Load target image
        target_image = Image.open(image_path).convert('RGB')
        template_w, template_h = template_image.size
        
        print(f"Target image size: {target_image.size[0]}x{target_image.size[1]}")
        print(f"Template size: {template_w}x{template_h}")
        
        # Use the complete pipeline from embedded script
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_result.png")
        
        masks, boxes, scores = process_image_with_template(
            target_image, template_image, model, processor, device,
            size_ratio_min, size_ratio_max, use_refinement, num_refinement_prompts,
            output_path=output_path, image_path=image_path
        )
        
        if len(boxes) == 0:
            print("No detections found - saving empty result visualization")
            img = Image.open(image_path).convert('RGB')
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img)
            ax.set_title(f'SAM3 Template Matching\nNo shapes matching size criteria', 
                         fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return True, 0
        
        return True, len(scores)
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def main():
    """Main function to batch process all images in ellipses/ folder."""
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "template.png")
    ellipses_dir = os.path.join(script_dir, "ellipses")
    output_dir = os.path.join(script_dir, "ellipses_results")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if template exists
    if not os.path.exists(template_path):
        print(f"Error: Template image not found at {template_path}")
        return
    
    # Check if ellipses directory exists
    if not os.path.exists(ellipses_dir):
        print(f"Error: Ellipses directory not found at {ellipses_dir}")
        return
    
    # Load template
    print(f"Loading template from: {template_path}")
    template_image = Image.open(template_path).convert('RGB')
    template_w, template_h = template_image.size
    print(f"Template size: {template_w}x{template_h}")
    
    # Find all image files in ellipses directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(ellipses_dir, ext)))
        image_files.extend(glob.glob(os.path.join(ellipses_dir, ext.upper())))
    
    if len(image_files) == 0:
        print(f"No image files found in {ellipses_dir}")
        return
    
    image_files.sort()  # Process in alphabetical order
    print(f"\nFound {len(image_files)} image(s) to process")
    
    # Load the model (once for all images)
    print("\nLoading SAM3 model...")
    model = build_sam3_image_model(device=device, eval_mode=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=0.3)
    print("Model loaded successfully!")
    
    # Process each image
    results = []
    size_ratio_min = 0.8
    size_ratio_max = 1.2
    
    print(f"\nProcessing parameters:")
    print(f"  Size ratio range: {size_ratio_min:.1f}x - {size_ratio_max:.1f}x (width & height)")
    print(f"  No confidence filtering - using all detections")
    print(f"  Two-pass refinement: {'Enabled' if True else 'Disabled'}")
    print(f"  Number of refinement prompts: {2}")
    
    use_refinement = True
    num_refinement_prompts = 2
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        success, num_detections = process_single_image(
            image_path, template_image, model, processor, device, output_dir,
            size_ratio_min, size_ratio_max, use_refinement, num_refinement_prompts
        )
        results.append({
            'image': os.path.basename(image_path),
            'success': success,
            'num_detections': num_detections
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(image_files)}")
    print(f"Successful: {sum(1 for r in results if r['success'])}")
    print(f"Failed: {sum(1 for r in results if not r['success'])}")
    print(f"\nDetections per image:")
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"  {status} {r['image']}: {r['num_detections']} detection(s)")
    
    total_detections = sum(r['num_detections'] for r in results)
    print(f"\nTotal detections across all images: {total_detections}")
    print(f"\nResults saved to: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
