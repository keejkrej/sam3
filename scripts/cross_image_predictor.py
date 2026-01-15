#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""
Script to demonstrate cross-image prompt reuse in SAM3.
Extracts a prompt from one image and applies it to another.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
from importlib.resources import files
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, plot_results, plot_bbox

def main():
    parser = argparse.ArgumentParser(description="SAM3 Cross-Image Prompt Reuse")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first image (to extract prompt from)")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second image (to apply prompt to)")
    parser.add_argument("--box", type=float, nargs=4, required=True, help="Bounding box in [x, y, w, h] format for img1")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for segmentation (default: 0.5)")
    parser.add_argument("--output1", type=str, default="image1_segmentation.png", help="Path to save image 1 segmentation")
    parser.add_argument("--output2", type=str, default="image2_segmentation.png", help="Path to save image 2 segmentation")
    parser.add_argument("--pool-only", action="store_true", help="Use only ROI-pooled features, skip coordinate-dependent encodings")
    
    args = parser.parse_args()

    # Hardware setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.autocast(device, dtype=torch.bfloat16):
        # Build Model
        bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)

        # Load images
        image1 = Image.open(args.img1)
        image2 = Image.open(args.img2)
        w1, h1 = image1.size

        print(f"Extracting prompt from {args.img1} with box {args.box}...")
        
        # 1. Encode prompt on image 1
        state1 = processor.set_image(image1)
        box_xywh = torch.tensor(args.box).view(-1, 4)
        box_cxcywh = box_xywh_to_cxcywh(box_xywh)
        norm_box = normalize_bbox(box_cxcywh, w1, h1).flatten().tolist()
        
        state1 = processor._add_box_prompt(box=norm_box, label=True, state=state1, pool_only=args.pool_only)
        
        # Run inference on image 1 to see what's detected and save plot
        state1_inference = processor._forward_grounding(state1.copy())
        print(f"Detected {len(state1_inference.get('masks', []))} objects in image 1 with this prompt")
        
        plt.figure(figsize=(10, 10))
        plot_results(image1, state1_inference)
        # Draw the prompted bounding box for verification
        plot_bbox(h1, w1, args.box, box_format="XYWH", color="yellow", linestyle="dashed", text="PROMPT", relative_coords=False)
        plt.title(f"Prompt on Source Image: {os.path.basename(args.img1)}")
        plt.axis("off")
        plt.savefig(args.output1, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved image 1 segmentation to {args.output1}")

        # 2. Capture embeddings
        saved_prompt = state1["prompt"]
        saved_prompt_mask = state1["prompt_mask"]
        
        print(f"Applying extracted prompt to {args.img2}...")
        
        # 3. Apply to image 2
        state2 = processor.set_image(image2)
        state2["prompt"] = saved_prompt
        state2["prompt_mask"] = saved_prompt_mask
        
        # 4. Run inference
        state2 = processor._forward_grounding(state2)
        
        # 5. Visualize
        plt.figure(figsize=(10, 10))
        plot_results(image2, state2)
        plt.title(f"Reused Prompt from {os.path.basename(args.img1)} on {os.path.basename(args.img2)}")
        plt.axis("off")
        plt.savefig(args.output2, bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"Success! Results saved to {args.output2}")

if __name__ == "__main__":
    main()
