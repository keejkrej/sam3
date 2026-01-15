#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""
Script to demonstrate cross-image prompt reuse in SAM3.
Extracts a prompt from one image and applies it to another.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from importlib.resources import files
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox, plot_results, plot_bbox


def load_mask(mask_path: str) -> torch.Tensor:
    """Load a binary mask image and convert to tensor.

    Args:
        mask_path: Path to mask image (PNG, JPEG, etc.)
                  White pixels (>128) indicate the object.

    Returns:
        Binary mask tensor of shape (H, W) with values 0 or 1.
    """
    mask_img = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask_np = np.array(mask_img)
    # Convert to binary: threshold at 128
    mask_binary = (mask_np > 128).astype(np.float32)
    return torch.from_numpy(mask_binary)


def plot_mask_prompt(mask, ax=None, color=[1, 1, 0], alpha=0.4):
    """Draw the input mask prompt with semi-transparent overlay and contour outline."""
    if ax is None:
        ax = plt.gca()
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    # Create RGBA overlay
    h, w = mask_np.shape
    mask_overlay = np.zeros((h, w, 4))
    mask_overlay[..., :3] = color
    mask_overlay[..., 3] = mask_np * alpha
    ax.imshow(mask_overlay)

    # Draw contour outline (like dashed box for box prompts)
    ax.contour(mask_np, levels=[0.5], colors=["yellow"], linewidths=2, linestyles="dashed")

    # Add "PROMPT" label at mask centroid
    if mask_np.sum() > 0:
        y_coords, x_coords = np.where(mask_np > 0.5)
        cx, cy = x_coords.mean(), y_coords.min() - 10  # Above the mask
        ax.text(cx, cy, "PROMPT", color="yellow", fontsize=12, fontweight="bold",
                ha="center", va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))


def main():
    parser = argparse.ArgumentParser(description="SAM3 Cross-Image Prompt Reuse")
    parser.add_argument("--img1", type=str, required=True, help="Path to the first image (to extract prompt from)")
    parser.add_argument("--img2", type=str, required=True, help="Path to the second image (to apply prompt to)")

    # Mutually exclusive group for box vs mask
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--box", type=float, nargs=4, help="Bounding box in [x, y, w, h] format for img1")
    prompt_group.add_argument("--mask", type=str, help="Path to binary mask image (white=object) for img1")

    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for segmentation (default: 0.5)")
    parser.add_argument("--output1", type=str, default="image1_segmentation.png", help="Path to save image 1 segmentation")
    parser.add_argument("--output2", type=str, default="image2_segmentation.png", help="Path to save image 2 segmentation")
    parser.add_argument("--pool-only", action="store_true", help="Use only ROI-pooled features (only applies to box prompts)")

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

        # 1. Encode prompt on image 1
        state1 = processor.set_image(image1)

        if args.mask:
            print(f"Extracting prompt from {args.img1} with mask {args.mask}...")

            # Load and preprocess mask
            mask = load_mask(args.mask)
            mask_h, mask_w = mask.shape

            # Resize mask to match image dimensions if needed
            if mask_h != h1 or mask_w != w1:
                mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((w1, h1), Image.NEAREST)
                mask = torch.from_numpy(np.array(mask_pil) / 255.0).float()

            # Resize to processor resolution
            mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
            mask_resized = mask_pil.resize((processor.resolution, processor.resolution), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float()

            state1 = processor._add_mask_prompt(mask=mask_tensor, label=True, state=state1)
            prompt_type = "mask"
        else:
            print(f"Extracting prompt from {args.img1} with box {args.box}...")

            box_xywh = torch.tensor(args.box).view(-1, 4)
            box_cxcywh = box_xywh_to_cxcywh(box_xywh)
            norm_box = normalize_bbox(box_cxcywh, w1, h1).flatten().tolist()

            state1 = processor._add_box_prompt(box=norm_box, label=True, state=state1, pool_only=args.pool_only)
            prompt_type = "box"

        # Run inference on image 1 to see what's detected and save plot
        state1_inference = processor._forward_grounding(state1.copy())
        print(f"Detected {len(state1_inference.get('masks', []))} objects in image 1 with this prompt")

        plt.figure(figsize=(10, 10))
        plot_results(image1, state1_inference)

        # Draw the prompt visualization
        if args.mask:
            plot_mask_prompt(mask)
            plt.title(f"Mask Prompt on Source Image: {os.path.basename(args.img1)}")
        else:
            plot_bbox(h1, w1, args.box, box_format="XYWH", color="yellow", linestyle="dashed", text="PROMPT", relative_coords=False)
            plt.title(f"Box Prompt on Source Image: {os.path.basename(args.img1)}")

        plt.axis("off")
        plt.savefig(args.output1, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved image 1 segmentation to {args.output1}")

        # 2. Capture embeddings
        saved_prompt = state1["prompt"]
        saved_prompt_mask = state1["prompt_mask"]

        print(f"Applying extracted {prompt_type} prompt to {args.img2}...")

        # 3. Apply to image 2
        state2 = processor.set_image(image2)
        state2["prompt"] = saved_prompt
        state2["prompt_mask"] = saved_prompt_mask

        # 4. Run inference
        state2 = processor._forward_grounding(state2)

        # 5. Visualize
        plt.figure(figsize=(10, 10))
        plot_results(image2, state2)
        plt.title(f"Reused {prompt_type.title()} Prompt from {os.path.basename(args.img1)} on {os.path.basename(args.img2)}")
        plt.axis("off")
        plt.savefig(args.output2, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"Success! Results saved to {args.output2}")

if __name__ == "__main__":
    main()
