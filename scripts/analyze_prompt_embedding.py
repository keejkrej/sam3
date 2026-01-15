#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""
Script to analyze prompt embeddings across different image duplication layouts.
Uses Euclidean distance analysis to compare how prompt embeddings differ for 1x1, 2x1, 1x2, and 2x2 layouts.
"""

import argparse

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


def create_layout(img: Image.Image, layout: str) -> Image.Image:
    """Create duplicated image layouts."""
    w, h = img.size
    cols, rows = map(int, layout.split('x'))
    
    new_img = Image.new("RGB", (w * cols, h * rows))
    
    for row in range(rows):
        for col in range(cols):
            x = col * w
            y = row * h
            new_img.paste(img, (x, y))
    
    return new_img


def extract_prompt_embedding(processor, img: Image.Image, box_xywh: list) -> dict:
    """Extract prompt embedding for a given image and box."""
    w, h = img.size
    
    state = processor.set_image(img)
    
    box_xywh_tensor = torch.tensor(box_xywh).view(-1, 4)
    box_cxcywh = box_xywh_to_cxcywh(box_xywh_tensor)
    norm_box = normalize_bbox(box_cxcywh, w, h).flatten().tolist()
    
    state = processor._add_box_prompt(box=norm_box, label=True, state=state)
    
    # Extract geometry features instead of combined prompt features
    geo_feats = state["geo_feats"]  # (num_geo_tokens, batch_size, d_model)
    geo_masks = state["geo_masks"]  # (batch_size, num_geo_tokens)
    
    # Get geometry tokens (remove batch dimension)
    geo_tokens = geo_feats[:, 0, :]  # (num_geo_tokens, d_model)
    
    # Also keep the original prompt for comparison
    prompt = state["prompt"]  # (seq_len, batch_size, d_model)
    prompt_mask = state["prompt_mask"]  # (batch_size, seq_len)
    
    # Extract valid (non-masked) tokens for reference
    valid_mask = ~prompt_mask[0]  # (seq_len,)
    valid_tokens = prompt[:, 0, :][valid_mask]  # (num_valid, d_model)
    
    # Run inference to get segmentation results
    state = processor._forward_grounding(state)
    
    return {
        "prompt": prompt.cpu(),
        "prompt_mask": prompt_mask.cpu(),
        "valid_tokens": valid_tokens.cpu(),
        "geo_mask": geo_masks.cpu(),
        "geo_tokens": geo_tokens.cpu(),  # Main feature for analysis
        "image_size": (w, h),
        "masks": state.get("masks", None),
        "boxes": state.get("boxes", None),
        "scores": state.get("scores", None),
        "txt_feats": state["txt_feats"].cpu(),
        "txt_masks": state["txt_masks"].cpu(),
        "geo_feats": state["geo_feats"].cpu(),
        "geo_masks": state["geo_masks"].cpu(),
        "visual_prompt_embed": state["visual_prompt_embed"].cpu(),
        "visual_prompt_mask": state["visual_prompt_mask"].cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt embeddings with Euclidean distance comparison")
    parser.add_argument("--img", type=str, required=True, help="Path to the source image")
    parser.add_argument("--box", type=float, nargs=4, required=True, help="Bounding box [x, y, w, h] - USE: 76 169 466 834 for cat.jpg")
    parser.add_argument("--output", type=str, default="prompt_embedding_analysis.png", help="Output path")
    
    args = parser.parse_args()

    # Hardware setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with torch.autocast(device, dtype=torch.bfloat16):
        # Build Model
        bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model)

        # Load source image
        src_img = Image.open(args.img)
        print(f"Source image size: {src_img.size}")
        print(f"Box (xywh): {args.box}")

        layouts = ["1x1", "2x1", "3x1", "1x2", "1x3", "2x2", "3x2", "2x3", "3x3"]
        results = {}
        
        for layout in layouts:
            print(f"\nProcessing layout: {layout}")
            layout_img = create_layout(src_img, layout)
            print(f"  Layout image size: {layout_img.size}")
            
            # Box coordinates stay the same (top-left object)
            result = extract_prompt_embedding(processor, layout_img, args.box)
            results[layout] = result
            results[layout]["image"] = layout_img
            
            print(f"  Prompt shape: {result['prompt'].shape}")
            mask = result['prompt_mask'].squeeze()
            print(f"  Prompt mask True count: {mask.sum().item()}")
            print(f"  Prompt mask (showing first 10): {mask[:10]}")
            valid_mask = ~mask
            print(f"  Valid mask (showing first 10): {valid_mask[:10]}")
            print(f"  Valid mask True count: {valid_mask.sum().item()}")
            print(f"  Valid tokens: {result['valid_tokens'].shape[0]}")
            print(f"  Valid tokens shape: {result['valid_tokens'].shape}")
            print(f" Embedding dimension (d_model): {result['valid_tokens'].shape[1]}")
            
            # Get the actual valid token indices
            mask = result['prompt_mask'].squeeze()
            geo_mask = result['geo_mask'].squeeze()
            valid_mask = ~mask
            valid_indices = torch.where(valid_mask)[0].tolist()
            
            # Show all token details
            total_seq_len = mask.shape[0]
            print(f"  Total sequence length: {total_seq_len}")
            print(f"  Prompt mask (all {total_seq_len} tokens): {mask.tolist()}")
            print(f"  Valid mask (all {total_seq_len} tokens): {valid_mask.tolist()}")
            print(f"  Valid token indices: {valid_indices}")
            
            # Count tokens by type
            num_valid = valid_mask.sum().item()
            num_masked = mask.sum().item()
            print(f"  Valid (unmasked) tokens: {num_valid}")
            print(f"  Masked tokens: {num_masked}")
            
            # Geo mask details
            geo_valid_count = (~geo_mask).sum().item()
            print(f"  Geometry valid tokens: {geo_valid_count}")
            print(f"  Geometry mask: {geo_mask.tolist()}")
            
            print(f"  Individual token norms: {result['valid_tokens'].norm(dim=-1)}")
            print(f"  Embedding mean norm: {result['valid_tokens'].norm(dim=-1).mean():.4f}")
            
            print(f"  Geo token norms: {result['geo_tokens'].norm(dim=-1)}")
            print(f"  Geo token mean norm: {result['geo_tokens'].norm(dim=-1).mean():.4f}")
            
            # Component feature shapes and breakdown
            txt_shape = result['txt_feats'].shape
            geo_shape = result['geo_feats'].shape
            visual_shape = result['visual_prompt_embed'].shape
            
            print(f"  txt_feats shape: {txt_shape}")
            print(f"  geo_feats shape: {geo_shape}")
            print(f"  visual_prompt_embed shape: {visual_shape}")
            
            # Token breakdown
            txt_tokens = txt_shape[0]
            geo_tokens = geo_shape[0]
            visual_tokens = visual_shape[0]
            
            print(f"  Token breakdown:")
            print(f"    Text tokens: {txt_tokens} (positions 0-{txt_tokens-1})")
            print(f"    Geometry tokens: {geo_tokens} (positions {txt_tokens}-{txt_tokens+geo_tokens-1})")
            print(f"    Visual tokens: {visual_tokens} (positions {txt_tokens+geo_tokens}-{txt_tokens+geo_tokens+visual_tokens-1})")
            print(f"    Total expected: {txt_tokens + geo_tokens + visual_tokens}")
            
            # Verify actual vs expected
            actual_total = result['prompt'].shape[0]
            print(f"    Actual total: {actual_total}")
            if txt_tokens + geo_tokens + visual_tokens != actual_total:
                print(f"    WARNING: Expected {txt_tokens + geo_tokens + visual_tokens} but got {actual_total}")

        # Prepare data for analysis - using geo_tokens instead of valid_tokens
        all_embeddings = []
        all_labels = []
        token_counts = {}
        
        for layout in layouts:
            tokens = results[layout]["geo_tokens"].numpy()  # Use geo_tokens
            all_embeddings.append(tokens)
            all_labels.extend([layout] * len(tokens))
            token_counts[layout] = len(tokens)
        
        all_embeddings = np.vstack(all_embeddings)
        print(f"\nTotal geometry embeddings for analysis: {all_embeddings.shape}")

        # Calculate pairwise Euclidean distances between layouts
        def calculate_token_distances(embeddings1, embeddings2):
            """Calculate Euclidean distance for each token separately."""
            token_distances = []
            for token_idx in range(min(len(embeddings1), len(embeddings2))):
                dist = np.linalg.norm(embeddings1[token_idx] - embeddings2[token_idx])
                token_distances.append(dist)
            return token_distances

        print("\n=== Geometry Token-by-Token Distance Analysis ===")
        
        # Calculate distances between all pairs
        layout_pairs = [
            ("1x1", "2x1"), ("1x1", "3x1"), ("1x1", "1x2"), ("1x1", "1x3"),
            ("1x1", "2x2"), ("1x1", "3x2"), ("1x1", "2x3"), ("1x1", "3x3"),
            ("2x1", "3x1"), ("2x1", "1x2"), ("2x1", "1x3"), ("2x1", "2x2"),
            ("2x1", "3x2"), ("2x1", "2x3"), ("2x1", "3x3"),
            ("3x1", "1x2"), ("3x1", "1x3"), ("3x1", "2x2"), ("3x1", "3x2"),
            ("3x1", "2x3"), ("3x1", "3x3"),
            ("1x2", "1x3"), ("1x2", "2x2"), ("1x2", "3x2"), ("1x2", "2x3"),
            ("1x2", "3x3"),
            ("1x3", "2x2"), ("1x3", "3x2"), ("1x3", "2x3"), ("1x3", "3x3"),
            ("2x2", "3x2"), ("2x2", "2x3"), ("2x2", "3x3"),
            ("3x2", "2x3"), ("3x2", "3x3"),
            ("2x3", "3x3")
        ]
        
        # Calculate distances for each geometry token separately (we have 2 geo tokens)
        token_distance_matrices = []  # One matrix per token
        
        for token_idx in range(2):  # We have 2 geometry tokens
            print(f"\n=== GEOMETRY TOKEN {token_idx} Distance Analysis ===")
            distance_matrix = {}
            
            for layout1, layout2 in layout_pairs:
                emb1 = results[layout1]["geo_tokens"].numpy()
                emb2 = results[layout2]["geo_tokens"].numpy()
                
                dist = np.linalg.norm(emb1[token_idx] - emb2[token_idx])
                distance_matrix[(layout1, layout2)] = dist
                
                print(f"{layout1} <-> {layout2}: {dist:.6f}")
            
            token_distance_matrices.append(distance_matrix)

        # Create distance visualization for each geometry token
        print(f"\n=== Creating geometry token-by-token visualizations ===")
        
        # Create figure with 2 subplots for each geometry token
        fig_tokens = plt.figure(figsize=(15, 6))
        
        for token_idx in range(2):
            ax = fig_tokens.add_subplot(1, 2, token_idx + 1)
            
            # Build matrix for this token
            matrix_data = np.zeros((len(layouts), len(layouts)))
            distance_matrix = token_distance_matrices[token_idx]
            
            for i, layout1 in enumerate(layouts):
                for j, layout2 in enumerate(layouts):
                    if i == j:
                        matrix_data[i, j] = 0
                    elif (layout1, layout2) in distance_matrix:
                        matrix_data[i, j] = distance_matrix[(layout1, layout2)]
                    else:
                        matrix_data[i, j] = distance_matrix[(layout2, layout1)]
            
            # Create heatmap
            im = ax.imshow(matrix_data, cmap='viridis')
            ax.set_xticks(range(len(layouts)))
            ax.set_yticks(range(len(layouts)))
            ax.set_xticklabels(layouts, rotation=45)
            ax.set_yticklabels(layouts)
            ax.set_title(f"Token {token_idx} Distances")
            
            # Add text annotations
            for i in range(len(layouts)):
                for j in range(len(layouts)):
                    text = ax.text(j, i, f'{matrix_data[i, j]:.4f}',
                                  ha="center", va="center", color="white", fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        token_output = args.output.replace(".png", "_tokens.png")
        plt.savefig(token_output, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nToken-by-token visualization saved to {token_output}")
        
        # Print summary of what each geometry token represents
        print(f"\n=== Geometry Token Analysis Summary ===")
        print("Geo Token 0: Box coordinate encoding - may vary with image normalization")
        print("Geo Token 1: Spatial position encoding - varies with duplication pattern")

        # Save segmentation results for each layout using plot_results
        for layout in layouts:
            img = results[layout]["image"]
            w, h = img.size
            state = {
                "masks": results[layout]["masks"],
                "boxes": results[layout]["boxes"],
                "scores": results[layout]["scores"],
            }
            
            plt.figure(figsize=(10, 10))
            plot_results(img, state)
            plot_bbox(h, w, args.box, box_format="XYWH", color="yellow", linestyle="dashed", text="PROMPT", relative_coords=False)
            
            num_det = len(state["masks"]) if state["masks"] is not None else 0
            plt.title(f"{layout} Layout - {num_det} detections")
            plt.axis("off")
            
            layout_output = args.output.replace(".png", f"_{layout}.png")
            plt.savefig(layout_output, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"Saved {layout} segmentation to {layout_output}")
        
        print(f"\nVisualization saved to {args.output}")

        # Print summary statistics
        print("\n=== Summary Statistics ===")
        for layout in layouts:
            tokens = results[layout]["valid_tokens"]
            print(f"{layout}:")
            print(f"  Shape: {tokens.shape}")
            print(f"  Mean norm: {tokens.norm(dim=-1).mean():.4f}")
            print(f"  Std norm: {tokens.norm(dim=-1).std():.4f}")


if __name__ == "__main__":
    main()
