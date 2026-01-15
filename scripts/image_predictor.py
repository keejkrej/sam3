#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
"""
Copy of the sam3_image_predictor_example.ipynb adapted to test the latest changes.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from importlib.resources import files

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

import torch

# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


def text_prompt_example(processor, image):
    """Example using a text prompt to segment objects."""
    print("\n=== Text Prompt Example ===")
    
    # Text prompt
    inference_state = processor.set_text_prompt(prompt="shoe", state=processor.set_image(image))
    print("found object(s)")
    print(f"found {len(inference_state['masks'])} objects")
    
    # Plot segmentation results for text prompt
    plt.figure(figsize=(10, 10))
    plot_results(image, inference_state)
    plt.title("Text Prompt: 'shoe' Segmentation")
    plt.axis("off")
    plt.savefig("text_prompt_segmentation.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved text prompt results to text_prompt_segmentation.png")
    

def single_box_example(processor, image, width, height):
    """Example using a single bounding box prompt."""
    print("\n=== Single Box Example ===")
    
    inference_state = processor.set_image(image)
    
    # Here the box is in  (x,y,w,h) format, where (x,y) is the top left corner.
    box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)

    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    print("Normalized box input:", norm_box_cxcywh)

    inference_state = processor.add_geometric_prompt(
        state=inference_state, box=norm_box_cxcywh, label=True
    )
    
    # Plot segmentation results
    plt.figure(figsize=(10, 10))
    plot_results(image, inference_state)
    plt.title("Box Prompt Segmentation")
    plt.axis("off")
    plt.savefig("box_prompt_segmentation.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved box prompt segmentation results to box_prompt_segmentation.png")


def single_box_reuse_example(processor, image, width, height):
    """Example showing how to reuse extracted prompt and prompt_mask on modified images."""
    print("\n=== Single Box Reuse (Various Modifications) Example ===")
    
    # First, create and encode a prompt on the original image
    inference_state = processor.set_image(image)
    box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
    
    # Add the prompt and extract the encoded features
    inference_state = processor._add_box_prompt(
        box=norm_box_cxcywh, label=True, state=inference_state
    )
    
    # Extract the encoded prompt and mask for reuse
    saved_prompt = inference_state["prompt"]
    saved_prompt_mask = inference_state["prompt_mask"]
    print(f"Extracted prompt shape: {saved_prompt.shape}")
    
    # Define various image modifications
    modifications = {
        "flipped": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        "cropped": lambda img: img.crop((100, 100, 600, 600)),
        "resized": lambda img: img.resize((width // 2, height // 2)),
        "grayscale": lambda img: img.convert("L").convert("RGB"), # Convert to L then back to RGB to maintain 3 channels
    }
    
    for name, transform in modifications.items():
        print(f"Testing reuse with {name} image...")
        modified_image = transform(image)
        
        # Create a fresh state with the modified image and reuse the saved prompt
        fresh_inference_state = processor.set_image(modified_image)
        fresh_inference_state["prompt"] = saved_prompt
        fresh_inference_state["prompt_mask"] = saved_prompt_mask
        
        # Run inference with reused prompt
        fresh_inference_state = processor._forward_grounding(fresh_inference_state)
        
        # Plot segmentation results
        plt.figure(figsize=(10, 10))
        plot_results(modified_image, fresh_inference_state)
        plt.title(f"Box Prompt Reused on {name.capitalize()} Image")
        plt.axis("off")
        filename = f"box_prompt_reused_{name}_segmentation.png"
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved {name} reused prompt segmentation results to {filename}")


def cross_image_reuse_example(processor, image1, image2, width1, height1):
    """Example showing how to reuse extracted prompt from one image on a completely different image."""
    print("\n=== Cross-Image Prompt Reuse Example ===")
    
    # First, create and encode a prompt on the first image
    inference_state1 = processor.set_image(image1)
    box_input_xywh = torch.tensor([480.0, 290.0, 110.0, 360.0]).view(-1, 4)
    box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
    norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width1, height1).flatten().tolist()
    
    # Add the prompt and extract the encoded features
    inference_state1 = processor._add_box_prompt(
        box=norm_box_cxcywh, label=True, state=inference_state1
    )
    
    # Extract the encoded prompt and mask for reuse
    saved_prompt = inference_state1["prompt"]
    saved_prompt_mask = inference_state1["prompt_mask"]
    print(f"Extracted prompt from image 1, shape: {saved_prompt.shape}")
    
    # Create a fresh state with the second image and reuse the saved prompt
    print("Applying prompt to image 2...")
    inference_state2 = processor.set_image(image2)
    inference_state2["prompt"] = saved_prompt
    inference_state2["prompt_mask"] = saved_prompt_mask
    
    # Run inference on image 2 with reused prompt
    inference_state2 = processor._forward_grounding(inference_state2)
    
    # Plot segmentation results on image 2
    plt.figure(figsize=(10, 10))
    plot_results(image2, inference_state2)
    plt.title("Box Prompt Reused from Image 1 on Image 2")
    plt.axis("off")
    plt.savefig("box_prompt_cross_image_segmentation.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved cross-image reused prompt segmentation results to box_prompt_cross_image_segmentation.png")


def multiple_boxes_example(processor, image, width, height):
    """Example using multiple bounding boxes with different labels."""
    print("\n=== Multiple Boxes Example ===")
    
    inference_state = processor.set_image(image)
    
    box_input_xywh_multiple = [[480.0, 290.0, 110.0, 360.0], [370.0, 280.0, 115.0, 375.0]]
    box_input_cxcywh_multiple = box_xywh_to_cxcywh(torch.tensor(box_input_xywh_multiple).view(-1,4))
    norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh_multiple, width, height).tolist()

    box_labels = [True, False]

    processor.reset_all_prompts(inference_state)

    for box, label in zip(norm_boxes_cxcywh, box_labels):
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=box, label=label
        )

    # Plot segmentation results for multiple boxes
    img0 = Image.open(f"{sam3_root}/assets/images/test_image.jpg")
    plt.figure(figsize=(10, 10))
    plot_results(img0, inference_state)
    plt.title("Multiple Boxes Segmentation")
    plt.axis("off")
    plt.savefig("multiple_boxes_segmentation.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved multiple boxes segmentation to multiple_boxes_segmentation.png")


def main():
    """Main function to run all SAM3 inference examples."""
    # Build Model - use importlib.resources to find BPE file in the correct location
    bpe_path = str(files("sam3").joinpath("assets/bpe_simple_vocab_16e6.txt.gz"))
    model = build_sam3_image_model(bpe_path=bpe_path)

    # Load images
    image_path = f"{sam3_root}/assets/images/test_image.jpg"
    image = Image.open(image_path)
    width, height = image.size
    
    image2_path = f"{sam3_root}/assets/images/test_image_2.jpg"
    image2 = Image.open(image2_path)
    
    # Create processor
    processor = Sam3Processor(model, confidence_threshold=0.5)
    
    # Run all examples
    text_prompt_example(processor, image)
    single_box_example(processor, image, width, height)
    single_box_reuse_example(processor, image, width, height)
    cross_image_reuse_example(processor, image, image2, width, height)
    multiple_boxes_example(processor, image, width, height)
    
    print("\nInference test completed successfully!")
    print("Results saved to:")
    print("- text_prompt_segmentation.png")
    print("- box_prompt_segmentation.png")
    print("- box_prompt_reused_flipped_segmentation.png")
    print("- box_prompt_reused_cropped_segmentation.png")
    print("- box_prompt_reused_resized_segmentation.png")
    print("- box_prompt_reused_grayscale_segmentation.png")
    print("- box_prompt_cross_image_segmentation.png")
    print("- multiple_boxes_segmentation.png")


if __name__ == "__main__":
    main()
