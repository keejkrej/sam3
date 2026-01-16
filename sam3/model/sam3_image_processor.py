# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state=None):
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def _add_text_prompt(self, prompt: str, state: Dict, pool_only: bool = False):
        """Adds a text prompt and encodes it without running inference."""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before _add_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        
        # Create dummy geometric prompt and encode
        geometric_prompt = self.model._get_dummy_prompt()
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            encoded = self.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=geometric_prompt,
                pool_only=pool_only,
            )
        
        state["prompt"] = encoded["prompt"]
        state["prompt_mask"] = encoded["prompt_mask"]
        state["txt_feats"] = encoded["txt_feats"]
        state["txt_masks"] = encoded["txt_masks"]
        state["geo_feats"] = encoded["geo_feats"]
        state["geo_masks"] = encoded["geo_masks"]
        state["visual_prompt_embed"] = encoded["visual_prompt_embed"]
        state["visual_prompt_mask"] = encoded["visual_prompt_mask"]
        state["backbone_out"] = encoded["backbone_out"]

        return state

    @torch.inference_mode()
    def _add_box_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and encodes it without running inference."""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before _add_box_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        # Initialize geometric prompt if needed
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        # Encode prompts immediately
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            encoded = self.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"],
            )
        
        # Store encoded prompts directly in state
        state["prompt"] = encoded["prompt"]
        state["prompt_mask"] = encoded["prompt_mask"]
        state["txt_feats"] = encoded["txt_feats"]
        state["txt_masks"] = encoded["txt_masks"]
        state["geo_feats"] = encoded["geo_feats"]
        state["geo_masks"] = encoded["geo_masks"]
        state["visual_prompt_embed"] = encoded["visual_prompt_embed"]
        state["visual_prompt_mask"] = encoded["visual_prompt_mask"]
        state["backbone_out"] = encoded["backbone_out"]
        
        # Remove geometric_prompt as we only rely on prompt and prompt_mask for inference
        if "geometric_prompt" in state:
            del state["geometric_prompt"]

        return state

    @torch.inference_mode()
    def _add_mask_prompt(self, mask: torch.Tensor, label: bool, state: Dict):
        """Adds a mask prompt and encodes it without running inference.

        Args:
            mask: Binary mask tensor of shape (H, W) where 1/True indicates the object.
            label: Whether this is a positive (True) or negative (False) prompt.
            state: The processor state dictionary containing backbone_out.

        Returns:
            Updated state dictionary with encoded prompt.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before _add_mask_prompt")

        if "language_features" not in state["backbone_out"]:
            # Set dummy text prompt for visual-only mode
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        # Initialize geometric prompt if needed
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Reshape mask to expected format: N_masks x B x 1 x H x W
        masks = mask.to(device=self.device, dtype=torch.float32).view(1, 1, 1, *mask.shape[-2:])
        labels = torch.tensor([label], device=self.device, dtype=torch.long).view(1, 1)
        state["geometric_prompt"].append_masks(masks, labels)

        # Encode prompts (pool_only not applicable for masks)
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            encoded = self.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"],
                pool_only=False,
            )

        # Store encoded prompts directly in state
        state["prompt"] = encoded["prompt"]
        state["prompt_mask"] = encoded["prompt_mask"]
        state["txt_feats"] = encoded["txt_feats"]
        state["txt_masks"] = encoded["txt_masks"]
        state["geo_feats"] = encoded["geo_feats"]
        state["geo_masks"] = encoded["geo_masks"]
        state["visual_prompt_embed"] = encoded["visual_prompt_embed"]
        state["visual_prompt_mask"] = encoded["visual_prompt_mask"]
        state["backbone_out"] = encoded["backbone_out"]

        # Remove geometric_prompt as we only rely on prompt and prompt_mask for inference
        if "geometric_prompt" in state:
            del state["geometric_prompt"]

        return state

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = [
            "geometric_prompt", "prompt", "prompt_mask", "boxes", "masks", "masks_logits", "scores",
            "txt_feats", "txt_masks", "geo_feats", "geo_masks", "visual_prompt_embed", "visual_prompt_mask"
        ]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    # Public API for inference
    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference."""
        self._add_text_prompt(prompt, state)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference."""
        self._add_box_prompt(box, label, state)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            prompt=state["prompt"],
            prompt_mask=state["prompt_mask"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
