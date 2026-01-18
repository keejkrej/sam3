# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import torch
from PIL import Image
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

    def _mask_to_bbox(self, mask):
        """Convert mask to bounding box [x, y, w, h]"""
        nonzero = torch.nonzero(mask > 0.5, as_tuple=False)
        if len(nonzero) == 0:
            return [0, 0, mask.shape[-1], mask.shape[-2]]
        
        y_coords = nonzero[:, 0]
        x_coords = nonzero[:, 1]
        
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        return [x_min.item(), y_min.item(), 
                (x_max - x_min + 1).item(), (y_max - y_min + 1).item()]

    def _mask_to_grid_simple(self, mask, spacing=16, max_points=50):
        """
        Simple uniform grid sampling within mask
        
        Args:
            mask: [H, W] binary mask
            spacing: Pixel spacing between grid points
            max_points: Maximum number of points to return
        
        Returns:
            points: List of [x, y] pixel coordinates
            labels: List of boolean labels (all True)
        """
        H, W = mask.shape[-2:]
        points = []
        
        # Sample at regular intervals
        for y in range(0, H, spacing):
            for x in range(0, W, spacing):
                if len(points) >= max_points:
                    break
                if mask[y, x] > 0.5:
                    points.append([x, y])
            if len(points) >= max_points:
                break
        
        return points, [True] * len(points)

    def _mask_to_grid_hybrid(self, mask, max_points=100, boundary_ratio=0.3):
        """
        Hybrid sampling: boundary points + interior grid
        
        Args:
            mask: [H, W] binary mask  
            max_points: Total maximum points to sample
            boundary_ratio: Fraction of points for boundary (0.0-1.0)
        
        Returns:
            points: List of [x, y] pixel coordinates
            labels: List of boolean labels (all True)
        """
        import torch.nn.functional as F
        
        H, W = mask.shape[-2:]
        points = []
        
        # 1. Extract boundary points using morphological operations
        mask_float = mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Create simple 3x3 kernel for boundary detection
        kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask_float.dtype)
        dilated = F.conv2d(mask_float, kernel, padding=1) > 0
        eroded = F.conv2d(mask_float, kernel, padding=1) == 9
        boundary = dilated & (~eroded)
        
        # Sample boundary points
        boundary_coords = torch.nonzero(boundary.squeeze(), as_tuple=False)
        n_boundary = int(max_points * boundary_ratio)
        
        if len(boundary_coords) > 0:
            # Uniformly sample boundary points
            if len(boundary_coords) > n_boundary:
                step = len(boundary_coords) // n_boundary
                for i in range(0, len(boundary_coords), step):
                    if len(points) < n_boundary:
                        y, x = boundary_coords[i][0], boundary_coords[i][1]
                        points.append([x.item(), y.item()])
            else:
                for coord in boundary_coords:
                    y, x = coord[0], coord[1]
                    points.append([x.item(), y.item()])
        
        # 2. Sample interior grid points
        remaining_points = max_points - len(points)
        if remaining_points > 0:
            interior_mask = mask.float() * (1.0 - boundary.squeeze().float())
            interior_mask = (interior_mask > 0.5).float()
            
            # Calculate adaptive spacing for interior
            mask_area = interior_mask.sum().item()
            if mask_area > 0:
                spacing = max(8, int((mask_area / remaining_points) ** 0.5))
                
                for y in range(0, H, spacing):
                    for x in range(0, W, spacing):
                        if len(points) >= max_points:
                            break
                        if interior_mask[y, x] > 0.5:
                            points.append([x, y])
                    if len(points) >= max_points:
                        break
        
        # 3. Add centroid if not already present
        if len(points) < max_points:
            coords = torch.nonzero(mask > 0.5, as_tuple=False)
            if len(coords) > 0:
                centroid_y = coords[:, 0].float().mean().round().long()
                centroid_x = coords[:, 1].float().mean().round().long()
                centroid_point = [centroid_x.item(), centroid_y.item()]
                
                # Avoid duplicates
                if centroid_point not in points:
                    points.append(centroid_point)
        
        # Fallback to simple grid if no points found
        if len(points) == 0:
            # Fallback to simple uniform grid
            spacing = max(8, int((H * W / max_points) ** 0.5))
            for y in range(0, H, spacing):
                for x in range(0, W, spacing):
                    if len(points) >= max_points:
                        break
                    if mask[y, x] > 0.5:
                        points.append([x, y])
                if len(points) >= max_points:
                    break
        
        return points, [True] * len(points)

    @torch.inference_mode()
    def add_prompt(self, payload: Dict, state: Dict) -> Dict:
        """Add a prompt using a unified payload interface.
        
        Args:
            payload: Dictionary with 'type' key and prompt data
            state: The processor state dictionary
            
        Examples:
            add_prompt({"type": "text", "text": "a cat"}, state)
            add_prompt({"type": "point", "point": [100, 200], "label": True}, state)
            add_prompt({"type": "box", "box": [10, 20, 100, 200], "label": True}, state)
            add_prompt({"type": "mask", "mask": mask_tensor, "label": True}, state)
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_prompt")
        
        prompt_type = payload.get("type")
        
        # Handle text prompt
        if prompt_type == "text":
            text_outputs = self.model.backbone.forward_text([payload["text"]], device=self.device)
            state["backbone_out"].update(text_outputs)
            
            # Create dummy geometric prompt and encode
            geometric_prompt = self.model._get_dummy_prompt()
            with torch.profiler.record_function("SAM3Image._encode_prompt"):
                encoded = self.model._encode_prompt(
                    backbone_out=state["backbone_out"],
                    find_input=self.find_stage,
                    geometric_prompt=geometric_prompt,
                )
            
            # Store encoded results
            for key, value in encoded.items():
                state[key] = value
            return state
        
        # Handle geometric prompts (point, box, mask)
        if prompt_type not in ["point", "box", "mask"]:
            raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
        # Set up dummy text for visual-only mode
        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)
        
        # Initialize geometric prompt if needed
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        
        # Add specific prompt data
        if prompt_type == "point":
            img_w = state["original_width"]
            img_h = state["original_height"]
            normalized_point = [payload["point"][0] / img_w, payload["point"][1] / img_h]
            points = torch.tensor(normalized_point, device=self.device, dtype=torch.float32).view(1, 1, 2)
            labels = torch.tensor([payload["label"]], device=self.device, dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_points(points, labels)
        
        elif prompt_type == "box":
            img_w = state["original_width"]
            img_h = state["original_height"]
            # Convert XYWH to CxCyWH using built-in helper and normalize to [0,1] range
            # Box format: [x, y, w, h] in pixel coordinates
            box_tensor = torch.tensor(payload["box"], device=self.device, dtype=torch.float32).view(1, 4)
            box_cxcywh = box_ops.box_xywh_to_cxcywh(box_tensor)
            # Normalize to [0,1] range
            normalized_box = box_cxcywh / torch.tensor([img_w, img_h, img_w, img_h], device=self.device, dtype=torch.float32)
            boxes = normalized_box.view(1, 1, 4)
            labels = torch.tensor([payload["label"]], device=self.device, dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_boxes(boxes, labels)
        
        elif prompt_type == "mask":
            mask = payload["mask"]
            encoding_method = payload.get("encoding_method", "default")
            
            # Get original image dimensions
            img_h = state["original_height"]
            img_w = state["original_width"]
            
            # Ensure mask is a tensor
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            
            # Resize mask to match original image dimensions if needed
            mask_h, mask_w = mask.shape[-2:]
            if mask_h != img_h or mask_w != img_w:
                mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
                mask_pil = mask_pil.resize((img_w, img_h), Image.Resampling.NEAREST)
                mask = torch.from_numpy(np.array(mask_pil) / 255.0).float()
            
            if encoding_method == "default":
                # Original mask encoding logic
                # Resize to processor resolution
                mask_pil = Image.fromarray((mask.numpy() * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
                mask_tensor = torch.from_numpy(np.array(mask_resized) / 255.0).float()
                
                masks = mask_tensor.to(device=self.device, dtype=torch.float32).view(1, 1, 1, *mask_tensor.shape[-2:])
                labels = torch.tensor([payload["label"]], device=self.device, dtype=torch.long).view(1, 1)
                state["geometric_prompt"].append_masks(masks, labels)
                
            elif encoding_method == "box":
                # Convert mask to bounding box
                bbox = self._mask_to_bbox(mask)
                self.add_prompt({
                    "type": "box",
                    "box": bbox,
                    "label": payload["label"]
                }, state)
                
            elif encoding_method == "grid_simple":
                # Simple uniform grid sampling
                spacing = payload.get("spacing", 16)
                max_points = payload.get("max_points", 50)
                
                points, labels = self._mask_to_grid_simple(
                    mask, spacing=spacing, max_points=max_points
                )
                
                # Add as point prompts
                if points:
                    normalized_points = [
                        [x / img_w, y / img_h] for x, y in points
                    ]
                    
                    points_tensor = torch.tensor(
                        normalized_points, device=self.device, dtype=torch.float32
                    ).view(len(points), 1, 2)
                    labels_tensor = torch.tensor(
                        labels, device=self.device, dtype=torch.bool
                    ).view(len(points), 1)
                    
                    state["geometric_prompt"].append_points(points_tensor, labels_tensor)
                    
            elif encoding_method == "grid_hybrid":
                # Hybrid boundary + interior sampling
                max_points = payload.get("max_points", 100)
                boundary_ratio = payload.get("boundary_ratio", 0.3)
                
                points, labels = self._mask_to_grid_hybrid(
                    mask, max_points=max_points, boundary_ratio=boundary_ratio
                )
                
                # Add as point prompts
                if points:
                    normalized_points = [
                        [x / img_w, y / img_h] for x, y in points
                    ]
                    
                    points_tensor = torch.tensor(
                        normalized_points, device=self.device, dtype=torch.float32
                    ).view(len(points), 1, 2)
                    labels_tensor = torch.tensor(
                        labels, device=self.device, dtype=torch.bool
                    ).view(len(points), 1)
                    
                    state["geometric_prompt"].append_points(points_tensor, labels_tensor)
            
            else:
                raise ValueError(f"Unsupported encoding_method: {encoding_method}")
        
        # Encode prompts
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            encoded = self.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"],
            )
        
        # Store encoded results
        for key, value in encoded.items():
            state[key] = value
        
        # Clean up geometric_prompt
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
        self.add_prompt({"type": "text", "text": prompt}, state)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference."""
        self.add_prompt({"type": "box", "box": box, "label": label}, state)
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
