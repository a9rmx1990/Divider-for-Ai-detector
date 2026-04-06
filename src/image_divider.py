import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Union, List, Generator
from pathlib import Path


@dataclass
class PatchMetadata:
    """Metadata detailing the origin and size of a single image patch."""
    index: int                  # Unique index for this patch
    x_start: int                # X-coordinate start in the original image (0-indexed)
    y_start: int                # Y-coordinate start in the original image (0-indexed)
    x_end: int                  # X-coordinate end in the original image
    y_end: int                  # Y-coordinate end in the original image
    patch_size: int             # Dimension of the patch (assumed square: patch_size x patch_size)
    overlap_ratio: float        # Percentage [0.0 - 1.0) of overlap used during generation
    pad_left: int               # Number of pixels padded on the left of this patch (always 0 here)
    pad_right: int              # Number of pixels padded on the right of this patch
    pad_top: int                # Number of pixels padded on the top of this patch (always 0 here)
    pad_bottom: int             # Number of pixels padded on the bottom of this patch
    scale_level: int            # Scale index denoting which multi-scale level this patch belongs to
    original_width: int         # Original image width
    original_height: int        # Original image height


@dataclass
class Patch:
    """An single structured image patch holding image array and metadata."""
    data: np.ndarray            # The loaded image patch, shape: (patch_size, patch_size, C)
    metadata: PatchMetadata     # Corresponding tracking metadata


class ImageDivider:
    """
    A robust, memory-efficient Divider module for AI-generated image detection.
    
    Responsibilities:
      - Split large input images into overlapping, fixed-size patches.
      - Never resize or distort internal image content (avoids confusing artifact detectors).
      - Maintain perfect edge boundary information via symmetric or constant padding.
      - Support streaming memory models for huge input files (4K+).
    """
    
    def __init__(
        self,
        patch_sizes: Union[int, List[int]],
        overlap_ratio: float = 0.25,
        padding_mode: str = 'reflect'
    ):
        """
        Initializes the Divider.
        
        Args:
            patch_sizes (int or list[int]): One or multiple target patch sizes.
                e.g., 224, or [64, 128, 224] for multi-scale setups.
            overlap_ratio (float): Ratio determining spatial overlap between adjacent patches.
                Must be between 0.0 (no overlap) and 1.0 (exclusive). (Recommended: 0.25 - 0.50).
            padding_mode (str): Behavior to use when patch dimension exceeds the image edge.
                Options: 
                - 'reflect': Mirrors the boundary (best for CNNs to prevent flat border edges).
                - 'constant': Pads with 0 (black).
                - 'edge': Replicates the last pixel line.
        """
        if isinstance(patch_sizes, int):
            self.patch_sizes = [patch_sizes]
        else:
            self.patch_sizes = sorted(patch_sizes)
            
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("Overlap ratio must be firmly between 0.0 and 1.0 (exclusive).")
            
        self.overlap_ratio = overlap_ratio
        
        self.pad_map = {
            'reflect': cv2.BORDER_REFLECT_101,
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE
        }
        
        if padding_mode not in self.pad_map:
            raise ValueError(f"padding_mode must be one of {list(self.pad_map.keys())}")
        
        self.padding_mode = padding_mode
        self.cv2_pad_type = self.pad_map[padding_mode]

    def _calculate_padded_size(self, dim_size: int, patch_size: int, stride: int) -> int:
        """
        Calculates the minimum padded dimension size required to flawlessly fit sliding window patches,
        ensuring the last patch covers up to dim_size perfectly.
        """
        if dim_size <= patch_size:
            return patch_size
        
        # Calculate how many full sliding window jumps we strictly need to fit dim_size
        n_minus_1 = math.ceil((dim_size - patch_size) / stride)
        return n_minus_1 * stride + patch_size

    def stream_patches(self, image_input: Union[str, Path, np.ndarray]) -> Generator[Patch, None, None]:
        """
        A memory-efficient generator that lazily yields formatted patches.
        
        Args:
            image_input (Union[str, Path, np.ndarray]): Either file path to an image or a loaded NumPy array.
            
        Yields:
            Patch instances holding localized numpy slices and metadata tracking its relative topology.
        """
        # Load image if file path is provided
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to load image from {image_input}")
                
            # OpenCV loads files as BGR; we convert them to RGB for cleaner NN ingestion
            if len(img.shape) >= 3 and img.shape[2] >= 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_input
            
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must essentially be a verified local path or an underlying numpy array.")
            
        orig_h, orig_w = img.shape[:2]
        global_patch_index = 0
        
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            # Stride logic: if overlap_ratio = 0.25 and patch is 128, stride translates to 96
            stride = max(1, int(patch_size * (1 - self.overlap_ratio)))
            
            padded_w = self._calculate_padded_size(orig_w, patch_size, stride)
            padded_h = self._calculate_padded_size(orig_h, patch_size, stride)
            
            pad_x = padded_w - orig_w
            pad_y = padded_h - orig_h
            
            # Non-destructively pad the global image boundaries using native compiled OpenCV functions
            if pad_x > 0 or pad_y > 0:
                padded_img = cv2.copyMakeBorder(
                    img, 
                    top=0, bottom=pad_y, 
                    left=0, right=pad_x, 
                    borderType=self.cv2_pad_type,
                    value=[0, 0, 0] if self.padding_mode == 'constant' else None
                )
            else:
                padded_img = img
                
            # Perform bounding box patch extraction logic (row-major streaming layout)
            for y in range(0, padded_h - patch_size + 1, stride):
                for x in range(0, padded_w - patch_size + 1, stride):
                    # We copy locally so underlying main memory reference array isn't held completely for each tiny slice object.
                    patch_data = padded_img[y:y + patch_size, x:x + patch_size].copy()
                    
                    x_start = x
                    y_start = y
                    x_end = min(x + patch_size, orig_w)
                    y_end = min(y + patch_size, orig_h)
                    
                    # Compute localized out-of-bounds pad tracking (pixels generated strictly from padding routine)
                    pad_right = patch_size - (x_end - x_start)
                    pad_bottom = patch_size - (y_end - y_start)
                    
                    metadata = PatchMetadata(
                        index=global_patch_index,
                        x_start=x_start,
                        y_start=y_start,
                        x_end=x_end,
                        y_end=y_end,
                        patch_size=patch_size,
                        overlap_ratio=self.overlap_ratio,
                        pad_left=0,
                        pad_right=pad_right,
                        pad_top=0,
                        pad_bottom=pad_bottom,
                        scale_level=scale_idx,
                        original_width=orig_w,
                        original_height=orig_h
                    )
                    
                    yield Patch(data=patch_data, metadata=metadata)
                    global_patch_index += 1
