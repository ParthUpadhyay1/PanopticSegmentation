from __future__ import annotations
import numpy as np

# Challenge encoding:
# R = class_id
# instance_id = G*256 + B
# G = instance_id // 256, B = instance_id % 256

def encode_panoptic_rgb(class_map: np.ndarray, instance_map: np.ndarray) -> np.ndarray:
    """Create an HxWx3 uint8 image encoding (class_id, instance_id)."""
    if class_map.shape != instance_map.shape:
        raise ValueError("class_map and instance_map must have same shape")

    class_map = class_map.astype(np.uint8)
    instance_map = instance_map.astype(np.int32)

    g = (instance_map // 256).astype(np.uint8)
    b = (instance_map % 256).astype(np.uint8)

    rgb = np.stack([class_map, g, b], axis=-1).astype(np.uint8)
    return rgb
