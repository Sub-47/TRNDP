import numpy as np


def ensure_standard_map(array, world_size, name, dtype=np.float32):
    array = np.asarray(array)
    shape = (world_size, world_size)
    if array.shape != shape:
        raise ValueError(f"{name} map must have shape {shape}, got {array.shape}")
    return array.astype(dtype)


def is_normalized(array, tolerance=1e-4):
    return bool(array.min() >= -tolerance and array.max() <= 1 + tolerance)
