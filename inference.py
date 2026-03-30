import torch
import numpy as np


PATCH_SIZE = 64
STRIDE = 32
THRESHOLD = 0.35


def sliding_window_inference(model, volume, device="cpu", patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Run patch-based sliding window inference on a 3D volume.

    Args:
        model: The 3D UNet model.
        volume: numpy array of shape (D, H, W).
        device: torch device string.
        patch_size: Size of cubic patches.
        stride: Stride between patches (half-overlap by default).

    Returns:
        probability_map: numpy array of shape (D, H, W) with values in [0, 1].
    """
    model.eval()
    D, H, W = volume.shape

    # Pad volume so patches tile evenly
    pad_d = (patch_size - D % patch_size) % patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    padded = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="reflect")

    pD, pH, pW = padded.shape
    output = np.zeros((pD, pH, pW), dtype=np.float32)
    counts = np.zeros((pD, pH, pW), dtype=np.float32)

    # Normalize volume to [0, 1]
    v_min, v_max = padded.min(), padded.max()
    if v_max > v_min:
        padded_norm = (padded - v_min) / (v_max - v_min)
    else:
        padded_norm = padded.astype(np.float32)

    total_patches = 0
    for d in range(0, pD - patch_size + 1, stride):
        for h in range(0, pH - patch_size + 1, stride):
            for w in range(0, pW - patch_size + 1, stride):
                total_patches += 1

    processed = 0
    with torch.no_grad():
        for d in range(0, pD - patch_size + 1, stride):
            for h in range(0, pH - patch_size + 1, stride):
                for w in range(0, pW - patch_size + 1, stride):
                    patch = padded_norm[d:d + patch_size, h:h + patch_size, w:w + patch_size]
                    tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)

                    pred = torch.sigmoid(model(tensor))
                    pred_np = pred.squeeze().cpu().numpy()

                    output[d:d + patch_size, h:h + patch_size, w:w + patch_size] += pred_np
                    counts[d:d + patch_size, h:h + patch_size, w:w + patch_size] += 1.0
                    processed += 1

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    output /= counts

    # Crop back to original size
    probability_map = output[:D, :H, :W]
    return probability_map


def apply_threshold(probability_map, threshold=THRESHOLD):
    """Convert probability map to binary mask."""
    return (probability_map >= threshold).astype(np.uint8)
