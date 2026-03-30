import numpy as np
from scipy.ndimage import gaussian_filter


def generate_smooth_noise(shape, scale=16, seed=42):
    """Generate smooth Perlin-noise-like texture via upscaled Gaussian-filtered noise."""
    rng = np.random.RandomState(seed)
    small_shape = tuple(max(s // scale, 4) for s in shape)
    noise = rng.randn(*small_shape).astype(np.float32)
    # Upsample via scipy zoom
    from scipy.ndimage import zoom
    factors = tuple(s / ss for s, ss in zip(shape, small_shape))
    upsampled = zoom(noise, factors, order=3)
    # Crop/pad to exact shape
    result = np.zeros(shape, dtype=np.float32)
    slices = tuple(slice(0, min(u, s)) for u, s in zip(upsampled.shape, shape))
    result[slices] = upsampled[slices]
    return result


def generate_curved_surface(shape, thickness=3, seed=42):
    """
    Generate a smooth curved surface mask through the volume.
    Simulates a papyrus sheet curving through the 3D space.
    """
    D, H, W = shape
    rng = np.random.RandomState(seed)

    # Create a base height map (smooth 2D surface)
    y_coords, x_coords = np.meshgrid(np.linspace(0, 2 * np.pi, H), np.linspace(0, 2 * np.pi, W), indexing="ij")

    # Combine sinusoidal waves for a realistic curved sheet
    height_map = (
        D * 0.4
        + D * 0.15 * np.sin(y_coords * 0.8 + 0.3)
        + D * 0.1 * np.cos(x_coords * 1.2 + 0.7)
        + D * 0.05 * np.sin(y_coords * 2.1 + x_coords * 1.5)
    )

    # Add small-scale smooth perturbations
    perturbation = generate_smooth_noise((H, W), scale=8, seed=seed + 1)
    perturbation = (perturbation - perturbation.mean()) / (perturbation.std() + 1e-8)
    height_map += perturbation * D * 0.03

    # Create 3D mask from height map
    mask = np.zeros(shape, dtype=np.float32)
    z_coords = np.arange(D).reshape(-1, 1, 1)

    for t in range(thickness):
        offset = t - thickness // 2
        layer = np.exp(-0.5 * ((z_coords - height_map[np.newaxis, :, :] - offset) ** 2) / 1.0)
        mask += layer

    mask = (mask > 0.5).astype(np.float32)
    return mask


def generate_synthetic_ct_volume(shape=(64, 128, 128), seed=42):
    """
    Generate a synthetic CT volume with a visible papyrus sheet.
    Returns volume with values roughly in [0, 65535] range (uint16-like).
    """
    D, H, W = shape

    # Background: smooth noise simulating material density variations
    bg = generate_smooth_noise(shape, scale=16, seed=seed)
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    bg = bg * 20000 + 15000  # Background intensity range

    # Fine grain noise
    rng = np.random.RandomState(seed + 10)
    fine_noise = rng.randn(*shape).astype(np.float32) * 2000
    fine_noise = gaussian_filter(fine_noise, sigma=1.0)

    # Papyrus sheet: higher intensity curved band
    surface_mask = generate_curved_surface(shape, thickness=3, seed=seed)
    papyrus_intensity = generate_smooth_noise(shape, scale=8, seed=seed + 20)
    papyrus_intensity = (papyrus_intensity - papyrus_intensity.min()) / (papyrus_intensity.max() - papyrus_intensity.min() + 1e-8)
    papyrus_intensity = papyrus_intensity * 15000 + 40000  # Papyrus is brighter

    volume = bg + fine_noise + surface_mask * papyrus_intensity
    volume = np.clip(volume, 0, 65535).astype(np.float32)

    return volume


def generate_synthetic_ground_truth(shape=(64, 128, 128), seed=42):
    """Generate a clean ground truth surface mask."""
    return generate_curved_surface(shape, thickness=3, seed=seed)


def generate_synthetic_prediction(gt_mask, quality="partial", seed=42):
    """
    Generate a synthetic prediction that looks like a partially-trained model output.

    Args:
        gt_mask: Ground truth binary mask.
        quality: 'partial' for 4-epoch-like results, 'good' for better results.
        seed: Random seed.

    Returns:
        pred_prob: Probability map (float32, [0, 1]).
        pred_mask: Binary mask after thresholding.
    """
    rng = np.random.RandomState(seed)
    shape = gt_mask.shape

    if quality == "partial":
        # Start with GT, add imperfections typical of early training
        pred_prob = gt_mask.copy().astype(np.float32)

        # Lower confidence: scale down probabilities
        pred_prob *= rng.uniform(0.4, 0.85, size=shape).astype(np.float32)

        # Add holes (missed detections) — zero out random patches
        for _ in range(15):
            d = rng.randint(0, shape[0])
            h = rng.randint(0, shape[1])
            w = rng.randint(0, shape[2])
            sz_h = rng.randint(5, 20)
            sz_w = rng.randint(5, 20)
            pred_prob[
                max(0, d - 1):min(shape[0], d + 2),
                max(0, h - sz_h // 2):min(shape[1], h + sz_h // 2),
                max(0, w - sz_w // 2):min(shape[2], w + sz_w // 2),
            ] *= rng.uniform(0.0, 0.3)

        # Add false positives: random blobs near the surface
        for _ in range(10):
            d = rng.randint(0, shape[0])
            h = rng.randint(0, shape[1])
            w = rng.randint(0, shape[2])
            sz = rng.randint(3, 10)
            blob = np.zeros(shape, dtype=np.float32)
            blob[
                max(0, d - sz):min(shape[0], d + sz),
                max(0, h - sz):min(shape[1], h + sz),
                max(0, w - sz):min(shape[2], w + sz),
            ] = rng.uniform(0.3, 0.7)
            pred_prob += blob

        # Smooth slightly to look more like neural network output
        pred_prob = gaussian_filter(pred_prob, sigma=0.8)

        # Add global noise
        noise = rng.randn(*shape).astype(np.float32) * 0.08
        pred_prob += noise

        pred_prob = np.clip(pred_prob, 0, 1)
    else:
        pred_prob = gt_mask.copy().astype(np.float32) * rng.uniform(0.7, 0.95, size=shape).astype(np.float32)
        pred_prob = gaussian_filter(pred_prob, sigma=0.5)
        pred_prob = np.clip(pred_prob, 0, 1)

    pred_mask = (pred_prob >= 0.35).astype(np.uint8)
    return pred_prob, pred_mask


def generate_all_synthetic_data(shape=(64, 128, 128), seed=42):
    """Generate a complete set of synthetic demo data."""
    volume = generate_synthetic_ct_volume(shape, seed=seed)
    gt_mask = generate_synthetic_ground_truth(shape, seed=seed)
    pred_prob, pred_mask = generate_synthetic_prediction(gt_mask, quality="partial", seed=seed)
    return {
        "volume": volume,
        "gt_mask": gt_mask,
        "pred_prob": pred_prob,
        "pred_mask": pred_mask,
    }
