"""
================================================================================
VESUVIUS SURFACE DETECTION DEMO — COMBINED SOURCE
================================================================================
A Streamlit demo app for the Vesuvius Challenge (Surface Detection) Kaggle
competition. Detects papyrus surfaces inside 3D CT scan volumes of ancient
Herculaneum scrolls using a 3D UNet segmentation model.

Quick Start:
    pip install -r requirements.txt
    streamlit run app.py

Project Structure:
    vesuvius-app/
    ├── app.py              # Main Streamlit app
    ├── model.py            # SimpleUNet3D definition
    ├── inference.py        # Patch-based inference logic
    ├── synthetic.py        # Synthetic data generation
    ├── utils.py            # Visualization helpers, metrics
    ├── requirements.txt
    └── README.md

Requirements:
    streamlit
    torch
    numpy
    matplotlib
    plotly
    tifffile
    Pillow
    scipy
    scikit-image
================================================================================
"""


# ==============================================================================
# model.py — SimpleUNet3D Definition
# ==============================================================================

import torch
import torch.nn as nn


class SimpleUNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
                nn.Conv3d(o, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(1, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(64, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec3 = block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_architecture_summary():
    """Return a structured summary of the architecture."""
    return {
        "Encoder": [
            {"stage": "Enc1", "in_ch": 1, "out_ch": 16, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
            {"stage": "Pool", "in_ch": 16, "out_ch": 16, "ops": "MaxPool3d(2)"},
            {"stage": "Enc2", "in_ch": 16, "out_ch": 32, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
            {"stage": "Pool", "in_ch": 32, "out_ch": 32, "ops": "MaxPool3d(2)"},
            {"stage": "Enc3", "in_ch": 32, "out_ch": 64, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
            {"stage": "Pool", "in_ch": 64, "out_ch": 64, "ops": "MaxPool3d(2)"},
        ],
        "Bottleneck": [
            {"stage": "Bottleneck", "in_ch": 64, "out_ch": 128, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
        ],
        "Decoder": [
            {"stage": "Up3", "in_ch": 128, "out_ch": 64, "ops": "ConvTranspose3d(2,2) + Skip(Enc3)"},
            {"stage": "Dec3", "in_ch": 128, "out_ch": 64, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
            {"stage": "Up2", "in_ch": 64, "out_ch": 32, "ops": "ConvTranspose3d(2,2) + Skip(Enc2)"},
            {"stage": "Dec2", "in_ch": 64, "out_ch": 32, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
            {"stage": "Up1", "in_ch": 32, "out_ch": 16, "ops": "ConvTranspose3d(2,2) + Skip(Enc1)"},
            {"stage": "Dec1", "in_ch": 32, "out_ch": 16, "ops": "Conv3d->BN->ReLU->Conv3d->BN->ReLU"},
        ],
        "Output": [
            {"stage": "Out", "in_ch": 16, "out_ch": 1, "ops": "Conv3d(1x1x1)"},
        ],
    }


# ==============================================================================
# inference.py — Patch-Based Sliding Window Inference
# ==============================================================================

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


# ==============================================================================
# synthetic.py — Synthetic Data Generation
# ==============================================================================

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

        # Add holes (missed detections) -- zero out random patches
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


# ==============================================================================
# utils.py — Visualization Helpers & Metrics
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from skimage.measure import marching_cubes


# -- Color constants --
GOLD = "#D4A843"
CYAN = "#00D4FF"
DARK_BG = "#0E1117"


def compute_metrics(gt_mask, pred_mask):
    """Compute segmentation metrics between ground truth and prediction."""
    gt = gt_mask.astype(bool).flatten()
    pred = pred_mask.astype(bool).flatten()

    tp = np.sum(gt & pred)
    fp = np.sum(~gt & pred)
    fn = np.sum(gt & ~pred)
    tn = np.sum(~gt & ~pred)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "Dice Score": float(dice),
        "IoU (Jaccard)": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "Accuracy": float(accuracy),
        "True Positives": int(tp),
        "False Positives": int(fp),
        "False Negatives": int(fn),
    }


def create_slice_figure(slices_dict, title="", cmap_volume="gray", figsize=(16, 4)):
    """
    Create a matplotlib figure showing multiple slices side by side.

    Args:
        slices_dict: dict of {label: 2d_array}
        title: Overall title.
    """
    n = len(slices_dict)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    fig.patch.set_facecolor(DARK_BG)

    for ax, (label, img) in zip(axes, slices_dict.items()):
        ax.set_facecolor(DARK_BG)
        if "GT" in label or "Ground Truth" in label:
            ax.imshow(img, cmap="cyan_r" if hasattr(plt.cm, "cyan_r") else "cool", vmin=0, vmax=1)
        elif "Pred" in label and "Overlay" not in label:
            cmap = mcolors.LinearSegmentedColormap.from_list("gold_cmap", ["#0E1117", GOLD])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        elif "Overlay" in label:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(label, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")

    if title:
        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def create_overlay(ct_slice, gt_slice=None, pred_slice=None):
    """
    Create an RGB overlay image.
    CT in grayscale, GT in cyan, prediction in gold/amber.
    """
    # Normalize CT slice to [0, 1]
    ct_norm = ct_slice.astype(np.float32)
    if ct_norm.max() > ct_norm.min():
        ct_norm = (ct_norm - ct_norm.min()) / (ct_norm.max() - ct_norm.min())

    # Grayscale base
    rgb = np.stack([ct_norm * 0.6, ct_norm * 0.6, ct_norm * 0.6], axis=-1)

    # GT overlay in cyan
    if gt_slice is not None:
        gt_mask = gt_slice > 0.5
        rgb[gt_mask, 0] *= 0.3
        rgb[gt_mask, 1] = np.clip(rgb[gt_mask, 1] + 0.5, 0, 1)
        rgb[gt_mask, 2] = np.clip(rgb[gt_mask, 2] + 0.7, 0, 1)

    # Prediction overlay in gold/amber
    if pred_slice is not None:
        pred_mask = pred_slice > 0.5
        # Only show pred where there's no GT (or always)
        rgb[pred_mask, 0] = np.clip(rgb[pred_mask, 0] + 0.7, 0, 1)
        rgb[pred_mask, 1] = np.clip(rgb[pred_mask, 1] + 0.45, 0, 1)
        rgb[pred_mask, 2] *= 0.3

    return np.clip(rgb, 0, 1)


def create_probability_histogram(prob_map, figsize=(8, 4)):
    """Create a histogram of prediction probability values."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    values = prob_map.flatten()
    # Only plot non-zero values for clarity
    nonzero = values[values > 0.01]

    if len(nonzero) > 0:
        ax.hist(nonzero, bins=80, color=GOLD, alpha=0.85, edgecolor="none")
        ax.axvline(x=0.35, color=CYAN, linestyle="--", linewidth=2, label="Threshold (0.35)")
        ax.legend(facecolor="#1a1a2e", edgecolor=GOLD, labelcolor="white")

    ax.set_xlabel("Probability", color="white", fontsize=11)
    ax.set_ylabel("Voxel Count", color="white", fontsize=11)
    ax.set_title("Prediction Confidence Distribution", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    fig.tight_layout()
    return fig


def create_3d_surface_plot(mask, downsample=2, title="3D Surface Render"):
    """
    Create a 3D isosurface visualization using plotly.

    Args:
        mask: Binary 3D mask.
        downsample: Factor to reduce mesh complexity.
    """
    # Downsample for performance
    if downsample > 1:
        mask_ds = mask[::downsample, ::downsample, ::downsample]
    else:
        mask_ds = mask

    if mask_ds.sum() < 10:
        # Not enough surface to render
        fig = go.Figure()
        fig.add_annotation(text="Insufficient surface voxels for 3D rendering",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="white", size=16))
        fig.update_layout(
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
        )
        return fig

    try:
        verts, faces, _, _ = marching_cubes(mask_ds.astype(np.float32), level=0.5)

        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 2],  # W axis
                y=verts[:, 1],  # H axis
                z=verts[:, 0],  # D axis
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=GOLD,
                opacity=0.7,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3),
                lightposition=dict(x=100, y=200, z=300),
            )
        ])

        fig.update_layout(
            title=dict(text=title, font=dict(color="white", size=16)),
            scene=dict(
                xaxis=dict(title="W", backgroundcolor=DARK_BG, gridcolor="#222", color="white"),
                yaxis=dict(title="H", backgroundcolor=DARK_BG, gridcolor="#222", color="white"),
                zaxis=dict(title="D", backgroundcolor=DARK_BG, gridcolor="#222", color="white"),
                bgcolor=DARK_BG,
            ),
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            margin=dict(l=0, r=0, t=40, b=0),
            height=550,
        )
        return fig

    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Could not generate 3D surface mesh",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="white", size=16))
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG)
        return fig


def get_volume_stats(volume):
    """Return basic statistics about a volume."""
    return {
        "Shape": volume.shape,
        "Dtype": str(volume.dtype),
        "Min": float(volume.min()),
        "Max": float(volume.max()),
        "Mean": float(volume.mean()),
        "Std": float(volume.std()),
        "Non-zero voxels": int(np.count_nonzero(volume)),
        "Total voxels": int(volume.size),
    }


# ==============================================================================
# app.py — Main Streamlit Application
# ==============================================================================

import streamlit as st

st.set_page_config(
    page_title="Vesuvius Surface Detection",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for dark/moody theme --
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #D4A843, #F5DEB3, #D4A843);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #aaa;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #D4A843;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #D4A843;
    }
    .metric-label {
        color: #aaa;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    .pipeline-step {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        color: #eee;
        font-weight: 600;
    }
    .pipeline-arrow {
        color: #D4A843;
        font-size: 2rem;
        text-align: center;
        padding-top: 0.8rem;
    }
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.85rem;
        margin-top: 4rem;
        padding: 1.5rem;
        border-top: 1px solid #222;
    }
    .arch-table th {
        background-color: #1a1a2e !important;
        color: #D4A843 !important;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #111125 100%);
    }
    div[data-testid="stSidebar"] .stRadio label {
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Navigation ---
st.sidebar.markdown("## 🏛️ Vesuvius Demo")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Explorer", "Model Architecture", "Inference & Results", "Metrics & Analysis"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#666'>Course Project Demo<br>"
    "Vesuvius Challenge — Surface Detection<br>"
    "3D UNet Segmentation</small>",
    unsafe_allow_html=True,
)


# --- Session State: Load / Generate Data ---
@st.cache_data(show_spinner="Generating synthetic demo data...")
def load_synthetic_data():
    return generate_all_synthetic_data(shape=(64, 128, 128), seed=42)


def ensure_data():
    """Ensure synthetic data is available in session state."""
    if "data" not in st.session_state:
        st.session_state.data = load_synthetic_data()
        st.session_state.data_source = "synthetic"


# --- Page: Overview ---
if page == "Overview":
    st.markdown('<div class="main-header">Vesuvius Surface Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Detecting Papyrus Surfaces in 3D CT Scans of Ancient Herculaneum Scrolls</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Challenge overview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### The Vesuvius Challenge")
        st.markdown("""
        In **79 AD**, the eruption of Mount Vesuvius buried the ancient city of Herculaneum,
        carbonizing thousands of papyrus scrolls. These scrolls are too fragile to physically unroll,
        but modern **micro-CT scanning** can image their internal structure in 3D.

        The challenge: **detect the papyrus surface layers** within these dense 3D volumes so that
        ink traces on those surfaces can later be read — potentially revealing lost works of
        ancient philosophy and literature.

        This demo showcases a **3D UNet segmentation model** trained to identify papyrus surfaces
        within CT scan volumes.
        """)

    with col2:
        st.markdown("### Key Facts")
        st.markdown("""
        - **Input**: 3D CT volume (grayscale)
        - **Output**: Binary surface mask
        - **Model**: 3D UNet (encoder-decoder)
        - **Training**: 4 epochs (limited GPU)
        - **Loss**: BCE + Dice (50/50)
        - **Patch size**: 64 cubed voxels
        """)

    st.markdown("---")

    # Pipeline visualization
    st.markdown("### Processing Pipeline")
    cols = st.columns([2, 1, 2, 1, 2, 1, 2])

    steps = [
        ("📦", "Raw CT Volume", "3D grayscale scan"),
        ("→", "", ""),
        ("🧠", "3D UNet", "Patch-based inference"),
        ("→", "", ""),
        ("🌡️", "Probability Map", "Per-voxel confidence"),
        ("→", "", ""),
        ("🎯", "Binary Mask", "Threshold @ 0.35"),
    ]

    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            if title:
                st.markdown(
                    f'<div class="pipeline-step">'
                    f'<div style="font-size:2rem">{icon}</div>'
                    f'<div>{title}</div>'
                    f'<div style="color:#888;font-size:0.8rem">{desc}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div class="pipeline-arrow">{icon}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Architecture summary
    with st.expander("Architecture Details", expanded=False):
        st.markdown("""
        **SimpleUNet3D** — A 3D encoder-decoder with skip connections:

        | Path | Channels | Description |
        |------|----------|-------------|
        | Encoder 1 | 1 -> 16 | Two Conv3d + BN + ReLU blocks |
        | Encoder 2 | 16 -> 32 | + MaxPool3d(2) downsampling |
        | Encoder 3 | 32 -> 64 | + MaxPool3d(2) downsampling |
        | **Bottleneck** | **64 -> 128** | Deepest feature representation |
        | Decoder 3 | 128 -> 64 | ConvTranspose3d + skip from Enc3 |
        | Decoder 2 | 64 -> 32 | ConvTranspose3d + skip from Enc2 |
        | Decoder 1 | 32 -> 16 | ConvTranspose3d + skip from Enc1 |
        | Output | 16 -> 1 | 1x1x1 Conv3d -> sigmoid |

        **Training**: BCE + Dice loss, patch size 64 cubed, stride 32, 4 epochs on Kaggle GPU.
        """)

    with st.expander("Inference Strategy", expanded=False):
        st.markdown("""
        **Sliding Window Patch-Based Inference:**
        1. Pad the volume to fit patch boundaries
        2. Extract overlapping 64 cubed patches with stride 32 (50% overlap)
        3. Run each patch through the model
        4. Average predictions in overlapping regions
        5. Threshold at **0.35** to produce binary mask

        The overlap averaging reduces patch boundary artifacts and produces smoother predictions.
        """)


# --- Page: Data Explorer ---
elif page == "Data Explorer":
    st.markdown('<div class="main-header">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Explore CT scan volumes with orthogonal slice views</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Data source selection
    data_mode = st.radio(
        "Data Source",
        ["Synthetic Demo Data", "Upload .tif Volume"],
        horizontal=True,
    )

    volume = None

    if data_mode == "Upload .tif Volume":
        uploaded_file = st.file_uploader("Upload a .tif CT volume", type=["tif", "tiff"])
        if uploaded_file is not None:
            import tifffile
            import io
            volume = tifffile.imread(io.BytesIO(uploaded_file.read())).astype(np.float32)
            st.session_state.uploaded_volume = volume
            st.success(f"Loaded volume: shape={volume.shape}, dtype={volume.dtype}")
        elif "uploaded_volume" in st.session_state:
            volume = st.session_state.uploaded_volume
        else:
            st.info("Upload a .tif file or switch to synthetic demo data.")
    else:
        ensure_data()
        volume = st.session_state.data["volume"]

    if volume is not None:
        # Volume metadata
        stats = get_volume_stats(volume)
        st.markdown("### Volume Metadata")
        meta_cols = st.columns(4)
        meta_items = [
            ("Shape", f"{stats['Shape']}"),
            ("Dtype", stats["Dtype"]),
            ("Intensity Range", f"{stats['Min']:.0f} — {stats['Max']:.0f}"),
            ("Mean +/- Std", f"{stats['Mean']:.1f} +/- {stats['Std']:.1f}"),
        ]
        for col, (label, val) in zip(meta_cols, meta_items):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div style="color:#eee;font-size:1.1rem;font-weight:600">{val}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### Orthogonal Slice Views")

        D, H, W = volume.shape

        # Normalize volume for display
        v_disp = volume.astype(np.float32)
        v_disp = (v_disp - v_disp.min()) / (v_disp.max() - v_disp.min() + 1e-8)

        view_cols = st.columns(3)

        with view_cols[0]:
            st.markdown("**XY Plane** (axial)")
            z_idx = st.slider("Z slice", 0, D - 1, D // 2, key="xy_z")
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.imshow(v_disp[z_idx, :, :], cmap="gray")
            ax.set_title(f"Z = {z_idx}", color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with view_cols[1]:
            st.markdown("**XZ Plane** (coronal)")
            y_idx = st.slider("Y slice", 0, H - 1, H // 2, key="xz_y")
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.imshow(v_disp[:, y_idx, :], cmap="gray", aspect="auto")
            ax.set_title(f"Y = {y_idx}", color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with view_cols[2]:
            st.markdown("**YZ Plane** (sagittal)")
            x_idx = st.slider("X slice", 0, W - 1, W // 2, key="yz_x")
            fig, ax = plt.subplots(figsize=(5, 5))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.imshow(v_disp[:, :, x_idx], cmap="gray", aspect="auto")
            ax.set_title(f"X = {x_idx}", color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Intensity histogram
        with st.expander("Intensity Distribution"):
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.hist(volume.flatten(), bins=100, color="#D4A843", alpha=0.85, edgecolor="none")
            ax.set_xlabel("Intensity", color="white")
            ax.set_ylabel("Count", color="white")
            ax.set_title("Volume Intensity Histogram", color="white", fontweight="bold")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# --- Page: Model Architecture ---
elif page == "Model Architecture":
    st.markdown('<div class="main-header">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">SimpleUNet3D — 3D Encoder-Decoder with Skip Connections</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Parameter count
    model = SimpleUNet3D()
    n_params = count_parameters(model)

    p_cols = st.columns(3)
    with p_cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{n_params:,}</div>'
            f'<div class="metric-label">Total Parameters</div></div>',
            unsafe_allow_html=True,
        )
    with p_cols[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{n_params * 4 / 1e6:.1f} MB</div>'
            f'<div class="metric-label">Model Size (FP32)</div></div>',
            unsafe_allow_html=True,
        )
    with p_cols[2]:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">3</div>'
            '<div class="metric-label">Encoder Stages</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Visual architecture diagram
    st.markdown("### Architecture Diagram")

    diagram = """
    ```
    Input (1ch)
        |
        v
    +--------------+
    |   Enc1       |  1 -> 16 channels
    |  Conv3d x2   |  (BN + ReLU)
    +------+-------+
           | --------------------------------+  Skip Connection 1
           v                                 |
       MaxPool3d(2)                          |
           |                                 |
    +------+-------+                         |
    |   Enc2       |  16 -> 32 channels      |
    |  Conv3d x2   |  (BN + ReLU)           |
    +------+-------+                         |
           | -----------------------+ Skip 2 |
           v                        |        |
       MaxPool3d(2)                 |        |
           |                        |        |
    +------+-------+                |        |
    |   Enc3       |  32 -> 64      |        |
    |  Conv3d x2   |  (BN + ReLU)  |        |
    +------+-------+                |        |
           | --------------+ Skip 3 |        |
           v               |        |        |
       MaxPool3d(2)        |        |        |
           |               |        |        |
    +------+-------+       |        |        |
    | Bottleneck   |       |        |        |
    |  64 -> 128   |       |        |        |
    |  Conv3d x2   |       |        |        |
    +------+-------+       |        |        |
           |               |        |        |
       ConvTrans3d(2,2)    |        |        |
           |               |        |        |
           v               |        |        |
    +------+-------+       |        |        |
    |   Dec3       |<------+        |        |
    | 128 -> 64    |  (concat)      |        |
    +------+-------+                |        |
           |                        |        |
       ConvTrans3d(2,2)            |        |
           |                        |        |
    +------+-------+                |        |
    |   Dec2       |<---------------+        |
    |  64 -> 32    |  (concat)               |
    +------+-------+                         |
           |                                 |
       ConvTrans3d(2,2)                      |
           |                                 |
    +------+-------+                         |
    |   Dec1       |<------------------------+
    |  32 -> 16    |  (concat)
    +------+-------+
           |
    +------+-------+
    |  Conv3d 1x1  |  16 -> 1
    |  (Output)    |
    +--------------+
           |
           v
      Sigmoid -> Probability Map
    ```
    """
    st.markdown(diagram)

    st.markdown("---")

    # Detailed layer table
    st.markdown("### Layer Details")

    arch = get_architecture_summary()
    for section_name, layers in arch.items():
        st.markdown(f"**{section_name}**")
        table_data = []
        for layer in layers:
            table_data.append({
                "Stage": layer["stage"],
                "In Channels": layer["in_ch"],
                "Out Channels": layer["out_ch"],
                "Operations": layer["ops"],
            })
        st.table(table_data)

    st.markdown("---")

    # Training configuration
    st.markdown("### Training Configuration")
    train_cols = st.columns(4)
    configs = [
        ("Loss Function", "BCE + Dice (50/50)"),
        ("Patch Size", "64 x 64 x 64"),
        ("Stride", "32 (50% overlap)"),
        ("Threshold", "0.35"),
    ]
    for col, (label, value) in zip(train_cols, configs):
        with col:
            st.markdown(
                f'<div class="metric-card"><div style="color:#D4A843;font-size:1.3rem;font-weight:700">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    # Source code
    with st.expander("View Model Source Code"):
        st.code("""
class SimpleUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
                nn.Conv3d(o, o, 3, padding=1),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
            )
        self.enc1 = block(1, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = block(64, 128)
        self.up3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec3 = block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec2 = block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.dec1 = block(32, 16)
        self.out = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)
""", language="python")


# --- Page: Inference & Results ---
elif page == "Inference & Results":
    st.markdown('<div class="main-header">Inference & Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Patch-based sliding window segmentation results</div>', unsafe_allow_html=True)
    st.markdown("---")

    ensure_data()

    # Check for model upload
    run_real_inference = False
    st.markdown("#### Model Weights")
    model_file = st.file_uploader("Upload model weights (.pth) for real inference, or skip for demo predictions", type=["pth", "pt"])

    if model_file is not None:
        st.info("Model weights uploaded. Running real inference (this may take a while on CPU)...")
        run_real_inference = True

    volume = st.session_state.data["volume"]
    gt_mask = st.session_state.data["gt_mask"]

    if run_real_inference:
        import io

        model = SimpleUNet3D()
        state_dict = torch.load(io.BytesIO(model_file.read()), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        # Normalize volume for inference
        v_norm = volume.astype(np.float32)
        v_norm = (v_norm - v_norm.min()) / (v_norm.max() - v_norm.min() + 1e-8)

        with st.spinner("Running patch-based inference..."):
            pred_prob = sliding_window_inference(model, v_norm, device="cpu")
            pred_mask = apply_threshold(pred_prob)

        st.session_state.data["pred_prob"] = pred_prob
        st.session_state.data["pred_mask"] = pred_mask
        st.success("Inference complete!")
    else:
        pred_prob = st.session_state.data["pred_prob"]
        pred_mask = st.session_state.data["pred_mask"]
        st.caption("Using synthetic demo predictions (upload .pth weights for real inference)")

    st.markdown("---")

    # Prediction statistics
    st.markdown("### Prediction Statistics")
    stat_cols = st.columns(4)
    fg_voxels = int(pred_mask.sum())
    total_voxels = int(pred_mask.size)
    coverage = fg_voxels / total_voxels * 100

    with stat_cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{fg_voxels:,}</div>'
            f'<div class="metric-label">Foreground Voxels</div></div>',
            unsafe_allow_html=True,
        )
    with stat_cols[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{coverage:.2f}%</div>'
            f'<div class="metric-label">Volume Coverage</div></div>',
            unsafe_allow_html=True,
        )
    with stat_cols[2]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{pred_prob.mean():.4f}</div>'
            f'<div class="metric-label">Mean Probability</div></div>',
            unsafe_allow_html=True,
        )
    with stat_cols[3]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{pred_prob.max():.4f}</div>'
            f'<div class="metric-label">Max Probability</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Interactive slice comparison
    st.markdown("### Slice Comparison")

    plane = st.radio("View Plane", ["XY (Axial)", "XZ (Coronal)", "YZ (Sagittal)"], horizontal=True, key="results_plane")

    D, H, W = volume.shape

    if plane == "XY (Axial)":
        max_idx = D - 1
        default_idx = D // 2
    elif plane == "XZ (Coronal)":
        max_idx = H - 1
        default_idx = H // 2
    else:
        max_idx = W - 1
        default_idx = W // 2

    slice_idx = st.slider("Slice Index", 0, max_idx, default_idx, key="results_slice")

    # Extract slices based on plane
    if plane == "XY (Axial)":
        ct_slice = volume[slice_idx, :, :]
        gt_slice = gt_mask[slice_idx, :, :]
        pred_p_slice = pred_prob[slice_idx, :, :]
        pred_m_slice = pred_mask[slice_idx, :, :]
    elif plane == "XZ (Coronal)":
        ct_slice = volume[:, slice_idx, :]
        gt_slice = gt_mask[:, slice_idx, :]
        pred_p_slice = pred_prob[:, slice_idx, :]
        pred_m_slice = pred_mask[:, slice_idx, :]
    else:
        ct_slice = volume[:, :, slice_idx]
        gt_slice = gt_mask[:, :, slice_idx]
        pred_p_slice = pred_prob[:, :, slice_idx]
        pred_m_slice = pred_mask[:, :, slice_idx]

    overlay = create_overlay(ct_slice, gt_slice, pred_m_slice)

    # Side-by-side display
    fig_cols = st.columns(4)

    with fig_cols[0]:
        st.markdown("**CT Slice**")
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ct_disp = ct_slice.astype(np.float32)
        ct_disp = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-8)
        ax.imshow(ct_disp, cmap="gray")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with fig_cols[1]:
        st.markdown("**Ground Truth**")
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.imshow(gt_slice, cmap="cool", vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with fig_cols[2]:
        st.markdown("**Prediction**")
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        gold_cmap = mcolors.LinearSegmentedColormap.from_list("gold", ["#0E1117", "#D4A843"])
        ax.imshow(pred_p_slice, cmap=gold_cmap, vmin=0, vmax=1)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with fig_cols[3]:
        st.markdown("**Overlay**")
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")
        ax.imshow(overlay)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Legend
    st.markdown(
        "<div style='text-align:center;color:#888;font-size:0.9rem'>"
        "<span style='color:#00D4FF'>■</span> Ground Truth (Cyan) &nbsp;&nbsp; "
        "<span style='color:#D4A843'>■</span> Prediction (Gold) &nbsp;&nbsp; "
        "Overlap shown where both are present</div>",
        unsafe_allow_html=True,
    )

    # Probability histogram
    with st.expander("Prediction Probability Distribution"):
        fig = create_probability_histogram(pred_prob)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# --- Page: Metrics & Analysis ---
elif page == "Metrics & Analysis":
    st.markdown('<div class="main-header">Metrics & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantitative evaluation and 3D visualization</div>', unsafe_allow_html=True)
    st.markdown("---")

    ensure_data()

    gt_mask = st.session_state.data["gt_mask"]
    pred_mask = st.session_state.data["pred_mask"]
    pred_prob = st.session_state.data["pred_prob"]

    # Compute metrics
    metrics = compute_metrics(gt_mask, pred_mask)

    st.markdown("### Segmentation Metrics")

    m_cols = st.columns(4)
    primary_metrics = [
        ("Dice Score", metrics["Dice Score"], "Higher is better (0-1)"),
        ("IoU (Jaccard)", metrics["IoU (Jaccard)"], "Higher is better (0-1)"),
        ("Precision", metrics["Precision"], "How accurate are positive predictions"),
        ("Recall", metrics["Recall"], "How many true surfaces were found"),
    ]

    for col, (name, value, desc) in zip(m_cols, primary_metrics):
        with col:
            # Color-code: green if > 0.5, yellow if > 0.3, red otherwise
            if value > 0.5:
                color = "#4CAF50"
            elif value > 0.3:
                color = "#D4A843"
            else:
                color = "#FF5252"
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:2rem;font-weight:800;color:{color}">{value:.4f}</div>'
                f'<div class="metric-label">{name}</div>'
                f'<div style="color:#555;font-size:0.75rem;margin-top:4px">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Detailed metrics
    with st.expander("Detailed Metrics"):
        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.metric("True Positives", f"{metrics['True Positives']:,}")
        with detail_cols[1]:
            st.metric("False Positives", f"{metrics['False Positives']:,}")
        with detail_cols[2]:
            st.metric("False Negatives", f"{metrics['False Negatives']:,}")

        st.markdown("""
        **Interpretation:**
        - The model was trained for only **4 epochs** with limited GPU time
        - Predictions are expected to be **noisy and fragmented** compared to ground truth
        - Higher precision than recall suggests the model is conservative — it finds some surfaces accurately but misses many
        - Additional training epochs and hyperparameter tuning would improve these scores
        """)

    st.markdown("---")

    # Probability histogram
    st.markdown("### Prediction Confidence Distribution")
    fig = create_probability_histogram(pred_prob, figsize=(10, 4))
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # 3D surface visualization
    st.markdown("### 3D Surface Visualization")

    viz_choice = st.radio(
        "Display",
        ["Ground Truth Surface", "Predicted Surface", "Both (side by side)"],
        horizontal=True,
    )

    if viz_choice == "Both (side by side)":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ground Truth**")
            fig_gt = create_3d_surface_plot(gt_mask, downsample=2, title="Ground Truth Surface")
            st.plotly_chart(fig_gt, use_container_width=True)
        with col2:
            st.markdown("**Prediction**")
            fig_pred = create_3d_surface_plot(pred_mask, downsample=2, title="Predicted Surface")
            st.plotly_chart(fig_pred, use_container_width=True)
    elif viz_choice == "Ground Truth Surface":
        fig_3d = create_3d_surface_plot(gt_mask, downsample=2, title="Ground Truth Surface")
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        fig_3d = create_3d_surface_plot(pred_mask, downsample=2, title="Predicted Surface")
        st.plotly_chart(fig_3d, use_container_width=True)

    # Comparison slice view
    with st.expander("Error Analysis — Per-Slice Comparison"):
        st.markdown("View where the model succeeds and fails across slices.")
        D = gt_mask.shape[0]
        err_idx = st.slider("Z slice for error analysis", 0, D - 1, D // 2, key="err_slice")

        gt_s = gt_mask[err_idx]
        pred_s = pred_mask[err_idx]

        # Create error map: TP=green, FP=red, FN=blue
        error_map = np.zeros((*gt_s.shape, 3), dtype=np.float32)
        tp = (gt_s > 0.5) & (pred_s > 0.5)
        fp = (gt_s < 0.5) & (pred_s > 0.5)
        fn = (gt_s > 0.5) & (pred_s < 0.5)

        error_map[tp] = [0.3, 0.9, 0.3]   # Green: correct
        error_map[fp] = [0.9, 0.2, 0.2]   # Red: false positive
        error_map[fn] = [0.2, 0.4, 0.9]   # Blue: false negative

        err_cols = st.columns(3)
        with err_cols[0]:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.imshow(gt_s, cmap="cool", vmin=0, vmax=1)
            ax.set_title("Ground Truth", color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with err_cols[1]:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            gold_cmap = mcolors.LinearSegmentedColormap.from_list("gold", ["#0E1117", "#D4A843"])
            ax.imshow(pred_s, cmap=gold_cmap, vmin=0, vmax=1)
            ax.set_title("Prediction", color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with err_cols[2]:
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            ax.imshow(error_map)
            ax.set_title("Error Map", color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown(
            "<div style='text-align:center;color:#888;font-size:0.9rem'>"
            "<span style='color:#4CAF50'>■</span> True Positive &nbsp;&nbsp; "
            "<span style='color:#E53935'>■</span> False Positive &nbsp;&nbsp; "
            "<span style='color:#5C6BC0'>■</span> False Negative</div>",
            unsafe_allow_html=True,
        )


# --- Footer ---
st.markdown("---")
st.markdown(
    '<div class="footer">'
    "Built for the <b>Vesuvius Challenge</b> — Surface Detection Task<br>"
    "Course project demo | 3D UNet Segmentation on CT Scan Volumes<br>"
    '<span style="color:#D4A843">Powered by Streamlit & PyTorch</span>'
    "</div>",
    unsafe_allow_html=True,
)
