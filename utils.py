import numpy as np
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
