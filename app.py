import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from db import log_inference, fetch_inference_history

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


# ─── Sidebar Navigation ─────────────────────────────────────────────
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


# ─── Session State: Load / Generate Data ─────────────────────────────
@st.cache_data(show_spinner="Generating synthetic demo data...")
def load_synthetic_data():
    from synthetic import generate_all_synthetic_data
    return generate_all_synthetic_data(shape=(64, 128, 128), seed=42)


def ensure_data():
    """Ensure synthetic data is available in session state."""
    if "data" not in st.session_state:
        st.session_state.data = load_synthetic_data()
        st.session_state.data_source = "synthetic"


# ─── Page: Overview ──────────────────────────────────────────────────
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
        - **Patch size**: 64³ voxels
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
        | Encoder 1 | 1 → 16 | Two Conv3d + BN + ReLU blocks |
        | Encoder 2 | 16 → 32 | + MaxPool3d(2) downsampling |
        | Encoder 3 | 32 → 64 | + MaxPool3d(2) downsampling |
        | **Bottleneck** | **64 → 128** | Deepest feature representation |
        | Decoder 3 | 128 → 64 | ConvTranspose3d + skip from Enc3 |
        | Decoder 2 | 64 → 32 | ConvTranspose3d + skip from Enc2 |
        | Decoder 1 | 32 → 16 | ConvTranspose3d + skip from Enc1 |
        | Output | 16 → 1 | 1×1×1 Conv3d → sigmoid |

        **Training**: BCE + Dice loss, patch size 64³, stride 32, 4 epochs on Kaggle GPU.
        """)

    with st.expander("Inference Strategy", expanded=False):
        st.markdown("""
        **Sliding Window Patch-Based Inference:**
        1. Pad the volume to fit patch boundaries
        2. Extract overlapping 64³ patches with stride 32 (50% overlap)
        3. Run each patch through the model
        4. Average predictions in overlapping regions
        5. Threshold at **0.35** to produce binary mask

        The overlap averaging reduces patch boundary artifacts and produces smoother predictions.
        """)


# ─── Page: Data Explorer ─────────────────────────────────────────────
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
        from utils import get_volume_stats

        # Volume metadata
        stats = get_volume_stats(volume)
        st.markdown("### Volume Metadata")
        meta_cols = st.columns(4)
        meta_items = [
            ("Shape", f"{stats['Shape']}"),
            ("Dtype", stats["Dtype"]),
            ("Intensity Range", f"{stats['Min']:.0f} — {stats['Max']:.0f}"),
            ("Mean ± Std", f"{stats['Mean']:.1f} ± {stats['Std']:.1f}"),
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


# ─── Page: Model Architecture ────────────────────────────────────────
elif page == "Model Architecture":
    st.markdown('<div class="main-header">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">SimpleUNet3D — 3D Encoder-Decoder with Skip Connections</div>', unsafe_allow_html=True)
    st.markdown("---")

    from model import SimpleUNet3D, count_parameters, get_architecture_summary

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

    # Build a visual text-based diagram
    diagram = """
    ```
    Input (1ch)
        │
        ▼
    ┌─────────────┐
    │   Enc1       │  1 → 16 channels
    │  Conv3d×2    │  (BN + ReLU)
    └──────┬──────┘
           │ ─────────────────────────────────┐  Skip Connection 1
           ▼                                   │
       MaxPool3d(2)                            │
           │                                   │
    ┌──────┴──────┐                            │
    │   Enc2       │  16 → 32 channels         │
    │  Conv3d×2    │  (BN + ReLU)              │
    └──────┬──────┘                            │
           │ ──────────────────────┐  Skip 2   │
           ▼                       │           │
       MaxPool3d(2)                │           │
           │                       │           │
    ┌──────┴──────┐                │           │
    │   Enc3       │  32 → 64      │           │
    │  Conv3d×2    │  (BN + ReLU)  │           │
    └──────┬──────┘                │           │
           │ ─────────────┐ Skip 3 │           │
           ▼               │       │           │
       MaxPool3d(2)        │       │           │
           │               │       │           │
    ┌──────┴──────┐        │       │           │
    │ Bottleneck   │        │       │           │
    │  64 → 128    │        │       │           │
    │  Conv3d×2    │        │       │           │
    └──────┬──────┘        │       │           │
           │               │       │           │
       ConvTrans3d(2,2)    │       │           │
           │               │       │           │
           ▼               │       │           │
    ┌──────┴──────┐        │       │           │
    │   Dec3       │◄──────┘       │           │
    │ 128 → 64     │  (concat)     │           │
    └──────┬──────┘                │           │
           │                       │           │
       ConvTrans3d(2,2)           │           │
           │                       │           │
    ┌──────┴──────┐                │           │
    │   Dec2       │◄──────────────┘           │
    │  64 → 32     │  (concat)                 │
    └──────┬──────┘                            │
           │                                   │
       ConvTrans3d(2,2)                        │
           │                                   │
    ┌──────┴──────┐                            │
    │   Dec1       │◄──────────────────────────┘
    │  32 → 16     │  (concat)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Conv3d 1×1  │  16 → 1
    │  (Output)    │
    └─────────────┘
           │
           ▼
      Sigmoid → Probability Map
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
        ("Patch Size", "64 × 64 × 64"),
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


# ─── Page: Inference & Results ────────────────────────────────────────
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
        from model import SimpleUNet3D
        from inference import sliding_window_inference, apply_threshold
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
        st.session_state.data["inference_model_file"] = model_file.name
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

    from utils import create_overlay

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
        from utils import create_probability_histogram
        fig = create_probability_histogram(pred_prob)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ─── Page: Metrics & Analysis ────────────────────────────────────────
elif page == "Metrics & Analysis":
    st.markdown('<div class="main-header">Metrics & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Quantitative evaluation and 3D visualization</div>', unsafe_allow_html=True)
    st.markdown("---")

    ensure_data()

    gt_mask = st.session_state.data["gt_mask"]
    pred_mask = st.session_state.data["pred_mask"]
    pred_prob = st.session_state.data["pred_prob"]

    from utils import compute_metrics, create_probability_histogram, create_3d_surface_plot

    # Compute metrics
    metrics = compute_metrics(gt_mask, pred_mask)

    # Log to MongoDB if this was a real inference run (not synthetic demo)
    if st.session_state.data.get("inference_model_file"):
        fg = int(pred_mask.sum())
        log_inference(
            model_filename=st.session_state.data["inference_model_file"],
            metrics=metrics,
            fg_voxels=fg,
            mean_prob=float(pred_prob.mean()),
        )
        # Clear flag so we only log once per run
        st.session_state.data["inference_model_file"] = None

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

    # Inference history from MongoDB
    st.markdown("---")
    st.markdown("### Inference History")
    history = fetch_inference_history(limit=10)
    if history:
        import pandas as pd
        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        df = df.rename(columns={
            "timestamp": "Time",
            "model_file": "Model File",
            "dice_score": "Dice",
            "iou": "IoU",
            "precision": "Precision",
            "recall": "Recall",
            "fg_voxels": "FG Voxels",
            "mean_prob": "Mean Prob",
            "threshold": "Threshold",
        })
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No inference history yet. Run real inference to populate the log.")

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


# ─── Footer ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="footer">'
    "Built for the <b>Vesuvius Challenge</b> — Surface Detection Task<br>"
    "Course project demo | 3D UNet Segmentation on CT Scan Volumes<br>"
    '<span style="color:#D4A843">Powered by Streamlit & PyTorch</span>'
    "</div>",
    unsafe_allow_html=True,
)
