# Vesuvius Surface Detection Demo

A Streamlit demo app for the Vesuvius Challenge (Surface Detection) Kaggle competition. Detects papyrus surfaces inside 3D CT scan volumes of ancient Herculaneum scrolls using a 3D UNet segmentation model.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app works out of the box with synthetic demo data — no uploads needed.

## Features

- **Overview**: Explains the Vesuvius Challenge and the processing pipeline
- **Data Explorer**: Orthogonal slice views (XY, XZ, YZ) with interactive sliders
- **Model Architecture**: Visual diagram and layer details of SimpleUNet3D
- **Inference & Results**: Side-by-side CT / Ground Truth / Prediction / Overlay comparison
- **Metrics & Analysis**: Dice, IoU, Precision, Recall + 3D surface visualization

## Real Inference

To run real inference, upload:
1. A `.tif` CT volume in the Data Explorer page
2. Model weights (`.pth`) in the Inference & Results page

## Project Structure

```
vesuvius-app/
├── app.py              # Main Streamlit app
├── model.py            # SimpleUNet3D definition
├── inference.py        # Patch-based inference logic
├── synthetic.py        # Synthetic data generation
├── utils.py            # Visualization helpers, metrics
├── requirements.txt
└── README.md
```
