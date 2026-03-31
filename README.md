VESUVIUS SURFACE DETECTION DEMO
================================

Detects papyrus surfaces in 3D CT scans of ancient Herculaneum scrolls
using a 3D UNet deep learning model. Built for the Vesuvius Challenge
Kaggle competition.


HOW TO RUN
----------
  pip install -r requirements.txt
  streamlit run app.py

Works immediately with built-in synthetic data — no uploads required.


THE PROBLEM
-----------
Ancient scrolls were buried in volcanic ash and can't be physically
unrolled. CT scans create 3D volumes of the scrolls, and the goal is
to detect the papyrus surface layers hidden inside those volumes.


HOW IT WORKS
------------

THE MODEL (model.py)
  A 3D UNet — an encoder-decoder neural network with skip connections.
  - Takes a 3D CT volume as input
  - Outputs a binary mask: which voxels are papyrus vs. background
  - About 2.7 million parameters

INFERENCE (inference.py)
  CT volumes are too large to process all at once, so sliding window
  inference is used:
  - Breaks the volume into overlapping 64x64x64 patches
  - Runs each patch through the model
  - Stitches results back together, averaging overlapping areas

SYNTHETIC DATA (synthetic.py)
  Generates fake-but-realistic CT volumes for demo purposes:
  - Simulated papyrus sheets as curved sinusoidal surfaces
  - Realistic background noise and texture

METRICS & VISUALIZATION (utils.py)
  - Dice / IoU / Precision / Recall scores to evaluate predictions
  - Slice views, overlays, probability histograms, 3D surface plots


APP PAGES
---------
  Overview            Background on the challenge and pipeline
  Data Explorer       Slice through the CT volume (XY, XZ, YZ)
  Model Architecture  Layer-by-layer breakdown of the UNet
  Inference & Results CT / Ground Truth / Prediction / Overlay view
  Metrics & Analysis  Scores + 3D surface visualization


USING REAL DATA (optional)
--------------------------
  1. Go to Data Explorer and upload a .tif CT volume
  2. Go to Inference & Results and upload model weights (.pth)

The app will run live inference on your data instead of the demo.


FILE STRUCTURE
--------------
  app.py            Main UI and page routing
  model.py          3D UNet definition
  inference.py      Sliding window inference
  synthetic.py      Demo data generation
  utils.py          Metrics and visualization
  requirements.txt  Python dependencies
