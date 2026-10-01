import streamlit as st
from pymongo import MongoClient
from datetime import datetime, timezone


@st.cache_resource
def get_db():
    """Return a MongoDB database instance, cached across reruns."""
    uri = st.secrets.get("MONGO_URI", None)
    if not uri:
        return None
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return client["vesuvius"]


def log_inference(model_filename: str, metrics: dict, fg_voxels: int, mean_prob: float, threshold: float = 0.35):
    """Save an inference run record to MongoDB."""
    db = get_db()
    if db is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc),
        "model_file": model_filename,
        "dice_score": round(float(metrics.get("Dice Score", 0)), 6),
        "iou": round(float(metrics.get("IoU (Jaccard)", 0)), 6),
        "precision": round(float(metrics.get("Precision", 0)), 6),
        "recall": round(float(metrics.get("Recall", 0)), 6),
        "fg_voxels": int(fg_voxels),
        "mean_prob": round(float(mean_prob), 6),
        "threshold": threshold,
    }
    db["inference_logs"].insert_one(record)


def fetch_inference_history(limit: int = 10):
    """Return the most recent inference log entries."""
    db = get_db()
    if db is None:
        return []
    cursor = db["inference_logs"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return list(cursor)
