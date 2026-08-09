from __future__ import annotations

import json
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

ROOT = Path(st.sidebar.text_input("Dataset root", "datasets/work"))
FRAMES = ROOT / "frames"
LABELS = ROOT / "labels"
META = ROOT / "reviews.jsonl"
LABELS.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="GolfIQ Vision Lab", layout="wide")
st.title("GolfIQ Vision Lab")
st.caption("Review legally sourced golf footage and create source-traceable YOLO labels.")

frames = sorted([p for p in FRAMES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
if not frames:
    st.warning(f"No frames found in {FRAMES}")
    st.stop()

idx = st.number_input("Frame", min_value=0, max_value=len(frames) - 1, value=0, step=1)
frame_path = frames[int(idx)]
image = Image.open(frame_path).convert("RGB")
width, height = image.size

left, right = st.columns([2, 1])
with left:
    st.image(image, caption=frame_path.name, use_container_width=True)

with right:
    st.subheader("Review")
    source_group = st.text_input("Source group", frame_path.stem.split("__")[0])
    visibility = st.selectbox("Ball visibility", ["visible", "not_visible", "unclear"])
    impact = st.selectbox("Impact state", ["pre_impact", "impact", "post_impact", "unknown"])
    x = st.number_input("Box x (pixels)", min_value=0, max_value=width, value=0)
    y = st.number_input("Box y (pixels)", min_value=0, max_value=height, value=0)
    w = st.number_input("Box width", min_value=0, max_value=width, value=0)
    h = st.number_input("Box height", min_value=0, max_value=height, value=0)
    reviewer = st.text_input("Reviewer", "charlie")
    notes = st.text_area("Notes")

    if st.button("Save review", type="primary"):
        label_path = LABELS / f"{frame_path.stem}.txt"
        if visibility == "visible":
            if min(w, h) <= 0:
                st.error("Visible balls require a non-zero box.")
                st.stop()
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            nw = w / width
            nh = h / height
            label_path.write_text(f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}\n")
        else:
            label_path.write_text("")

        record = {
            "frame": str(frame_path),
            "label": str(label_path),
            "source_group": source_group,
            "visibility": visibility,
            "impact": impact,
            "reviewer": reviewer,
            "notes": notes,
            "image_width": width,
            "image_height": height,
        }
        with META.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        st.success("Review saved")

st.divider()
st.write(f"Frames: {len(frames)} | Labels: {len(list(LABELS.glob('*.txt')))}")
