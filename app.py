import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Bio-Image Quantifier: Extraction", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Extraction Engine")
st.caption("Objective Data Extraction Pipeline. No plotting, just numbers.")

# --- Constants ---
DEFAULT_HUE = {
    "Red_Low": (0, 35), "Red_High": (170, 180),  
    "Green": (25, 95), "Blue": (90, 150), "Brown": (0, 40)         
}
COLORS = ["Brown (DAB)", "Green (GFP)", "Red (RFP)", "Blue (DAPI)"]

# --- Sidebar ---
with st.sidebar:
    st.header("1. Settings")
    show_mask = st.checkbox("Show Mask (Debug)", value=False)
    
    with st.expander("Color Calibration (HSV)", expanded=False):
        h_red_l = st.slider("Red (Low)", 0, 50, DEFAULT_HUE["Red_Low"])
        h_red_h = st.slider("Red (High)", 150, 180, DEFAULT_HUE["Red_High"])
        h_green = st.slider("Green", 15, 120, DEFAULT_HUE["Green"])
        h_blue = st.slider("Blue", 80, 170, DEFAULT_HUE["Blue"])
        h_brown = st.slider("Brown", 0, 60, DEFAULT_HUE["Brown"])

    mode = st.selectbox("Mode:", ["Area Fraction", "Cell Count", "Colocalization"])
    
    st.divider()
    sample_group = st.text_input("Group Label:", value="Control")
    
    # Params
    target_a, target_b = "Blue (DAPI)", "Red (RFP)"
    sens, bright = 30, 60
    
    if "Area" in mode:
        target_a = st.selectbox("Target:", COLORS, index=2)
        sens = st.slider("Sens", 5, 50, 30)
        bright = st.slider("Bright", 0, 255, 60)
    elif "Count" in mode:
        bright = st.slider("Threshold", 0, 255, 50)
    elif "Colocalization" in mode:
        c1, c2 = st.columns(2)
        target_a = c1.selectbox("Base:", COLORS, index=3)
        target_b = c2.selectbox("Target:", COLORS, index=2)
        sens = c1.slider("Sens", 5, 50, 30)
        bright = c2.slider("Bright", 0, 255, 30)

    if st.button("🗑 Reset Data"):
        st.session_state.analysis_history = []
        st.rerun()

# --- Logic ---
def get_mask(hsv, color, s, b):
    min_sat = max(0, 40 - s)
    _, v = cv2.threshold(hsv[:,:,2], b, 255, cv2.THRESH_BINARY)
    mask = np.zeros_like(v)
    if "Red" in color: mask = cv2.inRange(hsv, (h_red_l[0], min_sat, 0), (h_red_l[1], 255, 255)) | cv2.inRange(hsv, (h_red_h[0], min_sat, 0), (h_red_h[1], 255, 255))
    elif "Green" in color: mask = cv2.inRange(hsv, (h_green[0], min_sat, 0), (h_green[1], 255, 255))
    elif "Blue" in color: mask = cv2.inRange(hsv, (h_blue[0], min_sat, 0), (h_blue[1], 255, 255))
    elif "Brown" in color: mask = cv2.inRange(hsv, (h_brown[0], min_sat, 0), (h_brown[1], 255, 255))
    return cv2.bitwise_and(mask, v)

# --- Main ---
uploaded_files = st.file_uploader("2. Upload Images", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    batch = []
    for f in uploaded_files:
        f.seek(0)
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        val, unit = 0, ""
        
        if "Area" in mode:
            m = get_mask(hsv, target_a, sens, bright)
            val = (cv2.countNonZero(m)/m.size)*100
            unit = "% Area"
        elif "Count" in mode:
            # Simple count logic
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, bright, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            val, unit = len([c for c in cnts if cv2.contourArea(c)>50]), "cells"
        elif "Colocalization" in mode:
            m1 = get_mask(hsv, target_a, sens, bright)
            m2 = get_mask(hsv, target_b, sens, bright)
            inter = cv2.bitwise_and(m1, m2)
            val = (cv2.countNonZero(inter)/cv2.countNonZero(m1)*100) if cv2.countNonZero(m1)>0 else 0
            unit = "% Coloc"
            
        batch.append({"Group": sample_group, "Value": val, "Unit": unit})
        st.caption(f"Processed: {val:.2f} {unit}")

    if st.button(f"📥 Save {len(batch)} results", type="primary"):
        st.session_state.analysis_history.extend(batch)
        st.success("Saved!")

if st.session_state.analysis_history:
    st.divider()
    df = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(df)
    st.download_button("💾 Download CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
