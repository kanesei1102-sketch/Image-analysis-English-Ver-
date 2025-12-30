import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="Bio-Image Quantifier: Diagnostic", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Diagnostic Edition")
st.caption("Debug Mode Enabled: Adjust sliders until you see the masks clearly.")

# --- Constants ---
DEFAULT_HUE = {
    "Red_Low": (0, 35), "Red_High": (170, 180),  
    "Green": (25, 95), "Blue": (90, 150), "Brown": (0, 40)         
}
COLORS = ["Red (RFP)", "Green (GFP)", "Blue (DAPI)", "Brown (DAB)"]

# --- Sidebar ---
with st.sidebar:
    st.header("1. Tuning Panel")
    
    # Global Calibration
    with st.expander("🎨 Color Definitions (HSV)", expanded=True):
        h_red_l = st.slider("Red (Low)", 0, 50, DEFAULT_HUE["Red_Low"])
        h_red_h = st.slider("Red (High)", 150, 180, DEFAULT_HUE["Red_High"])
        h_green = st.slider("Green", 15, 120, DEFAULT_HUE["Green"])
        h_blue = st.slider("Blue", 80, 170, DEFAULT_HUE["Blue"])
        h_brown = st.slider("Brown", 0, 60, DEFAULT_HUE["Brown"])

    mode = st.selectbox("Mode:", ["Colocalization (%)", "Area Fraction (%)", "Cell Count"])
    
    st.divider()
    
    # Specific Settings for Colocalization
    if mode == "Colocalization (%)":
        st.markdown("### 🛠 Debugging")
        view_mode = st.radio("Show View:", ["Result (Overlay)", "Check Mask A (Base)", "Check Mask B (Target)"])
        
        st.divider()
        c1, c2 = st.columns(2)
        target_a = c1.selectbox("Base (A):", COLORS, index=2) # Blue default
        target_b = c2.selectbox("Target (B):", COLORS, index=0) # Red default
        
        st.caption("⬇️ **Lower these if detection is 0%**")
        bright_a = c1.slider("Bright A", 0, 255, 20) # Low default
        bright_b = c2.slider("Bright B", 0, 255, 20) # Low default
        sens_a = c1.slider("Sens A", 5, 50, 40) # High sens default
        sens_b = c2.slider("Sens B", 5, 50, 40) # High sens default

    elif mode == "Area Fraction (%)":
        target_a = st.selectbox("Target:", COLORS, index=0)
        sens_a = st.slider("Sens", 5, 50, 40)
        bright_a = st.slider("Bright", 0, 255, 20)
        view_mode = "Result (Overlay)"
        
    else: # Count
        bright_count = st.slider("Threshold", 0, 255, 40)
        min_size = st.slider("Min Size", 0, 500, 20)
        view_mode = "Result (Overlay)"

    if st.button("🗑 Reset"):
        st.session_state.analysis_history = []
        st.rerun()

# --- Logic ---
def get_mask(hsv, color, s, b):
    min_sat = max(0, 40 - s)
    _, v = cv2.threshold(hsv[:,:,2], b, 255, cv2.THRESH_BINARY)
    
    mask = np.zeros_like(v)
    if "Red" in color:
        mask = cv2.inRange(hsv, (h_red_l[0], min_sat, 0), (h_red_l[1], 255, 255)) | \
               cv2.inRange(hsv, (h_red_h[0], min_sat, 0), (h_red_h[1], 255, 255))
    elif "Green" in color: mask = cv2.inRange(hsv, (h_green[0], min_sat, 0), (h_green[1], 255, 255))
    elif "Blue" in color: mask = cv2.inRange(hsv, (h_blue[0], min_sat, 0), (h_blue[1], 255, 255))
    elif "Brown" in color: mask = cv2.inRange(hsv, (h_brown[0], min_sat, 0), (h_brown[1], 255, 255))
    return cv2.bitwise_and(mask, v)

# --- Main ---
uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    batch = []
    for f in uploaded_files:
        f.seek(0)
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        val, unit = 0, ""
        disp = rgb.copy()
        
        if mode == "Colocalization (%)":
            m1 = get_mask(hsv, target_a, sens_a, bright_a) # Base
            m2 = get_mask(hsv, target_b, sens_b, bright_b) # Target
            
            inter = cv2.bitwise_and(m1, m2)
            denom = cv2.countNonZero(m1)
            val = (cv2.countNonZero(inter)/denom*100) if denom > 0 else 0.0
            unit = "% Coloc"
            
            # Debug Views
            if view_mode == "Check Mask A (Base)":
                disp = cv2.cvtColor(m1, cv2.COLOR_GRAY2RGB) # Show White Mask
            elif view_mode == "Check Mask B (Target)":
                disp = cv2.cvtColor(m2, cv2.COLOR_GRAY2RGB) # Show White Mask
            else:
                # Result View: Red(Base) + Green(Target) = Yellow(Overlap)
                z = np.zeros_like(m1)
                # Assign colors roughly to match logic (Red=Ch1, Green=Ch2)
                disp = cv2.merge([z, m2, m1]) # R=m1, G=m2 (OpenCV is BGR, Streamlit reads as RGB... trickery here)
                # Let's stick to explicit:
                # We want overlap to be Yellow (R+G).
                # If m1 is Red component, m2 is Green component.
                # Create true RGB composition
                comp = np.zeros_like(rgb)
                comp[:,:,0] = m1 # Red Channel
                comp[:,:,1] = m2 # Green Channel
                disp = comp

        elif mode == "Area Fraction (%)":
            m = get_mask(hsv, target_a, sens_a, bright_a)
            val = (cv2.countNonZero(m)/m.size)*100
            unit = "% Area"
            disp = cv2.addWeighted(rgb, 0.5, cv2.cvtColor(m, cv2.COLOR_GRAY2RGB), 0.5, 0)
            # Make mask Green for visibility
            mask_rgb = np.zeros_like(rgb); mask_rgb[:,:,1] = m
            disp = cv2.addWeighted(rgb, 0.7, mask_rgb, 0.5, 0)

        elif mode == "Cell Count":
             # Simplified for demo
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in cnts if cv2.contourArea(c) > min_size]
            val, unit = len(valid), "cells"
            cv2.drawContours(disp, valid, -1, (0,255,0), 2)

        batch.append({"Group": "Test", "Value": val, "Unit": unit})
        
        with st.expander(f"Result: {val:.2f} {unit}", expanded=True):
            c1, c2 = st.columns(2)
            c1.image(rgb, caption="Original")
            c2.image(disp, caption=f"View: {view_mode}")

    if st.button("Download CSV"):
        df = pd.DataFrame(batch)
        st.download_button("Get CSV", df.to_csv(index=False).encode('utf-8'), "data.csv")
