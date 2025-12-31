import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime

# Graph libraries (altair, matplotlib, seaborn) imports are removed

st.set_page_config(page_title="Bio-Image Quantifier Pro (Extraction Only)", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Pro Edition (Extraction)")
st.caption("2025 Final Ver: Analysis & Extraction Only (No Graphing)")

# --- Color Definitions (Translated) ---
COLOR_MAP = {
    "Brown (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "Green (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "Red (RFP)":   {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "Blue (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

# --- Functions ---
def get_mask(hsv_img, color_name, sens, bright_min):
    if color_name == "Red (RFP)":
        lower1 = np.array([0, 30, bright_min])
        upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 30, bright_min])
        upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        conf = COLOR_MAP[color_name]
        l = np.clip(conf["lower"] - sens, 0, 255)
        u = np.clip(conf["upper"] + sens, 0, 255)
        l[2] = max(l[2], bright_min)
        return cv2.inRange(hsv_img, l, u)

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0:
            pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# --- Sidebar ---
with st.sidebar:
    st.header("Analysis Recipe")
    mode = st.selectbox("Select Analysis Mode:", [
        "1. Area Fraction (Area)",
        "2. Cell Count (Count)",
        "3. Colocalization (Colocalization)",
        "4. Spatial Distance (Spatial Distance)",
        "5. Ratio Trend Analysis (Ratio Analysis)"
    ])
    st.divider()

    if mode == "5. Ratio Trend Analysis (Ratio Analysis)":
        st.markdown("### 🔢 Batch Conditions")
        trend_metric = st.radio("Metric:", ["Colocalization", "Area"])
        ratio_val = st.number_input("Condition Value (Ratio/Conc):", value=0, step=10)
        ratio_unit = st.text_input("Unit:", value="%", key="unit")
        sample_group = f"{ratio_val}{ratio_unit}"
        st.info(f"Label: **{sample_group}**")
        st.divider()
        if trend_metric == "Colocalization":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", list(COLOR_MAP.keys()), index=3) 
                sens_a = st.slider("Sens A", 5, 50, 20, key="t_s_a")
                bright_a = st.slider("Bright A", 0, 255, 60, key="t_b_a")
            with c2:
                target_b = st.selectbox("CH-B (Target):", list(COLOR_MAP.keys()), index=2) 
                sens_b = st.slider("Sens B", 5, 50, 20, key="t_s_b")
                bright_b = st.slider("Bright B", 0, 255, 60, key="t_b_b")
        else:
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("Sensitivity", 5, 50, 20, key="t_s_a")
            bright_a = st.slider("Brightness", 0, 255, 60, key="t_b_a")
    else:
        sample_group = st.text_input("Group Name (X-axis):", value="Control")
        st.divider()
        if mode == "1. Area Fraction (Area)":
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()))
            sens_a = st.slider("Sensitivity", 5, 50, 20)
            bright_a = st.slider("Brightness", 0, 255, 60)
        elif mode == "2. Cell Count (Count)":
            min_size = st.slider("Min Size (px)", 10, 500, 50)
            bright_count = st.slider("Brightness Threshold", 0, 255, 50)
        elif mode == "3. Colocalization (Colocalization)":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("Sens A", 5, 50, 20)
                bright_a = st.slider("Bright A", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (Target):", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("Sens B", 5, 50, 20)
                bright_b = st.slider("Bright B", 0, 255, 60)
        elif mode == "4. Spatial Distance (Spatial Distance)":
            target_a = st.selectbox("Point A:", list(COLOR_MAP.keys()), index=2)
            target_b = st.selectbox("Point B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("Color Sens", 5, 50, 20)
            bright_common = st.slider("Brightness", 0, 255, 60)

    if st.button("Clear All History"):
        st.session_state.analysis_history = []
        st.rerun()
        with st.sidebar:
            st.divider()
            st.caption("【Disclaimer】")
            st.caption("""
            This tool is intended for assistive purposes in bio-image analysis. 
            Since results may vary depending on lighting conditions and user settings, 
            final interpretations should be made by the user based on professional expertise.
            """)
        with st.sidebar:
    # --- 既存のコントロール（輝度調整など）の後に挿入 ---
            st.divider()
            st.caption("【Disclaimer】")
            st.caption("""
            This tool is intended for assistive purposes in bio-image analysis. 
            Since results (e.g., area, intensity) may vary depending on lighting 
            conditions, resolution, and user-defined thresholds, final scientific 
            interpretations must be made by the user based on professional expertise. 
            The developer is not liable for any discrepancies or research outcomes 
            resulting from the use of this software.
            """)

# --- Main Area ---
uploaded_files = st.file_uploader("Upload Images (Batch)", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Processing {len(uploaded_files)} images...")
    batch_results = []
    
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            val, unit = 0.0, ""
            res_display = img_rgb.copy()
            
            # --- Analysis Logic ---
            if mode == "1. Area Fraction (Area)" or (mode.startswith("5.") and trend_metric == "Area"):
                mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = f"% Area"
                res_display = mask
            elif mode == "2. Cell Count (Count)":
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                final = cv2.bitwise_and(th, otsu)
                cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                val, unit = len(valid), "cells"
                cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
            elif mode == "3. Colocalization (Colocalization)" or (mode.startswith("5.") and trend_metric == "Colocalization"):
                mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = f"% Coloc"
                res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
            elif mode == "4. Spatial Distance (Spatial Distance)":
                mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
                mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
                pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                if pts_a and pts_b:
                    val = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                else: val = 0
                unit = "px Dist"
                res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)
            
            # --- Prevent Negative ---
            val = max(0.0, val)

            entry = {
                "Group": sample_group,
                "Value": val,
                "Unit": unit,
                "Is_Trend": mode.startswith("5."),
                "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            }
            batch_results.append(entry)
            
            # --- Expander ---
            with st.expander(f"📷 Image {i+1}: {file.name}", expanded=True):
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Analyzed", use_container_width=True)

    if st.button(f"Add {len(batch_results)} Results", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.rerun()

# --- Report Section (Data Only) ---
if st.session_state.analysis_history:
    st.divider()
    st.header("💾 Data Export")
    
    df = pd.DataFrame(st.session_state.analysis_history)
    df["Value"] = df["Value"].clip(lower=0) 

    # ★修正点: UTC(世界標準時)に9時間足して、日本時間(JST)にする
    now = datetime.datetime.now() + datetime.timedelta(hours=9)
    file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    st.dataframe(df, use_container_width=True)
    
    st.download_button(
        label="📥 Download CSV (JST Timestamp)", 
        data=df.to_csv(index=False).encode('utf-8'), 
        file_name=file_name, 
        mime="text/csv"
    )
