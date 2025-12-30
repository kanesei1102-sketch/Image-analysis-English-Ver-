import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bio-Image Quantifier Pro", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2025 Final Ver: Overlay Support, Batch Processing & Teal Blue Graphs")

# --- Constants (Optimized for sensitivity to detect Yellow overlap) ---
DEFAULT_HUE = {
    "Red_Low": (0, 35),      # Covers up to Yellow (~30)
    "Red_High": (170, 180),  
    "Green": (25, 95),       # Covers down to Yellow (~30)
    "Blue": (90, 150),       
    "Brown": (0, 40)         
}
COLORS = ["Brown (DAB)", "Green (GFP)", "Red (RFP)", "Blue (DAPI)"]

# --- Sidebar ---
with st.sidebar:
    st.header("Analysis Recipe")
    
    show_mask = st.checkbox("🛠 Show Binary Mask", value=False)

    with st.expander("🎨 Color Calibration (HSV)", expanded=True):
        h_red_l = st.slider("Red (Low) Range", 0, 50, DEFAULT_HUE["Red_Low"], key="h_r_l")
        h_red_h = st.slider("Red (High) Range", 150, 180, DEFAULT_HUE["Red_High"], key="h_r_h")
        h_green = st.slider("Green (GFP) Range", 15, 120, DEFAULT_HUE["Green"], key="h_g")
        h_blue = st.slider("Blue (DAPI) Range", 80, 170, DEFAULT_HUE["Blue"], key="h_b")
        h_brown = st.slider("Brown (DAB) Range", 0, 60, DEFAULT_HUE["Brown"], key="h_br")

    mode = st.selectbox("Analysis Mode:", [
        "1. Area Fraction (Single Color)",
        "2. Cell Count (Nuclei)",
        "3. Colocalization (General)",
        "4. Spatial Distance",
        "5. Ratio Trend Analysis"
    ])
    st.divider()

    # Variable Initialization
    target_a, target_b = "Blue (DAPI)", "Red (RFP)"
    sens_a, sens_b = 30, 30 
    bright_a, bright_b = 30, 30 
    sens_common, bright_common = 30, 30
    min_size, bright_count = 50, 50
    sample_group = "Control"
    ratio_val = 0
    trend_metric = ""

    if mode.startswith("5."):
        st.markdown("### 🔢 Trend Conditions")
        trend_metric = st.radio("Metric:", ["Colocalization Rate", "Area Fraction"])
        ratio_val = st.number_input("Sort Value (Ratio/Conc):", value=0, step=10)
        ratio_label = st.text_input("Label (e.g., 160:40):", value=f"{ratio_val}%")
        sample_group = ratio_label 
        st.divider()
        st.markdown("#### Parameters")
        if trend_metric == "Colocalization Rate":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", COLORS, index=3, key="m5_ta") 
                sens_a = st.slider("Sens A", 5, 50, 30, key="m5_sa")
                bright_a = st.slider("Bright A", 0, 255, 30, key="m5_ba")
            with c2:
                target_b = st.selectbox("CH-B (Target):", COLORS, index=2, key="m5_tb") 
                sens_b = st.slider("Sens B", 5, 50, 30, key="m5_sb")
                bright_b = st.slider("Bright B", 0, 255, 60, key="m5_bb")
        else:
            target_a = st.selectbox("Target Color:", COLORS, index=2, key="m5_ta_area")
            sens_a = st.slider("Sensitivity", 5, 50, 30, key="m5_sa_area")
            bright_a = st.slider("Brightness", 0, 255, 60, key="m5_ba_area")
    else:
        sample_group = st.text_input("Group Name (e.g., Control):", value="Control")
        st.divider()
        if mode.startswith("1."):
            target_a = st.selectbox("Target Color:", COLORS, index=2)
            sens_a = st.slider("Sensitivity", 5, 50, 30)
            bright_a = st.slider("Brightness", 0, 255, 60)
        elif mode.startswith("2."):
            min_size = st.slider("Min Size (px)", 10, 500, 50)
            bright_count = st.slider("Brightness Threshold", 0, 255, 50)
        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", COLORS, index=3)
                sens_a = st.slider("Sens A", 5, 50, 30)
                bright_a = st.slider("Bright A", 0, 255, 30)
            with c2:
                target_b = st.selectbox("CH-B (Target):", COLORS, index=2)
                sens_b = st.slider("Sens B", 5, 50, 30)
                bright_b = st.slider("Bright B", 0, 255, 60)
        elif mode.startswith("4."):
            target_a = st.selectbox("Point A:", COLORS, index=3)
            target_b = st.selectbox("Point B:", COLORS, index=2)
            sens_common = st.slider("Color Sens", 5, 50, 30)
            bright_common = st.slider("Brightness", 0, 255, 60)

    st.divider()
    graph_type = st.radio("📊 Graph Type:", ["Box Plot", "Bar Plot", "Violin Plot", "Strip Plot"], index=1)
    st.divider()
    if st.button("Clear History & Reset"):
        st.session_state.analysis_history = []
        st.rerun()

def get_mask_dynamic(hsv_img, color_name, sens, bright_min):
    min_saturation = max(0, 40 - sens) # High sensitivity logic
    h, s, v = cv2.split(hsv_img)
    _, v_mask = cv2.threshold(v, bright_min, 255, cv2.THRESH_BINARY)
    color_mask = np.zeros_like(v_mask)
    
    # Map English color names to logic
    if "Red" in color_name:
        l1, h1 = h_red_l; l2, h2 = h_red_h
        color_mask = cv2.inRange(hsv_img, np.array([l1, min_saturation, 0]), np.array([h1, 255, 255])) | \
                     cv2.inRange(hsv_img, np.array([l2, min_saturation, 0]), np.array([h2, 255, 255]))
    elif "Green" in color_name:
        l, h = h_green
        color_mask = cv2.inRange(hsv_img, np.array([l, min_saturation, 0]), np.array([h, 255, 255]))
    elif "Blue" in color_name:
        l, h = h_blue
        color_mask = cv2.inRange(hsv_img, np.array([l, min_saturation, 0]), np.array([h, 255, 255]))
    elif "Brown" in color_name:
        l, h = h_brown
        color_mask = cv2.inRange(hsv_img, np.array([l, min_saturation, 0]), np.array([h, 255, 255]))
        
    return cv2.bitwise_and(color_mask, v_mask)

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

uploaded_files = st.file_uploader("Upload Images (Batch Processing)", type=["jpg", "png", "tif"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"received {len(uploaded_files)} images. Processing...")
    batch_results = []
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            val, unit, res_display = 0.0, "", img_rgb.copy()
            
            is_area = "Area" in mode or (mode.startswith("5.") and "Area" in trend_metric)
            is_count = "Count" in mode
            is_coloc = "Colocalization" in mode or (mode.startswith("5.") and "Colocalization" in trend_metric)
            is_dist = "Distance" in mode

            if is_area:
                mask = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                unit = "% Area"
                res_display = mask
                if show_mask: res_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif is_count:
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                _, otsu = cv2.threshold(cv2.GaussianBlur(gray,(5,5),0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                final = cv2.bitwise_and(th, otsu)
                cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                val, unit = len(valid), "cells"
                cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
            elif is_coloc:
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_a, bright_a)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_b, bright_b)
                coloc = cv2.bitwise_and(mask_a, mask_b)
                denom = cv2.countNonZero(mask_a)
                val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                unit = "% Coloc"
                res_display = cv2.merge([np.zeros_like(mask_a), mask_a, mask_b]) if not show_mask else cv2.merge([np.zeros_like(mask_a), mask_a, mask_b])
            elif is_dist:
                mask_a = get_mask_dynamic(img_hsv, target_a, sens_common, bright_common)
                mask_b = get_mask_dynamic(img_hsv, target_b, sens_common, bright_common)
                pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                if pts_a and pts_b:
                    val = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                else: val = 0
                unit = "px Dist"
                res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([np.zeros_like(mask_a), mask_a, mask_b]), 0.4, 0)
            
            if unit == "": unit = "(No Unit)"
            batch_results.append({
                "Group": sample_group, "Value": val, "Unit": unit,
                "Is_Trend": mode.startswith("5."), "Ratio_Value": ratio_val if mode.startswith("5.") else 0
            })
            with st.expander(f"📷 Img {i+1}: {val:.2f} {unit}", expanded=True):
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Result (Debug)" if show_mask else "Result", use_container_width=True)

    st.divider()
    if st.button(f"Integrate {len(batch_results)} results into '{sample_group}'", type="primary"):
        st.session_state.analysis_history.extend(batch_results)
        st.success(f"✅ Data added to '{sample_group}'")

if st.session_state.analysis_history:
    st.divider()
    st.header("📈 Integrated Report")
    df = pd.DataFrame(st.session_state.analysis_history)
    if df["Is_Trend"].any(): df = df.sort_values(by="Ratio_Value")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("white")
    teal_color = "#006d77" 
    
    # Graph logic matching English inputs
    if "Bar" in graph_type:
        sns.barplot(data=df, x="Group", y="Value", ax=ax, color=teal_color, capsize=.1, errorbar="sd", alpha=0.9)
    elif "Box" in graph_type:
        sns.boxplot(data=df, x="Group", y="Value", ax=ax, color=teal_color, width=0.5, fliersize=0)
    elif "Violin" in graph_type:
        sns.violinplot(data=df, x="Group", y="Value", ax=ax, color=teal_color, inner="quartile")
    elif "Strip" in graph_type:
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, size=10, color=teal_color, jitter=True)
        sns.pointplot(data=df, x="Group", y="Value", ax=ax, errorbar=None, color="firebrick", markers="_", scale=1.5, join=False)

    if "Bar" in graph_type or "Box" in graph_type or "Violin" in graph_type:
        sns.stripplot(data=df, x="Group", y="Value", ax=ax, color="black", size=6, jitter=True, alpha=0.7)

    y_label = df['Unit'].iloc[0]
    if "%" in y_label: y_label = "Positive rate [%]"
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    sns.despine()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    st.pyplot(fig)
    
    st.markdown("### 📊 Statistics Summary")
    summary = df.groupby("Group")["Value"].agg(['count', 'mean', 'std']).reset_index()
    summary.columns = ["Condition", "N", "Mean", "SD"]
    st.dataframe(summary.style.format({"Mean": "{:.2f}", "SD": "{:.2f}"}))
    st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "data.csv", "text/csv")
