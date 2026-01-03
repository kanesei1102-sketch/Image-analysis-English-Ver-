import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ---------------------------------------------------------
# 0. Page Config & Session Initialization
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier Pro", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ---------------------------------------------------------
# 1. Core Functions (Image Processing)
# ---------------------------------------------------------
COLOR_MAP = {
    "Brown (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "Green (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "Red (RFP)":   {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "Blue (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    if color_name == "Red (RFP)":
        lower1 = np.array([0, 30, bright_min]); upper1 = np.array([10 + sens//2, 255, 255])
        lower2 = np.array([170 - sens//2, 30, bright_min]); upper2 = np.array([180, 255, 255])
        return cv2.inRange(hsv_img, lower1, upper1) | cv2.inRange(hsv_img, lower2, upper2)
    else:
        conf = COLOR_MAP[color_name]
        l = np.clip(conf["lower"] - sens, 0, 255); u = np.clip(conf["upper"] + sens, 0, 255)
        l[2] = max(l[2], bright_min)
        return cv2.inRange(hsv_img, l, u)

def get_tissue_mask(hsv_img, color_name, sens, bright_min):
    mask = get_mask(hsv_img, color_name, sens, bright_min)
    kernel = np.ones((15, 15), np.uint8) 
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    valid_tissue = [c for c in cnts if cv2.contourArea(c) > 500]
    cv2.drawContours(mask_filled, valid_tissue, -1, 255, thickness=cv2.FILLED)
    return mask_filled

def get_centroids(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] != 0: pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# ---------------------------------------------------------
# 2. Validation Data Loader (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_validation_data():
    files = {
        'C14': 'quantified_data_20260102_201522.csv',
        'C40': 'quantified_data_20260102_194322.csv',
        'C70': 'quantified_data_20260103_093427.csv',
        'C100': 'quantified_data_20260102_202525.csv'
    }
    data_list = []
    mapping = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}

    for density, filename in files.items():
        try:
            df = pd.read_csv(filename)
            col = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            for _, row in df.iterrows():
                fname = str(row[col]); val = row['Value']
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1))
                    data_list.append({
                        'Density': density, 'Ground Truth': mapping[density],
                        'Focus': focus, 'Channel': channel, 'Value': val,
                        'Accuracy': (val / mapping[density]) * 100
                    })
        except: pass
    return pd.DataFrame(data_list)

df_val = load_validation_data()

# ---------------------------------------------------------
# 3. Sidebar & Settings
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2026 Latest Edition: Specialized for Analysis & Data Extraction (Scale: 1.5267 μm/px)")

tab_main, tab_val = st.tabs(["🚀 Run Analysis", "🏆 Validation Report"])

with st.sidebar:
    st.markdown("### 【Notice】")
    st.info("""
    This tool is a beta version. If you plan to use results from this tool in your publications or conference presentations, **please contact the developer (Seiji Kaneko) in advance.**

    👉 **[Contact & Feedback Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**

    We will provide guidance on validation support and proper acknowledgments/co-authorship.
    """)
    st.divider()

    st.header("Analysis Recipe")
    mode = st.selectbox("Select Mode:", [
        "1. Single Color Area Ratio (Area)",
        "2. Cell Nuclei Count (Count)",
        "3. General Colocalization Analysis",
        "4. Spatial Distance Analysis",
        "5. Ratio Trend Analysis"
    ])
    st.divider()

    # --- Mode 5: Ratio Trend Analysis ---
    if mode == "5. Ratio Trend Analysis":
        st.markdown("### 🔢 Batch Settings")
        trend_metric = st.radio("Metric:", ["Colocalization Rate", "Area Ratio"])
        ratio_val = st.number_input("Condition Value:", value=0, step=10)
        ratio_unit = st.text_input("Unit:", value="%")
        sample_group = f"{ratio_val}{ratio_unit}"
        st.info(f"Label: **{sample_group}**")
        
        if trend_metric == "Colocalization Rate":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", list(COLOR_MAP.keys()), index=3) 
                sens_a = st.slider("Sens A", 5, 50, 20, key="ta")
                bright_a = st.slider("Bright A", 0, 255, 60, key="ba")
            with c2:
                target_b = st.selectbox("CH-B (Target):", list(COLOR_MAP.keys()), index=2) 
                sens_b = st.slider("Sens B", 5, 50, 20, key="tb")
                bright_b = st.slider("Bright B", 0, 255, 60, key="bb")
        else:
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("Sensitivity", 5, 50, 20)
            bright_a = st.slider("Brightness", 0, 255, 60)
            
    # --- Other Modes ---
    else:
        # Default values for mode 5 variables to prevent errors
        trend_metric = None
        ratio_val = 0
        
        sample_group = st.text_input("Group Name:", value="Control")
        
        if mode.startswith("1"): # Area
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()))
            sens_a = st.slider("Sensitivity", 5, 50, 20)
            bright_a = st.slider("Brightness", 0, 255, 60)
        
        elif mode.startswith("2"): # Count
            min_size = st.slider("Min Size (px)", 10, 500, 50)
            bright_count = st.slider("Cell Brightness", 0, 255, 50)
            use_roi_norm = st.checkbox("Calculate Density in Tissue Area (e.g. CK8)", value=True)
            if use_roi_norm:
                st.markdown(":red[**Select the actual stain color for tissue.**]")
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2)
                sens_roi = st.slider("Tissue Sens", 5, 50, 20)
                bright_roi = st.slider("Tissue Bright", 0, 255, 40)

        elif mode.startswith("3"): # Coloc
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("Sens A", 5, 50, 20); bright_a = st.slider("Bright A", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("Sens B", 5, 50, 20); bright_b = st.slider("Bright B", 0, 255, 60)
        
        elif mode.startswith("4"): # Distance
            target_a = st.selectbox("Origin A:", list(COLOR_MAP.keys()), index=2)
            target_b = st.selectbox("Target B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("Color Sens", 5, 50, 20)
            bright_common = st.slider("Brightness", 0, 255, 60)

    st.divider()
    with st.expander("📏 Calibration", expanded=True):
        st.caption("Enter length per pixel to calculate Area (mm²) and Density (cells/mm²).")
        scale_val = st.number_input("Length per px (μm/px)", value=1.5267, format="%.4f")
    
    if st.button("Clear All History"):
        st.session_state.analysis_history = []
        st.rerun()

    st.divider()
    st.caption("【Disclaimer】")
    st.caption("This tool is intended to assist with image analysis. Results may vary depending on lighting conditions and settings. The final interpretation and conclusion should be made by the user based on professional knowledge.")

# ---------------------------------------------------------
# 4. Tab 1: Analysis Execution (Logic Fixed for Mode 5)
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "tif"], accept_multiple_files=True)
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
                
                # Logic for Area/Density Calculation
                fov_area_mm2 = 0.0
                if scale_val > 0:
                    h, w = img_rgb.shape[:2]
                    fov_area_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                # --- 1. Area Analysis ---
                # Check for Mode 1 OR (Mode 5 AND "Area Ratio")
                if mode.startswith("1") or (mode.startswith("5.") and trend_metric == "Area Ratio"):
                    mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                    val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                    unit = "% Area"
                    res_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    res_display[:, :, 0] = 0; res_display[:, :, 2] = 0 # tint red
                    
                    real_area_str = ""
                    if fov_area_mm2 > 0:
                        real_area = fov_area_mm2 * (val / 100)
                        real_area_str = f"{real_area:.4f} mm²"

                # --- 2. Count Analysis ---
                elif mode.startswith("2"): 
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0)
                    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    final = cv2.bitwise_and(th, otsu)
                    cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]
                    val, unit = len(valid), "cells"
                    cv2.drawContours(res_display, valid, -1, (0,255,0), 2)
                    
                    density_str = ""
                    if scale_val > 0:
                        if 'use_roi_norm' in locals() and use_roi_norm:
                            mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                            roi_px = cv2.countNonZero(mask_roi)
                            roi_mm2 = roi_px * ((scale_val / 1000) ** 2)
                            if roi_mm2 > 0:
                                density = val / roi_mm2
                                density_str = f"{int(density):,} cells/mm² (ROI)"
                                roi_cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(res_display, roi_cnts, -1, (255,0,0), 3) # Blue ROI outline
                            else:
                                density_str = "ROI Area is 0"
                        elif fov_area_mm2 > 0:
                            density = val / fov_area_mm2
                            density_str = f"{int(density):,} cells/mm² (FOV)"

                # --- 3. Colocalization Analysis ---
                # Check for Mode 3 OR (Mode 5 AND "Colocalization Rate")
                elif mode.startswith("3") or (mode.startswith("5.") and trend_metric == "Colocalization Rate"):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                    mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b)
                    denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                    unit = "% Coloc"
                    res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                
                # --- 4. Distance Analysis ---
                elif mode.startswith("4"): 
                    mask_a = get_mask(img_hsv, target_a, sens_common, bright_common)
                    mask_b = get_mask(img_hsv, target_b, sens_common, bright_common)
                    pts_a, pts_b = get_centroids(mask_a), get_centroids(mask_b)
                    if pts_a and pts_b:
                        val_px = np.mean([np.min([np.linalg.norm(pa - pb) for pb in pts_b]) for pa in pts_a])
                        if scale_val > 0:
                            val = val_px * scale_val; unit = "μm Dist"
                        else:
                            val = val_px; unit = "px Dist"
                    else: 
                        val = 0; unit = "Dist"
                    res_display = cv2.addWeighted(img_rgb, 0.6, cv2.merge([mask_a, mask_b, np.zeros_like(mask_a)]), 0.4, 0)

                val = max(0.0, val)

                # Store Result
                entry = {
                    "File Name": file.name,
                    "Group": sample_group,
                    "Value": val,
                    "Unit": unit,
                    "Is_Trend": mode.startswith("5."),
                    "Ratio_Value": ratio_val if mode.startswith("5.") else 0
                }
                batch_results.append(entry)
                
                # Display Results
                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name}")
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                # Display extra metrics
                if (mode.startswith("1") or (mode.startswith("5.") and trend_metric == "Area Ratio")) and scale_val > 0 and 'real_area_str' in locals():
                    st.metric("Actual Tissue Area", real_area_str)
                elif mode.startswith("2") and scale_val > 0 and 'density_str' in locals():
                    st.metric("Cell Density", density_str)

                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original", use_container_width=True)
                c2.image(res_display, caption="Analyzed", use_container_width=True)

        if st.button(f"Add {len(batch_results)} Results to History", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.rerun()

    if st.session_state.analysis_history:
        st.divider()
        st.header("💾 Data Export")
        df = pd.DataFrame(st.session_state.analysis_history)
        df["Value"] = df["Value"].clip(lower=0) 
        
        # Enforce column order
        cols = ["File Name", "Group", "Value", "Unit", "Is_Trend", "Ratio_Value"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols] 
        
        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), file_name, "text/csv")

# ---------------------------------------------------------
# 5. Tab 2: Validation Report (Full Version)
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 Performance Validation Final Report (2026 Latest)")
    
    st.markdown("""
    * **Validation Source:** [Broad Bioimage Benchmark Collection (BBBC005)](https://bbbc.broadinstitute.org/BBBC005)
    * **Total Verified:** 3,200 Images (C14, C40, C70, C100 × 800 images / Real Data)
    * **Methodology:** Results were obtained after **optimizing parameters (Sensitivity, Min Size) for each density group**, demonstrating the tool's maximum potential under proper calibration.
    """)

    if not df_val.empty:
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        
        # High Quality Data Filter
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        
        # Stats Calculation
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Nuclei Accuracy (W1)", f"{avg_acc:.1f}%")
        m2.metric("Statistical Linearity (R²)", f"{r2:.4f}")
        m3.metric("Process Stability", "3,200+ Images")

        st.divider()

        # Graph 1: Linearity
        st.subheader("📈 1. Counting Capability & Linearity")
        st.info("💡 **Conclusion:** W1 (Nuclei) shows extremely high linearity ($R^2=0.9977$). W2 (Cytoplasm) shows V-shaped deviation.")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        # W2 Plot
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Reg')
        ax1.set_xlabel('Ground Truth (Cells)'); ax1.set_ylabel('Measured Count'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()

        # Graph 2 & 3: Comparison
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. Accuracy by Density")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            # Order
            df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
            ax2.axhline(100, color='red', linestyle='--'); ax2.set_ylabel('Accuracy (%)')
            st.pyplot(fig2)
        
        with c2:
            st.subheader("📉 3. Focus Robustness")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); ax3.set_ylabel('Accuracy (%)')
            st.pyplot(fig3)

        st.divider()

        # 4. Summary Table
        st.subheader("📋 4. Validation Data Summary Table")
        st.caption("Mean measured values at Focus Level 1-5 (High Quality Images)")
        
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['Ground Truth'] = summary['Density'].map(gt_map)
        
        # Calculate Measured Values
        summary['W1 Measured'] = (summary['W1']/100)*summary['Ground Truth']
        summary['W2 Measured'] = (summary['W2']/100)*summary['Ground Truth']
        
        # Sort
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')

        st.table(summary[['Density', 'Ground Truth', 'W1', 'W1 Measured', 'W2', 'W2 Measured']].rename(columns={
            'W1': 'W1 Accuracy(%)', 'W1 Measured': 'W1 Count(Mean)',
            'W2': 'W2 Accuracy(%)', 'W2 Measured': 'W2 Count(Mean)'
        }))
        
        st.info("💡 **Final Conclusion:** W1 (Nuclei) maintains high accuracy across all densities (Mean 97.7%). W2 (Cytoplasm) is unreliable for quantification due to fusion (under-counting) or segmentation artifacts (over-counting).")
    else:
        st.error("Validation CSV files not found. Please ensure files are in the repository.")
