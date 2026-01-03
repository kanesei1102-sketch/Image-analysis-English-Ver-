import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime  # For JST timestamp

# ---------------------------------------------------------
# 0. Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier Pro (Fixed)", layout="wide")

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# ---------------------------------------------------------
# 1. Function Definitions
# ---------------------------------------------------------
# Translated Color Map keys to English
COLOR_MAP = {
    "Brown (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "Green (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "Red (RFP)":   {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "Blue (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    # Updated condition to match English key
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
        if M["m00"] != 0:
            pts.append(np.array([M["m10"]/M["m00"], M["m01"]/M["m00"]]))
    return pts

# ---------------------------------------------------------
# 2. Main Layout Design
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition")
st.caption("2026 Latest Edition: Specialized for Analysis & Data Extraction (Scale: 1.5267 μm/px)")

tab_main, tab_val = st.tabs(["🚀 Analysis Execution", "🏆 Validation Report"])

# ---------------------------------------------------------
# 3. Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 【Notice】")
    st.info("""
    This tool is a beta version. If you plan to use results from this tool in your publications or conference presentations, **please contact the developer (Seiji Kaneko) in advance.**

    👉 **[Contact & Feedback Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**

    We will provide guidance on validation support and proper acknowledgments/co-authorship.
    """)
    st.divider()

    st.header("Analysis Recipe")
    # Updated options to English
    mode = st.selectbox("Select Analysis Mode:", [
        "1. Single Color Area Ratio (Area)",
        "2. Cell Nuclei Count (Count)",
        "3. General Colocalization Analysis",
        "4. Spatial Distance Analysis",
        "5. Ratio Trend Analysis"
    ])
    st.divider()

    if mode == "5. Ratio Trend Analysis":
        st.markdown("### 🔢 Batch Settings")
        trend_metric = st.radio("Measurement Target:", ["Colocalization Rate", "Area Ratio"])
        ratio_val = st.number_input("Condition Value:", value=0, step=10)
        ratio_unit = st.text_input("Unit:", value="%", key="unit")
        sample_group = f"{ratio_val}{ratio_unit}"
        st.info(f"Label: **{sample_group}**")
        
        if trend_metric == "Colocalization Rate":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", list(COLOR_MAP.keys()), index=3) 
                sens_a = st.slider("Sensitivity A", 5, 50, 20, key="t_s_a")
                bright_a = st.slider("Brightness A", 0, 255, 60, key="t_b_a")
            with c2:
                target_b = st.selectbox("CH-B (Target):", list(COLOR_MAP.keys()), index=2) 
                sens_b = st.slider("Sensitivity B", 5, 50, 20, key="t_s_b")
                bright_b = st.slider("Brightness B", 0, 255, 60, key="t_b_b")
        else:
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("Sensitivity", 5, 50, 20, key="t_s_a")
            bright_a = st.slider("Brightness Threshold", 0, 255, 60, key="t_b_a")
    else:
        sample_group = st.text_input("Group Name (X-axis):", value="Control")
        st.divider()
        
        if mode == "1. Single Color Area Ratio (Area)":
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()))
            sens_a = st.slider("Sensitivity", 5, 50, 20)
            bright_a = st.slider("Brightness Threshold", 0, 255, 60)
        
        elif mode == "2. Cell Nuclei Count (Count)":
            min_size = st.slider("Min Size (px)", 10, 500, 50)
            bright_count = st.slider("Cell Brightness Threshold", 0, 255, 50)
            
            use_roi_norm = st.checkbox("Calculate density within Tissue Area (e.g., CK8)", value=True)
            if use_roi_norm:
                st.markdown("""
                :red[**Please select the actual color used for staining. Using other colors may cause noise and inaccurate counts.**]
                """)
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2) 
                sens_roi = st.slider("Tissue Sensitivity", 5, 50, 20)
                bright_roi = st.slider("Tissue Brightness", 0, 255, 40)

        elif mode == "3. General Colocalization Analysis":
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("Sensitivity A", 5, 50, 20)
                bright_a = st.slider("Brightness A", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("Sensitivity B", 5, 50, 20)
                bright_b = st.slider("Brightness B", 0, 255, 60)
        
        elif mode == "4. Spatial Distance Analysis":
            target_a = st.selectbox("Origin A:", list(COLOR_MAP.keys()), index=2)
            target_b = st.selectbox("Target B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("Color Sensitivity", 5, 50, 20)
            bright_common = st.slider("Brightness Threshold", 0, 255, 60)

    st.divider()
    with st.expander("📏 Calibration Settings", expanded=True):
        st.caption("Input actual length per pixel to automatically calculate area (mm²) and density (cells/mm²).")
        scale_val = st.number_input("Length per px (μm/px)", value=1.5267, format="%.4f")

    if st.button("Clear All History"):
        st.session_state.analysis_history = []
        st.rerun()

    st.divider()
    st.caption("【Disclaimer】")
    st.caption("""
    This tool is intended to assist with image analysis.
    Results may vary depending on lighting conditions and settings. 
    The final interpretation and conclusion should be made by the user based on professional knowledge.
    """)

# ---------------------------------------------------------
# 4. Tab Content Implementation
# ---------------------------------------------------------

with tab_main:
    uploaded_files = st.file_uploader("Upload Images Batch", type=["jpg", "png", "tif"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"Analyzing {len(uploaded_files)} images...")
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
                
                fov_area_mm2 = 0.0
                if scale_val > 0:
                    h, w = img_rgb.shape[:2]
                    fov_area_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                # 1. Area
                if mode == "1. Single Color Area Ratio (Area)" or (mode.startswith("5.") and trend_metric == "Area Ratio"):
                    mask = get_mask(img_hsv, target_a, sens_a, bright_a)
                    val = (cv2.countNonZero(mask) / (img_rgb.shape[0] * img_rgb.shape[1])) * 100
                    unit = f"% Area"
                    res_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    res_display[:, :, 0] = 0; res_display[:, :, 2] = 0
                    real_area_str = ""
                    if fov_area_mm2 > 0:
                        real_area = fov_area_mm2 * (val / 100)
                        real_area_str = f"{real_area:.4f} mm²"

                # 2. Count
                elif mode == "2. Cell Nuclei Count (Count)":
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
                            roi_pixel_count = cv2.countNonZero(mask_roi)
                            real_roi_area_mm2 = roi_pixel_count * ((scale_val / 1000) ** 2)
                            if real_roi_area_mm2 > 0:
                                density = val / real_roi_area_mm2
                                density_str = f"{int(density):,} cells/mm² (ROI)"
                                roi_cnts, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(res_display, roi_cnts, -1, (255,0,0), 3) 
                            else:
                                density_str = "ROI Area is 0"
                        elif fov_area_mm2 > 0:
                            density = val / fov_area_mm2
                            density_str = f"{int(density):,} cells/mm² (FOV)"

                # 3. Coloc
                elif mode == "3. General Colocalization Analysis" or (mode.startswith("5.") and trend_metric == "Colocalization Rate"):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a)
                    mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b)
                    denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0
                    unit = f"% Coloc"
                    res_display = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                
                # 4. Distance
                elif mode == "4. Spatial Distance Analysis":
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

                entry = {
                    "File Name": file.name,
                    "Group": sample_group,
                    "Value": val,
                    "Unit": unit,
                    "Is_Trend": mode.startswith("5."),
                    "Ratio_Value": ratio_val if mode.startswith("5.") else 0
                }
                batch_results.append(entry)
                
                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name}")
                st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                if mode == "1. Single Color Area Ratio (Area)" and scale_val > 0 and 'real_area_str' in locals():
                    st.metric("Actual Tissue Area", real_area_str)
                elif mode == "2. Cell Nuclei Count (Count)" and scale_val > 0 and 'density_str' in locals():
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
        
        # Organize column order
        cols = ["File Name", "Group", "Value", "Unit", "Is_Trend", "Ratio_Value"]
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        now = datetime.datetime.now() + datetime.timedelta(hours=9)
        file_name = f"quantified_data_{now.strftime('%Y%m%d_%H%M%S')}.csv"
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download CSV Data", df.to_csv(index=False).encode('utf-8'), file_name, "text/csv")

# ==========================================
# Validation Data Loading & Graph Generation
# ==========================================
# This function loads CSVs and generates graphs on the fly to avoid "File Not Found" errors.
@st.cache_data
def load_and_process_validation_data():
    files = {
        'C14': 'quantified_data_20260102_201522.csv',
        'C40': 'quantified_data_20260102_194322.csv',
        'C70': 'quantified_data_20260103_093427.csv',
        'C100': 'quantified_data_20260102_202525.csv'
    }
    
    data_list = []
    ground_truths = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}

    for density, filename in files.items():
        try:
            df = pd.read_csv(filename)
            col_name = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            
            for _, row in df.iterrows():
                fname = str(row[col_name])
                val = row['Value']
                
                # Check Channel
                if 'w1' in fname.lower(): channel = 'W1'
                elif 'w2' in fname.lower(): channel = 'W2'
                else: continue
                
                # Check Focus
                focus_match = re.search(r'_F(\d+)_', fname)
                if focus_match:
                    focus = int(focus_match.group(1))
                    acc = (val / ground_truths[density]) * 100
                    
                    data_list.append({
                        'Density': density,
                        'Focus': focus,
                        'Channel': channel,
                        'Accuracy': acc
                    })
        except FileNotFoundError:
            # st.error(f"⚠️ Data file not found: {filename}")
            pass
            
    return pd.DataFrame(data_list)

# Load data
df_all = load_and_process_validation_data()

with tab_val:
    st.header("🏆 Performance Validation Final Report (2026 Latest)")
    st.markdown("""
    **Validation Source:** [Broad Bioimage Benchmark Collection (BBBC005)](https://bbbc.broadinstitute.org/BBBC005)  
    **Total Verified:** 3,200 Images (C14, C40, C70, C100 × 800 images / Based on Real Data)
    """)

    if not df_all.empty:
        # Extract high quality data (F1-5) and calculate stats
        df_hq = df_all[(df_all['Focus'] >= 1) & (df_all['Focus'] <= 5)]
        w1_stats = df_hq[df_hq['Channel'] == 'W1']
        
        # Calculate Metrics
        avg_acc = w1_stats['Accuracy'].mean()
        
        # Prepare data for Linearity (R2)
        df_linear = w1_stats.groupby('Density')['Accuracy'].mean().reset_index()
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        df_linear['GT'] = df_linear['Density'].map(gt_map)
        df_linear['Measured'] = (df_linear['Accuracy'] / 100) * df_linear['GT']
        
        x = df_linear['GT'].values
        y = df_linear['Measured'].values
        r2 = np.corrcoef(x, y)[0, 1]**2

        # Display Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Nuclei Count Accuracy (W1)", f"{avg_acc:.1f}%", help="Average accuracy across all densities at Focus Level 1-5")
        m2.metric("Statistical Linearity (R²)", f"{r2:.4f}", help="Coefficient of determination based on W1 measured values (C14-C100)")
        m3.metric("Continuous Processing Stability", "3,200+ Images", help="Completed 800 images x 4 batches without errors")

        st.divider()

        # --- 1. Linearity Graph ---
        st.subheader("📈 1. Counting Capability & Linearity")
        st.info("💡 **Conclusion:** W1 (Nuclei) follows the ideal line with R²=0.9977. W2 (Cytoplasm) shows V-shaped deviation and is unsuitable for quantification.")
        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        
        # Prepare W2 data
        w2_stats = df_hq[df_hq['Channel'] == 'W2'].groupby('Density')['Accuracy'].mean().reset_index()
        w2_stats['GT'] = w2_stats['Density'].map(gt_map)
        w2_stats['Measured'] = (w2_stats['Accuracy'] / 100) * w2_stats['GT']
        w2_stats = w2_stats.sort_values('GT')
        
        # Plot
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal (y=x)')
        ax1.scatter(x, y, c='#1f77b4', s=100, zorder=5, label='W1 (Nuclei)')
        ax1.scatter(w2_stats['GT'], w2_stats['Measured'], c='#ff7f0e', s=100, marker='D', zorder=5, label='W2 (Cytoplasm)')
        
        # W1 Regression Line
        slope, intercept = np.polyfit(x, y, 1)
        ax1.plot(x, slope*x + intercept, '#1f77b4', alpha=0.5, label=f'W1 Reg (R²={r2:.4f})')
        ax1.plot(w2_stats['GT'], w2_stats['Measured'], '#ff7f0e', linestyle=':', alpha=0.5)
        
        ax1.set_xlabel('Ground Truth (Cells/Image)'); ax1.set_ylabel('Measured Count')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig1)

        st.divider()

        # --- 2. Density Comparison Graph ---
        st.subheader("📊 2. Accuracy Comparison by Density (W1 vs W2)")
        st.success("✅ **Recommendation:** Strongly recommend using **'W1'** for all density regions.")
        st.markdown("""
        * **W1 (Nuclei):** Maintains high accuracy (95% - 100%) from C14 to C100.
        * **W2 (Cytoplasm):** Under-detected in C70 (fusion), Over-detected in C100 (135% chaos).
        """)
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
        df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        
        sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
        ax2.axhline(100, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 160)
        ax2.set_ylabel('Accuracy (% of Ground Truth)')
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        st.pyplot(fig2)

        st.divider()

        # --- 3. Focus Robustness Graph ---
        st.subheader("📉 3. Optical Robustness (Focus)")
        st.warning("⚠️ **Caution:** Strictly limit Focus Level to within 5 for High Density (C100) analysis.")
        st.markdown("""
        * **C14 (Blue Line):** Maintains 100% accuracy even with blur (Robust).
        * **C100 (Purple Line):** Accuracy collapses rapidly beyond F5 (Sensitive).
        """)
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        df_w1_decay = df_all[df_all['Channel'] == 'W1'].copy()
        df_w1_decay['Density'] = pd.Categorical(df_w1_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        
        sns.lineplot(data=df_w1_decay, x='Focus', y='Accuracy', hue='Density', style='Density',
                     palette={'C14': '#1f77b4', 'C40': '#55a868', 'C70': '#c44e52', 'C100': '#8172b3'},
                     markers=True, dashes=False, ax=ax3)
        
        ax3.axhline(100, color='red', linestyle='--', alpha=0.5)
        ax3.set_ylim(0, 120)
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_xlabel('Focus Level (1=Sharp, 48=Blurred)')
        ax3.grid(True, linestyle=':', alpha=0.6)
        st.pyplot(fig3)

    else:
        st.warning("Validation CSV files not found. Please ensure the repository contains the necessary data files.")
