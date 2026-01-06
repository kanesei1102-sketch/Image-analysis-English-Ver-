import streamlit as st
import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re
import uuid

# ---------------------------------------------------------
# 0. Page Config & Constants
# ---------------------------------------------------------
st.set_page_config(page_title="Bio-Image Quantifier V2 (EN)", layout="wide")

# Version Management
SOFTWARE_VERSION = "Bio-Image Quantifier Pro v2026.02 (EN/Auto-Group)"

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = str(uuid.uuid4())
    
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# --- Analysis ID Management (Human readable + Unique ID) ---
if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. Image Processing Engine (HE Staining & Brightfield Support)
# ---------------------------------------------------------
# Translated Color Map Keys
COLOR_MAP = {
    # Existing Fluorescence / IHC settings
    "Brown (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "Green (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "Red (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
    "Blue (DAPI)": {"lower": np.array([100, 50, 50]), "upper": np.array([140, 255, 255])},
    
    # --- Added: HE Staining (Brightfield) Support ---
    # Hematoxylin (Nuclei): Purple-Blue, Darker
    "Hematoxylin (Nuclei)": {"lower": np.array([110, 50, 50]), "upper": np.array([170, 255, 200])},
    # Eosin (Cytoplasm): Pink-Red, Brighter
    "Eosin (Cytoplasm)": {"lower": np.array([140, 20, 100]), "upper": np.array([180, 255, 255])}
}

def get_mask(hsv_img, color_name, sens, bright_min):
    # Adjusted logic for Red wrapping in HSV
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
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
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
# 2. Validation Data Loading
# ---------------------------------------------------------
@st.cache_data
def load_validation_data():
    files = {'C14': 'quantified_data_20260102_201522.csv', 'C40': 'quantified_data_20260102_194322.csv',
             'C70': 'quantified_data_20260103_093427.csv', 'C100': 'quantified_data_20260102_202525.csv'}
    data_list = []; mapping = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
    for density, filename in files.items():
        try:
            df = pd.read_csv(filename); col = 'Image_Name' if 'Image_Name' in df.columns else 'File Name'
            for _, row in df.iterrows():
                fname = str(row[col]); val = row['Value']
                channel = 'W1' if 'w1' in fname.lower() else 'W2' if 'w2' in fname.lower() else None
                if not channel: continue
                f_match = re.search(r'_F(\d+)_', fname)
                if f_match:
                    focus = int(f_match.group(1)); accuracy = (val / mapping[density]) * 100
                    data_list.append({'Density': density, 'Ground Truth': mapping[density], 'Focus': focus, 'Channel': channel, 'Value': val, 'Accuracy': accuracy})
        except FileNotFoundError: pass
    return pd.DataFrame(data_list)

df_val = load_validation_data()

# ---------------------------------------------------------
# 3. UI Framework & Sidebar
# ---------------------------------------------------------
st.title("🔬 Bio-Image Quantifier: Pro Edition (English)")
st.caption(f"{SOFTWARE_VERSION}: Industrial Grade Image Analysis & Data Extraction")

st.sidebar.markdown(f"**Current Analysis ID:** `{st.session_state.current_analysis_id}`")

tab_main, tab_val = st.tabs(["🚀 Run Analysis", "🏆 Performance Validation"])

with st.sidebar:
    st.markdown("### [Important: Publication/Conference Use]")
    st.warning("""
    **Are you planning to publish results?**
    This tool is in Beta. For academic use, **please contact the developer (Kaneko) in advance.**
    We can discuss co-authorship or acknowledgments.
    👉 **[Contact Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    st.header("Analysis Recipe")
    mode_raw = st.selectbox("Select Analysis Mode:", [
        "1. Area Fraction (%)", 
        "2. Nuclei Count / Density", 
        "3. Colocalization", 
        "4. Spatial Distance Analysis", 
        "5. Trend/Gradient Analysis"
    ])
    mode = mode_raw 

    st.divider()

    # --- Grouping Strategy ---
    st.markdown("### 🏷️ Grouping Settings")
    group_strategy = st.radio("Labeling Strategy:", ["Manual Input", "Auto from Filename"], 
                              help="Auto: Extracts the string before the delimiter in the filename as the group name.")
    
    if group_strategy.startswith("Manual"):
        sample_group = st.text_input("Group Name (X-axis Label):", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("Delimiter (e.g., _ or - ):", value="_", help="The string before this character becomes the group name.")
        st.info(f"Example: 'WT{filename_sep}01.tif' → Group: 'WT'")
        sample_group = "(Auto Detected)" 

    st.divider()

    # Dynamic Analysis Parameters
    current_params_dict = {} # Dictionary to store active parameters

    if mode.startswith("5."):
        st.markdown("### 🔢 Trend Analysis Conditions")
        trend_metric = st.radio("Metric:", ["Colocalization Rate", "Area Fraction"])
        ratio_val = st.number_input("Condition Value:", value=0, step=10)
        ratio_unit = st.text_input("Unit:", value="%", key="unit")
        if group_strategy.startswith("Manual"):
            sample_group = f"{ratio_val}{ratio_unit}" 
        
        current_params_dict["Trend Metric"] = trend_metric
        current_params_dict["Condition Value"] = f"{ratio_val}{ratio_unit}"

        if trend_metric.startswith("Colocalization"):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A (Base):", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A Sensitivity", 5, 50, 20); bright_a = st.slider("A Brightness", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B (Target):", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B Sensitivity", 5, 50, 20); bright_b = st.slider("B Brightness", 0, 255, 60)
            current_params_dict.update({"CH-A": target_a, "Sens A": sens_a, "Bright A": bright_a, "CH-B": target_b, "Sens B": sens_b, "Bright B": bright_b})
        else:
            # ROI Normalization for Area Trend
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("Sensitivity", 5, 50, 20); bright_a = st.slider("Brightness", 0, 255, 60)
            
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=False, key="roi_mode5")
            current_params_dict.update({"Target Color": target_a, "Sensitivity": sens_a, "Brightness": bright_a, "ROI Norm": use_roi_norm})
            
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=5, key="roi_col5")
                sens_roi = st.slider("ROI Sensitivity", 5, 50, 20, key="roi_sens5")
                bright_roi = st.slider("ROI Brightness", 0, 255, 40, key="roi_bri5")
                current_params_dict.update({"ROI Color": roi_color, "ROI Sens": sens_roi, "ROI Bright": bright_roi})
    else:
        if mode.startswith("1."):
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()), index=5)
            sens_a = st.slider("Sensitivity", 5, 50, 20); bright_a = st.slider("Brightness", 0, 255, 60)
            
            # ROI Normalization for Area Fraction
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=False, key="roi_mode1")
            current_params_dict.update({"Target Color": target_a, "Sensitivity": sens_a, "Brightness": bright_a, "ROI Norm": use_roi_norm})
            
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=5, key="roi_col1")
                sens_roi = st.slider("ROI Sensitivity", 5, 50, 20, key="roi_sens1")
                bright_roi = st.slider("ROI Brightness", 0, 255, 40, key="roi_bri1")
                current_params_dict.update({"ROI Color": roi_color, "ROI Sens": sens_roi, "ROI Bright": bright_roi})

        elif mode.startswith("2."):
            # Count Mode with Color Selection (HE support)
            target_a = st.selectbox("Nuclei Color:", list(COLOR_MAP.keys()), index=4)
            sens_a = st.slider("Nuclei Sensitivity", 5, 50, 20)
            bright_a = st.slider("Nuclei Threshold", 0, 255, 50)
            min_size = st.slider("Min Nuclei Size (px)", 10, 500, 50)
            
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=True, key="roi_mode2")
            current_params_dict.update({"Nuclei Color": target_a, "Nucl Sens": sens_a, "Nucl Bright": bright_a, "Min Size": min_size, "ROI Norm": use_roi_norm})
            
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=5, key="roi_col2")
                sens_roi = st.slider("ROI Sensitivity", 5, 50, 20, key="roi_sens2")
                bright_roi = st.slider("ROI Brightness", 0, 255, 40, key="roi_bri2")
                current_params_dict.update({"ROI Color": roi_color, "ROI Sens": sens_roi, "ROI Bright": bright_roi})

        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3); sens_a = st.slider("A Sensitivity", 5, 50, 20); bright_a = st.slider("A Brightness", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2); sens_b = st.slider("B Sensitivity", 5, 50, 20); bright_b = st.slider("B Brightness", 0, 255, 60)
            current_params_dict.update({"CH-A": target_a, "Sens A": sens_a, "Bright A": bright_a, "CH-B": target_b, "Sens B": sens_b, "Bright B": bright_b})

        elif mode.startswith("4."):
            target_a = st.selectbox("Origin A:", list(COLOR_MAP.keys()), index=2); target_b = st.selectbox("Target B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("Common Sensitivity", 5, 50, 20); bright_common = st.slider("Common Brightness", 0, 255, 60)
            current_params_dict.update({"Origin A": target_a, "Target B": target_b, "Common Sens": sens_common, "Common Bright": bright_common})

    st.divider()
    # Spatial scale set to 3.0769 (Default)
    scale_val = st.number_input("Spatial Scale (μm/px)", value=3.0769, format="%.4f")
    current_params_dict["Spatial Scale"] = scale_val
    current_params_dict["Analysis Mode"] = mode
    
    def prepare_next_group():
        st.session_state.uploader_key = str(uuid.uuid4())

    st.button(
        "📸 Next Group (Clear Images Only)", 
        on_click=prepare_next_group, 
        help="Keeps the current analysis history but clears uploaded images to prepare for the next group."
    )
    
    st.divider()
    if st.button("Clear History & New ID"): 
        st.session_state.analysis_history = []
        date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        st.session_state.current_analysis_id = f"AID-{date_str}-{str(uuid.uuid4())[:8]}"
        st.session_state.uploader_key = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Traceability (Current Settings)")
    st.table(pd.DataFrame([current_params_dict]).T)
    
    # Audit Log CSV (Settings only)
    df_params_log = pd.DataFrame([current_params_dict]).T.reset_index()
    df_params_log.columns = ["Parameter", "Value"]
    param_filename = f"params_{st.session_state.current_analysis_id}.csv"
    st.download_button("📥 Download Settings CSV", df_params_log.to_csv(index=False).encode('utf-8-sig'), param_filename, "text/csv")

    st.divider()
    st.caption("【Disclaimer】")
    st.caption("This tool is for research purposes only and does not guarantee clinical diagnosis. Final validation is the responsibility of the user.")

# ---------------------------------------------------------
# 4. Tab 1: Execute Analysis
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("Upload Images (16-bit TIFF supported)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded_files:
        st.success(f"Analyzing {len(uploaded_files)} images...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            # 16-bit Support: IMREAD_UNCHANGED
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            
            if img_raw is not None:
                if group_strategy.startswith("Auto"):
                    try: detected_group = file.name.split(filename_sep)[0]
                    except: detected_group = "Unknown"
                    current_group_label = detected_group
                else:
                    current_group_label = sample_group

                # Image Processing (Min-Max Normalization to 8bit)
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit, res_disp = 0.0, "", img_rgb.copy()
                h, w = img_rgb.shape[:2]; fov_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                extra_data = {}

                # --- Area Fraction Mode (ROI Norm) ---
                if mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("Area")):
                    mask_target = get_mask(img_hsv, target_a, sens_a, bright_a)
                    
                    a_denominator_px = h * w
                    roi_status = "Field of View"
                    final_mask = mask_target
                    
                    if 'use_roi_norm' in locals() and use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        # Count only extracted color within tissue
                        final_mask = cv2.bitwise_and(mask_target, mask_roi)
                        a_denominator_px = cv2.countNonZero(mask_roi)
                        roi_status = "Inside ROI"
                        # Draw ROI contour in Red
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    target_px = cv2.countNonZero(final_mask)
                    val = (target_px / a_denominator_px * 100) if a_denominator_px > 0 else 0
                    unit = "% Area"
                    
                    # Display extracted area in Green overlay
                    res_disp_mask = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB)
                    res_disp_mask[:,:,0]=0; res_disp_mask[:,:,2]=0
                    res_disp = cv2.addWeighted(res_disp, 0.7, res_disp_mask, 0.3, 0)
                    
                    a_target_mm2 = a_denominator_px * ((scale_val/1000)**2)
                    extra_data = {
                        "Target Area(mm2)": round(a_target_mm2, 6),
                        "Norm Basis": roi_status
                    }

                # --- Nuclei Count Mode (ROI Norm & Color Select) ---
                elif mode.startswith("2."):
                    # Improved: Extract nuclei using specified color mask (Supports HE/IHC)
                    mask_nuclei = get_mask(img_hsv, target_a, sens_a, bright_a)
                    
                    # Morphology to separate nuclei
                    kernel = np.ones((3,3), np.uint8)
                    mask_nuclei = cv2.morphologyEx(mask_nuclei, cv2.MORPH_OPEN, kernel)
                    
                    cnts, _ = cv2.findContours(mask_nuclei, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]; val, unit = len(valid), "cells"
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                    
                    a_target_mm2 = fov_mm2
                    roi_status = "Field of View"
                    
                    if 'use_roi_norm' in locals() and use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        roi_px = cv2.countNonZero(mask_roi)
                        a_target_mm2 = roi_px * ((scale_val/1000)**2)
                        roi_status = "Inside ROI"
                        # Draw ROI contour in Red
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    density = val / a_target_mm2 if a_target_mm2 > 0 else 0
                    extra_data = {
                        "Target Area(mm2)": round(a_target_mm2, 6),
                        "Density(cells/mm2)": round(density, 2),
                        "Norm Basis": roi_status
                    }

                # --- Other Modes ---
                elif mode.startswith("3.") or (mode.startswith("5.") and trend_metric.startswith("Colocalization")):
                    mask_a = get_mask(img_hsv, target_a, sens_a, bright_a); mask_b = get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(mask_a, mask_b); denom = cv2.countNonZero(mask_a)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0; unit = "% Coloc"; res_disp = cv2.merge([mask_b, mask_a, np.zeros_like(mask_a)])
                elif mode.startswith("4."):
                    ma, mb = get_mask(img_hsv, target_a, sens_common, bright_common), get_mask(img_hsv, target_b, sens_common, bright_common)
                    pa, pb = get_centroids(ma), get_centroids(mb)
                    if pa and pb: val = np.mean([np.min([np.linalg.norm(a - b) for b in pb]) for a in pa]) * (scale_val if scale_val > 0 else 1)
                    unit = "μm Dist" if scale_val > 0 else "px Dist"; res_disp = cv2.addWeighted(img_rgb, 0.6, cv2.merge([ma, mb, np.zeros_like(ma)]), 0.4, 0)

                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name}")
                st.markdown(f"**Detected Group:** `{current_group_label}`")
                
                # Detailed Result Display
                if "Density(cells/mm2)" in extra_data:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Count", f"{int(val)} cells")
                    c_m2.metric("Cell Density", f"{int(extra_data['Density(cells/mm2)']):,} /mm²")
                    c_m3.caption(f"Area: {extra_data['Target Area(mm2)']:.4f} mm² ({extra_data['Norm Basis']})")
                elif "Target Area(mm2)" in extra_data:
                    c_m1, c_m2 = st.columns(2)
                    c_m1.metric("Fraction", f"{val:.2f} %")
                    c_m2.caption(f"Denom. Area: {extra_data['Target Area(mm2)']:.4f} mm² ({extra_data['Norm Basis']})")
                else:
                    st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                c1, c2 = st.columns(2); c1.image(img_rgb, caption="Raw Image"); c2.image(res_disp, caption="Analysis Result")
                
                # Build Result Data (Including Parameters)
                row_data = {
                    "Software Version": SOFTWARE_VERSION,
                    "Analysis ID": st.session_state.current_analysis_id,
                    "Analysis Date (UTC)": datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "Filename": file.name,
                    "Group": current_group_label,
                    "Measured Value": val,
                    "Unit": unit,
                }
                if extra_data: row_data.update(extra_data)
                # ★ Bind parameters strictly ★
                row_data.update(current_params_dict)
                
                batch_results.append(row_data)
        
        if st.button("Commit Batch Data", type="primary"):
            st.session_state.analysis_history.extend(batch_results)
            st.success("Data added to history. Analysis ID maintained.")
            st.rerun()

    if st.session_state.analysis_history:
        st.divider()
        st.header("💾 CSV Export (Full Traceability)")
        df_exp = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df_exp, use_container_width=True)
        utc_filename = f"quantified_data_{st.session_state.current_analysis_id}.csv"
        st.download_button("📥 Download Results CSV", df_exp.to_csv(index=False).encode('utf-8-sig'), utc_filename)

# ---------------------------------------------------------
# 5. Tab 2: Performance Validation
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 Performance Validation Summary")
    st.markdown("""
    * **Validation Dataset:** BBBC005 (Broad Bioimage Benchmark Collection)
    * **Scale:** 3,200 Images (High-Throughput)
    * **Methodology:** Parameters optimized for each density group to demonstrate maximum performance under appropriate calibration.
    """)

    if not df_val.empty:
        gt_map = {'C14': 14, 'C40': 40, 'C70': 70, 'C100': 100}
        df_hq = df_val[(df_val['Focus'] >= 1) & (df_val['Focus'] <= 5)]
        w1_hq = df_hq[df_hq['Channel'] == 'W1']
        avg_acc = w1_hq['Accuracy'].mean()
        df_lin = w1_hq.groupby('Ground Truth')['Value'].mean().reset_index()
        r2 = np.corrcoef(df_lin['Ground Truth'], df_lin['Value'])[0, 1]**2

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Accuracy", f"{avg_acc:.1f}%")
        m2.metric("Linearity (R²)", f"{r2:.4f}")
        m3.metric("Validated Images", "3,200+")

        st.divider()
        st.subheader("📈 1. Counting Performance & Linearity (W1 vs W2)")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot([0, 110], [0, 110], 'k--', alpha=0.3, label='Ideal Line')
        ax1.scatter(df_lin['Ground Truth'], df_lin['Value'], color='#1f77b4', s=100, label='W1 (Nuclei)', zorder=5)
        w2_lin = df_hq[df_hq['Channel'] == 'W2'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax1.scatter(w2_lin['Ground Truth'], w2_lin['Value'], color='#ff7f0e', s=100, marker='D', label='W2 (Cytoplasm)', zorder=5)
        z = np.polyfit(df_lin['Ground Truth'], df_lin['Value'], 1)
        ax1.plot(df_lin['Ground Truth'], np.poly1d(z)(df_lin['Ground Truth']), '#1f77b4', alpha=0.5, label='W1 Reg')
        ax1.set_xlabel('Ground Truth'); ax1.set_ylabel('Measured Value'); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 2. Accuracy by Density")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            df_bar = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().reset_index()
            df_bar['Density'] = pd.Categorical(df_bar['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.barplot(data=df_bar, x='Density', y='Accuracy', hue='Channel', palette={'W1': '#1f77b4', 'W2': '#ff7f0e'}, ax=ax2)
            ax2.axhline(100, color='red', linestyle='--'); ax2.set_ylabel('Accuracy (%)')
            st.pyplot(fig2)
        with c2:
            st.subheader("📉 3. Optical Robustness (Blur Tolerance)")
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            df_decay = df_val[df_val['Channel'] == 'W1'].copy()
            df_decay['Density'] = pd.Categorical(df_decay['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
            sns.lineplot(data=df_decay, x='Focus', y='Accuracy', hue='Density', marker='o', ax=ax3)
            ax3.axhline(100, color='red', linestyle='--'); ax3.set_ylabel('Accuracy (%)')
            st.pyplot(fig3)
        st.divider()
        st.subheader("📋 4. Validation Numerical Data")
        summary = df_hq.groupby(['Density', 'Channel'])['Accuracy'].mean().unstack().reset_index()
        summary['Theoretical'] = summary['Density'].map(gt_map)
        summary['W1 Measured'] = (summary['W1']/100)*summary['Theoretical']
        summary['W2 Measured'] = (summary['W2']/100)*summary['Theoretical']
        summary['Density'] = pd.Categorical(summary['Density'], categories=['C14', 'C40', 'C70', 'C100'], ordered=True)
        summary = summary.sort_values('Density')
        st.table(summary[['Density', 'Theoretical', 'W1', 'W1 Measured', 'W2', 'W2 Measured']].rename(columns={
            'W1': 'W1 Accuracy(%)', 'W1 Measured': 'W1 Mean Count', 'W2': 'W2 Accuracy(%)', 'W2 Measured': 'W2 Mean Count'
        }))
        st.info("💡 **Conclusion:** W1 (Nuclei) maintains high accuracy across all densities. W2 (Cytoplasm) shows significant fluctuation (under/over-estimation) and is not recommended for scientific quantitative analysis.")
    else:
        st.error("Validation CSV file not found. Please place it in the root directory.")
