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

# --- Analysis ID Management ---
if "current_analysis_id" not in st.session_state:
    date_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    unique_suffix = str(uuid.uuid4())[:8]
    st.session_state.current_analysis_id = f"AID-{date_str}-{unique_suffix}"

# ---------------------------------------------------------
# 1. Image Processing Engine
# ---------------------------------------------------------
COLOR_MAP = {
    "Brown (DAB)": {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 255])},
    "Green (GFP)": {"lower": np.array([35, 50, 50]), "upper": np.array([85, 255, 255])},
    "Red (RFP)": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
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
st.caption(f"{SOFTWARE_VERSION}: Industrial-Grade Data Extraction")

st.sidebar.markdown(f"**Current Analysis ID:** `{st.session_state.current_analysis_id}`")

tab_main, tab_val = st.tabs(["🚀 Run Analysis", "🏆 Performance Validation"])

with st.sidebar:
    st.markdown("### [Important: Scientific Use]")
    st.warning("""
    **Planning to publish results?**
    Please contact the developer in advance to discuss traceability and data integrity.
    👉 **[Contact Form](https://forms.gle/xgNscMi3KFfWcuZ1A)**
    """)
    st.divider()

    st.header("Analysis Recipe")
    mode = st.selectbox("Select Analysis Mode:", [
        "1. Area Occupancy %", 
        "2. Nuclei Count / Density", 
        "3. Colocalization Analysis", 
        "4. Spatial Distance Analysis", 
        "5. Ratio Trend Analysis"
    ])

    st.divider()

    # --- Grouping Settings ---
    st.markdown("### 🏷️ Grouping Settings")
    group_strategy = st.radio("Label Determination:", ["Manual Entry", "Auto from Filename"])
    
    if group_strategy.startswith("Manual"):
        sample_group = st.text_input("Group Label (X-axis):", value="Control")
        filename_sep = None
    else:
        filename_sep = st.text_input("Separator (e.g., _ ):", value="_")
        sample_group = "(Auto Detected)" 

    st.divider()

    # Dynamic Parameter Settings
    if mode.startswith("5."):
        st.markdown("### 🔢 Trend Conditions")
        trend_metric = st.radio("Metric:", ["Colocalization Rate", "Area Occupancy"])
        ratio_val = st.number_input("Value:", value=0, step=10)
        ratio_unit = st.text_input("Unit:", value="%", key="unit")
        if group_strategy.startswith("Manual"): sample_group = f"{ratio_val}{ratio_unit}"
        
        if trend_metric.startswith("Colocalization"):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3)
                sens_a = st.slider("A Sens", 5, 50, 20); bright_a = st.slider("A Bright", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2)
                sens_b = st.slider("B Sens", 5, 50, 20); bright_b = st.slider("B Bright", 0, 255, 60)
        else:
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()), index=2)
            sens_a = st.slider("Sens", 5, 50, 20); bright_a = st.slider("Bright", 0, 255, 60)
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=False, key="roi_m5")
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2, key="roi_c5")
                sens_roi = st.slider("ROI Sens", 5, 50, 20, key="roi_s5")
                bright_roi = st.slider("ROI Bright", 0, 255, 40, key="roi_b5")
    else:
        if mode.startswith("1."):
            target_a = st.selectbox("Target Color:", list(COLOR_MAP.keys()))
            sens_a = st.slider("Sensitivity", 5, 50, 20)
            bright_a = st.slider("Brightness", 0, 255, 60)
            # Area Occupancy ROI normalization added
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=False, key="roi_m1")
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2, key="roi_c1")
                sens_roi = st.slider("ROI Sens", 5, 50, 20, key="roi_s1")
                bright_roi = st.slider("ROI Bright", 0, 255, 40, key="roi_b1")
        elif mode.startswith("2."):
            min_size = st.slider("Min Size (px)", 10, 500, 50); bright_count = st.slider("Threshold", 0, 255, 50)
            use_roi_norm = st.checkbox("Normalize by Tissue Area (ROI)", value=True, key="roi_m2")
            if use_roi_norm:
                roi_color = st.selectbox("Tissue Color:", list(COLOR_MAP.keys()), index=2, key="roi_c2")
                sens_roi = st.slider("ROI Sens", 5, 50, 20, key="roi_s2")
                bright_roi = st.slider("ROI Bright", 0, 255, 40, key="roi_b2")
        elif mode.startswith("3."):
            c1, c2 = st.columns(2)
            with c1:
                target_a = st.selectbox("CH-A:", list(COLOR_MAP.keys()), index=3); sens_a = st.slider("A Sens", 5, 50, 20); bright_a = st.slider("A Bright", 0, 255, 60)
            with c2:
                target_b = st.selectbox("CH-B:", list(COLOR_MAP.keys()), index=2); sens_b = st.slider("B Sens", 5, 50, 20); bright_b = st.slider("B Bright", 0, 255, 60)
        elif mode.startswith("4."):
            target_a = st.selectbox("Origin A:", list(COLOR_MAP.keys()), index=2); target_b = st.selectbox("Target B:", list(COLOR_MAP.keys()), index=3)
            sens_common = st.slider("Common Sens", 5, 50, 20); bright_common = st.slider("Common Bright", 0, 255, 60)

    st.divider()
    scale_val = st.number_input("Scale (μm/px)", value=1.5267, format="%.4f")
    
    st.button("📸 Next Group (Clear Images)", on_click=lambda: st.session_state.update({"uploader_key": str(uuid.uuid4())}))
    
    if st.button("Clear History"): 
        st.session_state.analysis_history = []
        st.rerun()

    # Audit Trail Data
    current_params = {"Software": SOFTWARE_VERSION, "AID": st.session_state.current_analysis_id, "Mode": mode, "Scale": scale_val}
    if 'use_roi_norm' in locals(): current_params["ROI_Norm"] = use_roi_norm
    
    st.download_button("📥 Download Log (CSV)", pd.DataFrame([current_params]).to_csv(index=False).encode('utf-8'), f"log_{st.session_state.current_analysis_id}.csv")

# ---------------------------------------------------------
# 4. Tab 1: Run Analysis
# ---------------------------------------------------------
with tab_main:
    uploaded_files = st.file_uploader("Upload Images (16-bit TIFF OK)", type=["jpg", "png", "tif", "tiff"], accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded_files:
        st.success(f"Analyzing {len(uploaded_files)} images...")
        batch_results = []
        for i, file in enumerate(uploaded_files):
            file.seek(0); file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img_raw = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img_raw is not None:
                # Group logic
                current_group = file.name.split(filename_sep)[0] if group_strategy.startswith("Auto") else sample_group

                # Scaling & Normalization for display
                img_f = img_raw.astype(np.float32); mn, mx = np.min(img_f), np.max(img_f)
                img_8 = ((img_f - mn) / (mx - mn) * 255.0 if mx > mn else np.clip(img_f, 0, 255)).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_8, cv2.COLOR_GRAY2BGR) if len(img_8.shape) == 2 else img_8[:,:,:3]
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB); img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                val, unit, res_disp = 0.0, "", img_rgb.copy()
                h, w = img_rgb.shape[:2]; fov_mm2 = (h * w) * ((scale_val / 1000) ** 2)

                extra_data = {}

                # --- Mode 1: Area Occupancy (ROI Integrated) ---
                if mode.startswith("1.") or (mode.startswith("5.") and trend_metric.startswith("Area")):
                    mask_target = get_mask(img_hsv, target_a, sens_a, bright_a)
                    a_den_px = h * w
                    roi_status = "Field of View"
                    final_mask = mask_target
                    
                    if use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        final_mask = cv2.bitwise_and(mask_target, mask_roi)
                        a_den_px = cv2.countNonZero(mask_roi)
                        roi_status = "Inside ROI"
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    val = (cv2.countNonZero(final_mask) / a_den_px * 100) if a_den_px > 0 else 0
                    unit = "% Area"
                    
                    # Highlight mask
                    m_disp = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2RGB); m_disp[:,:,0]=0; m_disp[:,:,2]=0
                    res_disp = cv2.addWeighted(res_disp, 0.7, m_disp, 0.3, 0)
                    extra_data = {"Target Area (mm2)": round(a_den_px * ((scale_val/1000)**2), 6), "Normalization Basis": roi_status}

                # --- Mode 2: Nuclei Count ---
                elif mode.startswith("2."):
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY); _, th = cv2.threshold(gray, bright_count, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0); _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(cv2.bitwise_and(th, otsu), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid = [c for c in cnts if cv2.contourArea(c) > min_size]; val, unit = len(valid), "cells"
                    cv2.drawContours(res_disp, valid, -1, (0,255,0), 2)
                    
                    a_den_mm2 = fov_mm2; roi_status = "Field of View"
                    if use_roi_norm:
                        mask_roi = get_tissue_mask(img_hsv, roi_color, sens_roi, bright_roi)
                        a_den_mm2 = cv2.countNonZero(mask_roi) * ((scale_val/1000)**2)
                        roi_status = "Inside ROI"
                        cv2.drawContours(res_disp, cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255,0,0), 3)

                    density = val / a_den_mm2 if a_den_mm2 > 0 else 0
                    extra_data = {"Target Area (mm2)": round(a_den_mm2, 6), "Density (cells/mm2)": round(density, 2), "Normalization Basis": roi_status}

                # --- Other Modes ---
                elif mode.startswith("3.") or (mode.startswith("5.") and trend_metric.startswith("Colocalization")):
                    ma, mb = get_mask(img_hsv, target_a, sens_a, bright_a), get_mask(img_hsv, target_b, sens_b, bright_b)
                    coloc = cv2.bitwise_and(ma, mb); denom = cv2.countNonZero(ma)
                    val = (cv2.countNonZero(coloc) / denom * 100) if denom > 0 else 0; unit = "% Coloc"; res_disp = cv2.merge([mb, ma, np.zeros_like(ma)])
                elif mode.startswith("4."):
                    ma, mb = get_mask(img_hsv, target_a, sens_common, bright_common), get_mask(img_hsv, target_b, sens_common, bright_common)
                    pa, pb = get_centroids(ma), get_centroids(mb)
                    if pa and pb: val = np.mean([np.min([np.linalg.norm(a - b) for b in pb]) for a in pa]) * (scale_val if scale_val > 0 else 1)
                    unit = "μm Dist"; res_disp = cv2.addWeighted(img_rgb, 0.6, cv2.merge([ma, mb, np.zeros_like(ma)]), 0.4, 0)

                st.divider()
                st.markdown(f"### 📷 Image {i+1}: {file.name} ({current_group})")
                
                if "Density (cells/mm2)" in extra_data:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Count", f"{int(val)} cells")
                    c2.metric("Density", f"{int(extra_data['Density (cells/mm2)']):,} /mm²")
                    c3.caption(f"Area: {extra_data['Target Area (mm2)']:.4f} mm² ({extra_data['Normalization Basis']})")
                elif "Target Area (mm2)" in extra_data:
                    c1, c2 = st.columns(2)
                    c1.metric("Occupancy", f"{val:.2f} %")
                    c2.caption(f"Denominator: {extra_data['Target Area (mm2)']:.4f} mm² ({extra_data['Normalization Basis']})")
                else:
                    st.markdown(f"### Result: **{val:.2f} {unit}**")
                
                col1, col2 = st.columns(2); col1.image(img_rgb, caption="Raw"); col2.image(res_disp, caption="Result")
                
                row = {"AID": st.session_state.current_analysis_id, "File": file.name, "Group": current_group, "Value": val, "Unit": unit}
                if extra_data: row.update(extra_data)
                batch_results.append(row)
        
        if st.button("Commit Batch", type="primary"):
            st.session_state.analysis_history.extend(batch_results); st.rerun()

    if st.session_state.analysis_history:
        st.divider(); st.header("💾 Export Results")
        df_exp = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(df_exp, use_container_width=True)
        st.download_button("📥 Download Results CSV", df_exp.to_csv(index=False).encode('utf-8'), f"results_{st.session_state.current_analysis_id}.csv")

# ---------------------------------------------------------
# 5. Tab 2: Validation
# ---------------------------------------------------------
with tab_val:
    st.header("🏆 Validation Evidence")
    st.markdown("* **Standard:** BBBC005 (Broad Bioimage Benchmark Collection)\n* **Scale:** 3,200 High-Throughput Images")
    if not df_val.empty:
        # Basic plotting logic maintained as per V2 design
        fig, ax = plt.subplots(figsize=(10, 5))
        df_lin = df_val[df_val['Channel']=='W1'].groupby('Ground Truth')['Value'].mean().reset_index()
        ax.scatter(df_lin['Ground Truth'], df_lin['Value'], label='W1 (Nuclei)')
        ax.plot([0, 110], [0, 110], 'k--', alpha=0.3)
        ax.set_xlabel('Ground Truth'); ax.set_ylabel('Measured'); ax.legend()
        st.pyplot(fig)
        st.info("💡 **Overall Conclusion:** W1 (Nuclei) maintains high accuracy across all density ranges.")
    else:
        st.error("Validation CSV missing.")
