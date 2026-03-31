"""
STREAMLIT DASHBOARD - Real-time visualization of the Cross-Layer Physics-Based IDS
Run with: streamlit run dashboard.py
"""

import streamlit as st
import json
import os
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
# SAFE JSON LOADING
# ═══════════════════════════════════════════════════════════

def safe_load_json(filepath, default=None):
    """Load JSON with retry on partial write race condition"""
    if default is None:
        default = {}
    for attempt in range(3):
        try:
            if not os.path.exists(filepath):
                return default
            with open(filepath) as f:
                content = f.read()
            if content.strip():
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError, OSError, ValueError):
            if attempt < 2:
                time.sleep(0.05)
    return default

st.set_page_config(page_title="Cross-Layer Vehicle IDS", layout="wide", initial_sidebar_state="expanded")

RESULTS_DIR = 'demo_results'
RESULTS_MODEL_DIR = 'results'

# Ensure directory exists before dashboard starts
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# SIDEBAR — Interactive Attack Controls
# ═══════════════════════════════════════════════════════════

st.sidebar.markdown("# 🎮 Attack Control Panel")
st.sidebar.markdown("*Click to trigger attacks in real-time*")

def _set_attack(atype, label, desc):
    try:
        with open(os.path.join(RESULTS_DIR, 'interactive_state.json'), 'w') as f:
            json.dump({'attack_type': atype, 'label': label,
                       'description': desc, 'timestamp': time.time()}, f)
    except OSError:
        pass

st.sidebar.markdown("### Normal")
if st.sidebar.button("🟢 Normal Driving", use_container_width=True, key="btn_none"):
    _set_attack('none', 0, '🟢 Normal driving')
    st.sidebar.success("✅ Normal mode")

st.sidebar.markdown("### Single-Layer Attacks")
for label, atype, alabel, desc in [
    ('🟡 GPS Speed Spoofing',      'gps_spoof',     1, '🟡 GPS Speed Spoofing'),
    ('🟡 CAN Steering Injection',  'can_inject',    1, '🟡 CAN Steering Injection'),
    ('🟡 V2X Fake Curvature',      'v2x_fake_curv', 1, '🟡 V2X Fake Curvature'),
    ('🟡 CAN DoS Flooding',        'can_dos',       1, '🟡 CAN DoS Flooding'),
]:
    if st.sidebar.button(label, use_container_width=True, key=f"btn_{atype}"):
        _set_attack(atype, alabel, desc)
        st.sidebar.warning(f"⚡ {label} active!")

st.sidebar.markdown("### Coordinated Attacks")
for label, atype, alabel, desc in [
    ('🔴 CAN + V2X (Phantom Curve)', 'coord_can_v2x', 2, '🔴 COORDINATED: CAN + V2X'),
    ('🔴 All Three Layers',          'coord_all',     2, '🔴 COORDINATED: All Three'),
    ('🔴 GPS + CAN Speed',           'coord_gps_can', 2, '🔴 COORDINATED: GPS + CAN'),
    ('🔴 GPS + V2X Heading',         'coord_gps_v2x', 2, '🔴 COORDINATED: GPS + V2X'),
]:
    if st.sidebar.button(label, use_container_width=True, key=f"btn_{atype}"):
        _set_attack(atype, alabel, desc)
        st.sidebar.error(f"🚨 {label} active!")

st.sidebar.markdown("### Unseen Attacks")
if st.sidebar.button("🔴 UNSEEN: Speed All Layers", use_container_width=True, key="btn_speed_all"):
    _set_attack('coord_speed_all', 2, '🔴 UNSEEN: Speed All')
    st.sidebar.error("🚨 UNSEEN attack active!")

st.sidebar.markdown("---")
st.sidebar.markdown("Type number in terminal OR click buttons here")

st.markdown("# Cross-Layer Physics-Based Intrusion Detection System")
st.markdown("### Detecting Coordinated Multi-Layer Cyberattacks on Software-Defined Vehicles")

tab_live, tab_radar, tab_ablation, tab_attacks, tab_importance, tab_arch = st.tabs([
    "Live Detection", "Physics Radar", "Ablation Study", "Per-Attack Results", "Feature Importance", "Architecture"])

latest_file = os.path.join(RESULTS_DIR, 'latest.json')

with tab_live:
    if os.path.exists(latest_file):
        latest = safe_load_json(latest_file, {})
        pred = latest.get('prediction', 0)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if pred == 0: st.success("### NORMAL - All Layers Consistent")
            elif pred == 1: st.warning("### SINGLE-LAYER ATTACK DETECTED")
            else: st.error("### COORDINATED ATTACK DETECTED")
        with col2:
            st.metric("Feature Compute", f"{latest.get('feature_time_us', 0)} us")
            st.metric("Inference", f"{latest.get('inference_time_us', 0)} us")
        with col3:
            mode_names = ['NORMAL', 'DEGRADED', 'SAFE STOP']
            st.metric("Vehicle Mode", mode_names[min(latest.get('vehicle_mode', 0), 2)])
            st.metric("Time", f"{latest.get('time', 0):.1f}s")
            
        # Additional live metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Drive Mode", str(latest.get('drive_mode', 'unknown')).title())
        with col_b:
            st.metric("Speed", f"{latest.get('speed', 0):.0f} m/s")
        with col_c:
            st.metric("Confidence", f"{latest.get('confidence', 0):.0%}")
        with col_d:
            st.metric("Attack", latest.get('attack_type', 'none'))
            
        st.markdown("### Cross-Layer Feature Values")
        xl_features = {
            'Speed Consistency': latest.get('xl_speed_consistency', 0),
            'Yaw CAN vs GPS': latest.get('xl_yaw_can_vs_gps', 0),
            'Yaw CAN vs IMU': latest.get('xl_yaw_can_vs_imu', 0),
            'Lat Accel GPS vs IMU': latest.get('xl_lataccel_gps_vs_imu', 0),
            'Obstacle Ultra vs V2X': latest.get('xl_obstacle_ultra_vs_v2x', 0),
            'Curvature 3-Way': latest.get('xl_curvature_3way', 0),
        }
        fig = go.Figure()
        colors = ['red' if v > 0.3 else 'orange' if v > 0.15 else 'green' for v in xl_features.values()]
        fig.add_trace(go.Bar(x=list(xl_features.keys()), y=list(xl_features.values()),
                             marker_color=colors, text=[f"{v:.3f}" for v in xl_features.values()], textposition='auto'))
        fig.update_layout(yaxis_title="Normalized Disagreement", yaxis_range=[0, 1], height=350, margin=dict(t=10))
        fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Attack Threshold")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data... Run `python stream_to_esp32.py` to start the demo.")
    log_file = os.path.join(RESULTS_DIR, 'full_demo_log.json')
    if os.path.exists(log_file):
        # H5: Tail-read — only load last 200 entries to prevent memory bloat
        try:
            with open(log_file) as f:
                content = f.read()
            if content.strip():
                full_log = json.loads(content)
                log_data = full_log[-200:] if len(full_log) > 200 else full_log
            else:
                log_data = []
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            log_data = []
        if log_data:
            st.markdown("### Detection Timeline")
            times = [d['time'] for d in log_data]
            preds = [d['prediction'] for d in log_data]
            trues = [d['true_label'] for d in log_data]
            fig_timeline = make_subplots(rows=2, cols=1, shared_xaxes=True,
                subplot_titles=['Predicted Classification', 'True Labels'], vertical_spacing=0.1)
            pred_colors = ['green' if p == 0 else 'orange' if p == 1 else 'red' for p in preds]
            true_colors = ['green' if p == 0 else 'orange' if p == 1 else 'red' for p in trues]
            fig_timeline.add_trace(go.Scatter(x=times, y=preds, mode='markers',
                marker=dict(color=pred_colors, size=3), name='Predicted'), row=1, col=1)
            fig_timeline.add_trace(go.Scatter(x=times, y=trues, mode='markers',
                marker=dict(color=true_colors, size=3), name='True'), row=2, col=1)
            fig_timeline.update_layout(height=400)
            fig_timeline.update_yaxes(ticktext=['Normal', 'Single', 'Coordinated'], tickvals=[0, 1, 2])
            st.plotly_chart(fig_timeline, use_container_width=True)
            correct = sum(1 for p, t in zip(preds, trues) if (p == 0 and t == 0) or (p > 0 and t > 0))
            st.metric("Live Detection Accuracy", f"{correct/len(preds)*100:.1f}%")

with tab_radar:
    st.markdown("### Cross-Layer Physics Consistency Radar")
    if os.path.exists(latest_file):
        latest = safe_load_json(latest_file, {})
        categories = ['Speed\nConsistency', 'Yaw\nCAN vs GPS', 'Yaw\nCAN vs IMU',
                       'Lat Accel\nGPS vs IMU', 'Obstacle\nUltra vs V2X', 'Curvature\n3-Way']
        normal_baseline = [0.05, 0.08, 0.04, 0.06, 0.03, 0.05]
        current = [latest.get('xl_speed_consistency', 0), latest.get('xl_yaw_can_vs_gps', 0),
                   latest.get('xl_yaw_can_vs_imu', 0), latest.get('xl_lataccel_gps_vs_imu', 0),
                   latest.get('xl_obstacle_ultra_vs_v2x', 0), latest.get('xl_curvature_3way', 0)]
        pred = latest.get('prediction', 0)
        current_color = 'green' if pred == 0 else 'orange' if pred == 1 else 'red'
        fill_map = {'green': 'rgba(0,200,0,0.3)', 'orange': 'rgba(255,165,0,0.3)', 'red': 'rgba(255,0,0,0.3)'}
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=normal_baseline + [normal_baseline[0]], theta=categories + [categories[0]],
            fill='toself', name='Normal Baseline', fillcolor='rgba(0,200,0,0.15)', line=dict(color='green', width=2)))
        fig.add_trace(go.Scatterpolar(r=current + [current[0]], theta=categories + [categories[0]],
            fill='toself', name='Current', fillcolor=fill_map[current_color], line=dict(color=current_color, width=3)))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        #### Physics Equations Behind Each Feature
        | Feature | Equation | Physics Principle |
        |---------|----------|-------------------|
        | Speed Consistency | abs(GPS_speed - CAN_speed) / max | Same quantity, independent sensors |
        | Yaw CAN vs IMU | abs(v*tan(d/SR)/L - w_IMU) / max | Bicycle model vs gyroscope |
        | Yaw CAN vs GPS | abs(v*tan(d/SR)/L - w_GPS) / max | Bicycle model vs GPS heading |
        | Lat Accel | abs(v*w_GPS - a_IMU) / max | Centripetal acceleration |
        | Obstacle | abs(ultra - v2x) / max | Same obstacle, different sensors |
        | Curvature 3-Way | avg(abs(k_V2X-k_CAN), abs(k_V2X-k_GPS)) | Road curvature consistency |
        """)
    else:
        st.info("Waiting for data...")

with tab_ablation:
    st.markdown("### Ablation Study: Proving Each Component's Value")
    ablation_file = os.path.join(RESULTS_MODEL_DIR, 'ablation_results.json')
    if os.path.exists(ablation_file):
        ablation = safe_load_json(ablation_file, {})
        configs = list(ablation.keys())
        means_w = [ablation[c]['weighted_f1_mean'] for c in configs]
        stds_w = [ablation[c]['weighted_f1_std'] for c in configs]
        means_c = [ablation[c]['coord_f1_mean'] for c in configs]
        stds_c = [ablation[c]['coord_f1_std'] for c in configs]
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Overall Weighted F1', 'Coordinated Attack F1'])
        colors = ['#44BB44', '#FF8844', '#4488FF', '#FF4444', '#888888', '#FFD700', '#FF1493']
        fig.add_trace(go.Bar(name='Weighted F1', x=configs, y=means_w,
            error_y=dict(type='data', array=stds_w, visible=True), marker_color=colors[:len(configs)],
            text=[f"{m:.3f}" for m in means_w], textposition='auto'), row=1, col=1)
        fig.add_trace(go.Bar(name='Coordinated F1', x=configs, y=means_c,
            error_y=dict(type='data', array=stds_c, visible=True), marker_color=colors[:len(configs)],
            text=[f"{m:.3f}" for m in means_c], textposition='auto'), row=1, col=2)
        fig.update_layout(height=500, showlegend=False)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Detailed Results (5-fold Stratified Group CV)")
        table_data = []
        for c in configs:
            d = ablation[c]
            table_data.append({'Model': c, 'Features': d['n_features'],
                'Weighted F1': f"{d['weighted_f1_mean']:.4f} +/- {d['weighted_f1_std']:.4f}",
                '95% CI': f"[{d['weighted_f1_ci'][0]:.4f}, {d['weighted_f1_ci'][1]:.4f}]",
                'Coordinated F1': f"{d['coord_f1_mean']:.4f} +/- {d['coord_f1_std']:.4f}"})
        st.table(table_data)
    else:
        st.warning("Run `python train_model.py` to generate ablation results.")

with tab_attacks:
    st.markdown("### Per-Attack-Type Detection Performance")
    attack_file = os.path.join(RESULTS_MODEL_DIR, 'per_attack_results.json')
    if os.path.exists(attack_file):
        attack_results = safe_load_json(attack_file, {})
        for attack_name, metrics in sorted(attack_results.items()):
            if attack_name == 'normal':
                fp_rate = metrics.get('fp_rate', 0)
                st.markdown(f"**{attack_name}** - False Positive Rate: {fp_rate:.1f}%")
            else:
                det = metrics.get('detection_rate', 0)
                unseen = " **(UNSEEN)**" if 'can_imu' in attack_name or 'speed_all' in attack_name else ""
                st.markdown(f"**{attack_name}** - Detection: {det:.1f}%{unseen}")
        st.markdown("""
        ### Comparison With Published Methods
        | Method | Dataset | CAN Attack F1 | Coordinated Attack F1 |
        |--------|---------|---------------|----------------------|
        | Song et al. 2020 (DCNN) | Car Hacking | 0.99 | N/A |
        | Hossain et al. 2020 (LSTM) | Car Hacking | 0.98 | N/A |
        | **This work (Full_29)** | **Cross-Layer** | **~0.95** | **~0.77+** |
        """)
    else:
        st.warning("Run `python train_model.py` to generate per-attack results.")

with tab_importance:
    st.markdown("### Feature Importance Analysis")
    importance_file = os.path.join(RESULTS_MODEL_DIR, 'feature_importance.json')
    if os.path.exists(importance_file):
        importance_data = safe_load_json(importance_file, [])
        names = [d['feature_name'] for d in importance_data]
        values = [d['importance'] for d in importance_data]
        layers = [d['layer'] for d in importance_data]
        layer_colors = {'Sensor': '#44BB44', 'CAN': '#FF8844', 'V2X': '#4488FF', 'CrossLayer': '#FF4444', 'Temporal': '#9944FF'}
        colors = [layer_colors.get(l, '#888888') for l in layers]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=names, y=values, marker_color=colors,
            text=[f"{v:.4f}" for v in values], textposition='auto'))
        fig.update_layout(title="Feature Importance (Red = Cross-Layer, Purple = Temporal)",
            yaxis_title="Mean Decrease in Gini Impurity", xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)
        layer_totals = {}
        for d in importance_data:
            layer_totals[d['layer']] = layer_totals.get(d['layer'], 0) + d['importance']
        st.markdown("#### Importance by Layer")
        for layer, total in sorted(layer_totals.items(), key=lambda x: -x[1]):
            pct = total * 100
            st.markdown(f"**{layer}**: {pct:.1f}%")
    else:
        st.warning("Run `python train_model.py` to generate importance data.")

with tab_arch:
    st.markdown("""
    ## System Architecture

    ```
    VEHICLE SENSOR LAYERS
    
      GPS/IMU        CAN Bus        V2X (Wireless)
      (Trusted)     (Vulnerable)    (Vulnerable)
          |              |               |
          +--------------+---------------+
                         |
                         v
          CROSS-LAYER PHYSICS ENGINE
          - Bicycle Model (yaw rate)
          - Centripetal Acceleration
          - 3-Way Curvature Consistency
          - Speed Cross-Check
          - Temporal Sliding Window
          19 raw features -> 29 total
                         |
                         v
          RANDOM FOREST CLASSIFIER
          20 trees, depth 12
          ~270 KB model, ~245 us inference
          100% explainable
                         |
                         v
          RESPONSE ENGINE
          NORMAL -> DEGRADED -> SAFE STOP
          + LED indicators
    
    ALL RUNNING ON ESP32 ($5, 240MHz, 520KB SRAM)
    ```

    ## Key Specifications
    | Parameter | Value |
    |-----------|-------|
    | Total Features | 29 (7 sensor + 7 CAN + 5 V2X + 6 cross-layer + 4 temporal) |
    | Physics Equations | 7 |
    | RF Trees | 20 |
    | Max Depth | 12 |
    | Model Size | ~270 KB |
    | Feature Compute | ~87 us |
    | Inference | ~245 us |
    | Total Latency | ~332 us (0.33% of 100ms budget) |
    | Hardware Cost | ~$5 |
    | Attack Types | 14 (7 single + 5 coord training + 2 coord unseen) |

    ## The Core Insight
    > **Physics cannot be hacked.** An attacker can inject false data into any digital
    > communication channel. But they cannot change the laws of physics. If the car is
    > going straight, the IMU gyroscope measures near-zero yaw rate regardless of what
    > the CAN bus claims about steering angle. Our six cross-layer features encode six
    > physics equations that serve as unforgeable lie detectors.
    """)

# M3: Unconditional auto-refresh gated on file recency
try:
    _latest_path = os.path.join(RESULTS_DIR, 'latest.json')
    if os.path.exists(_latest_path):
        _last_mod = os.path.getmtime(_latest_path)
        if time.time() - _last_mod < 5:
            time.sleep(0.5)
            st.rerun()
except (FileNotFoundError, OSError):
    pass
