import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hyperkalemia_engine import HyperkalemiaPredictor

st.set_page_config(
    page_title="1D ECG Hyperkalemia Diagnostic AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 6px;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 2px solid;
    }
    .card-hyper {
        background-color: #fef2f2;
        border-color: #ef4444;
    }
    .card-normal {
        background-color: #f0fdf4;
        border-color: #22c55e;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return HyperkalemiaPredictor()

predictor = get_predictor()

st.markdown("""
<div class="main-header">
    <h1>🫀 1D ECG Hyperkalemia AI Diagnostic System</h1>
    <p>Automated deep learning classification of Hyperkalemia vs. Normal from single-lead ECG (.txt) records.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("🎛️ Analysis Controls")
mode = st.sidebar.radio("Navigation", ["🔬 Single File Analysis", "📊 Batch Processing", "ℹ️ Model Specifications"])

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
sample_files = {}
if os.path.exists(SAMPLE_DIR):
    for f in os.listdir(SAMPLE_DIR):
        if f.endswith(".txt"):
            clean_name = f.replace(".txt", "")
            sample_files[clean_name] = os.path.join(SAMPLE_DIR, f)

if mode == "🔬 Single File Analysis":
    st.subheader("Single Patient ECG Classification")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📤 Input ECG Signal")
        uploaded_file = st.file_uploader("Upload .txt Recording", type=["txt", "csv"])
        
        selected_sample = None
        if sample_files:
            st.markdown("— **OR Test with Sample Patient** —")
            sample_choice = st.selectbox("Select Patient Record", ["None"] + list(sample_files.keys()))
            if sample_choice != "None":
                selected_sample = sample_files[sample_choice]
        
        stride = st.slider("Window Stride (Samples)", min_value=500, max_value=4000, value=2000, step=500,
                           help="Stride between consecutive 8s (4000 samples @ 500Hz) evaluation windows.")

    with col2:
        file_to_process = uploaded_file or selected_sample
        
        if file_to_process is not None:
            with st.spinner("Processing ECG waveform and running attention 1D model..."):
                try:
                    res = predictor.predict(file_to_process, stride=stride)
                    
                    pred = res["prediction"]
                    conf = res["confidence"]
                    p_norm = res["p_normal_pct"]
                    p_hyp = res["p_hyperkalemia_pct"]
                    risk = res["risk_level"]
                    metrics = res["metrics"]
                    
                    card_class = "card-hyper" if pred == "Hyperkalemia" else "card-normal"
                    card_color = "#dc2626" if pred == "Hyperkalemia" else "#16a34a"
                    icon = "🚨" if pred == "Hyperkalemia" else "✅"
                    
                    st.markdown(f"""
                    <div class="result-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h2 style="color: {card_color}; margin: 0; font-size: 1.8rem;">{icon} {pred.upper()}</h2>
                                <p style="color: #475569; margin: 4px 0 0 0; font-weight: 500;">Risk: <b>{risk}</b></p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.9rem; color: #64748b;">Confidence</div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #1e293b;">{conf}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("P(Normal)", f"{p_norm}%")
                    m2.metric("P(Hyperkalemia)", f"{p_hyp}%")
                    m3.metric("Est. Heart Rate", f"{metrics.get('estimated_bpm', 'N/A')} BPM")
                    m4.metric("8s Windows Analyzed", res["n_windows"])
                    
                    st.markdown("### 📈 Interactive ECG Waveform (500 Hz)")
                    fs = 500.0
                    ecg_sig = res["ecg_500hz"]
                    max_plot_sec = 20
                    max_samples = min(len(ecg_sig), int(max_plot_sec * fs))
                    t_axis = np.arange(max_samples) / fs
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t_axis,
                        y=ecg_sig[:max_samples],
                        mode="lines",
                        line=dict(color="#0284c7", width=1.5),
                        name="Preprocessed ECG (mV)"
                    ))
                    fig.update_layout(
                        height=350,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="Time (seconds)",
                        yaxis_title="Amplitude (mV)",
                        template="plotly_white",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### ⏱️ Window-by-Window Model Confidence")
                    df_win = pd.DataFrame(res["window_results"])
                    
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=df_win["start_time_sec"] + 4.0,
                        y=df_win["p_hyperkalemia"],
                        mode="lines+markers",
                        line=dict(color="#dc2626", width=2),
                        marker=dict(size=6),
                        name="P(Hyperkalemia) %"
                    ))
                    fig_timeline.add_hline(y=50.0, line_dash="dash", line_color="#64748b", annotation_text="50% Threshold")
                    fig_timeline.update_layout(
                        height=240,
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis_range=[-5, 105],
                        xaxis_title="Recording Time (seconds)",
                        yaxis_title="Probability (%)",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    with st.expander("📋 View Window Data Table"):
                        st.dataframe(df_win, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error analyzing recording: {e}")
        else:
            st.info("👈 Upload an ECG .txt file or pick a sample from the left panel to begin.")

elif mode == "📊 Batch Processing":
    st.subheader("Batch Patient File Processing")
    uploaded_batch = st.file_uploader("Upload Multiple .txt Files", type=["txt", "csv"], accept_multiple_files=True)
    
    if uploaded_batch:
        if st.button("⚡ Run Batch Analysis", type="primary"):
            results = []
            progress_bar = st.progress(0)
            
            for i, f_obj in enumerate(uploaded_batch):
                try:
                    res = predictor.predict(f_obj)
                    results.append({
                        "Filename": f_obj.name,
                        "Diagnosis": res["prediction"],
                        "Confidence (%)": res["confidence"],
                        "P(Normal) %": res["p_normal_pct"],
                        "P(Hyperkalemia) %": res["p_hyperkalemia_pct"],
                        "Risk Level": res["risk_level"],
                        "Duration (s)": res["metrics"]["duration_sec"],
                        "Est. BPM": res["metrics"].get("estimated_bpm", "N/A"),
                        "Windows": res["n_windows"]
                    })
                except Exception as e:
                    results.append({
                        "Filename": f_obj.name,
                        "Diagnosis": f"Error: {e}",
                        "Confidence (%)": 0,
                        "P(Normal) %": 0,
                        "P(Hyperkalemia) %": 0,
                        "Risk Level": "Failed",
                        "Duration (s)": 0,
                        "Est. BPM": "N/A",
                        "Windows": 0
                    })
                progress_bar.progress((i + 1) / len(uploaded_batch))
                
            df_results = pd.DataFrame(results)
            st.success(f"Successfully processed {len(results)} recordings!")
            st.dataframe(df_results, use_container_width=True)
            
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report (CSV)", csv_data, "hyperkalemia_triage_report.csv", "text/csv")

elif mode == "ℹ️ Model Specifications":
    st.subheader("Model Architecture & Preprocessing Pipeline")
    st.markdown("""
    ### 🔬 Deep Learning Model Summary
    - **Architecture**: 1D Convolutional Neural Network with 1D-Attention Mechanism.
    - **Input Shape**: `(4000, 1)` corresponding to **8 seconds** of single-lead ECG signal sampled at **500 Hz**.
    - **Layers**: 3x Conv1D blocks (64, 128, 256 filters) $\\to$ 1D Attention scoring $\\to$ Dense 128 $\\to$ Softmax (Normal vs. Hyperkalemia).
    
    ### ⚙️ Signal Processing Pipeline
    1. **Format Handling**: Auto-parsing of OpenSignals / Cardiobanplux JSON headers and raw text.
    2. **ADC Transfer Function**: Calibrated millivolt conversion for 16-bit acquisition systems:
       $$\\text{ECG (mV)} = \\left(\\frac{\\text{ADC}}{2^{16}} - 0.5\\right) \\times \\left(\\frac{3.0}{1000.0}\\right) \\times 1000.0$$
    3. **Resampling**: Anti-aliased decimation/resampling from native rate (1000 Hz) to model rate (**500 Hz**).
    4. **Filtering**: 0.5 Hz 4th-order Butterworth high-pass filter for baseline wander suppression.
    5. **Sliding Window Normalization**: Standardized using fitted training distribution parameters.
    """)
