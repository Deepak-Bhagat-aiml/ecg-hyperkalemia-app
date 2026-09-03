import os
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hyperkalemia_engine import HyperkalemiaPredictor

print("Initializing Hyperkalemia Predictor...")
predictor = HyperkalemiaPredictor()
print("Predictor ready!")

def plot_ecg_and_timeline(ecg_500hz, window_results, max_plot_seconds=30):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    fs = 500.0
    total_samples = min(len(ecg_500hz), int(max_plot_seconds * fs))
    time_axis = np.arange(total_samples) / fs
    signal_slice = ecg_500hz[:total_samples]
    
    ax1.plot(time_axis, signal_slice, color="#0284c7", lw=1.2, label="Preprocessed ECG (500 Hz, mV)")
    ax1.set_title(f"1D ECG Lead Tracing (First {round(total_samples/fs, 1)}s)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_ylabel("Amplitude (mV)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")
    
    for i, w in enumerate(window_results):
        if w["start_time_sec"] < (total_samples / fs):
            color = "#ef4444" if w["prediction"] == "Hyperkalemia" else "#22c55e"
            ax1.axvspan(w["start_time_sec"], min(w["end_time_sec"], total_samples/fs), color=color, alpha=0.08)
    
    if window_results:
        win_times = [w["start_time_sec"] + 4.0 for w in window_results]
        hyp_probs = [w["p_hyperkalemia"] for w in window_results]
        norm_probs = [w["p_normal"] for w in window_results]
        
        ax2.plot(win_times, hyp_probs, marker="o", color="#dc2626", lw=2, label="P(Hyperkalemia) %")
        ax2.axhline(50.0, color="#6b7280", linestyle=":", label="Decision Threshold (50%)")
        ax2.set_ylim(-5, 105)
        ax2.set_xlabel("Recording Time (seconds)", fontsize=10)
        ax2.set_ylabel("Probability (%)", fontsize=10)
        ax2.set_title("Window-by-Window Model Confidence", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    return fig

def analyze_single_file(file_obj):
    if file_obj is None:
        return "⚠️ Please upload an ECG .txt file.", None, None, None, None
    
    filepath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    filename = os.path.basename(filepath)
    
    try:
        res = predictor.predict(filepath)
        
        pred_label = res["prediction"]
        conf = res["confidence"]
        p_norm = res["p_normal_pct"]
        p_hyp = res["p_hyperkalemia_pct"]
        risk = res["risk_level"]
        metrics = res["metrics"]
        
        if pred_label == "Hyperkalemia":
            badge_color = "#dc2626"
            icon = "🚨"
            desc = "Elevated T-waves / Hyperkalemia pattern detected across analysis windows."
        else:
            badge_color = "#16a34a"
            icon = "✅"
            desc = "Normal ECG waveform profile without characteristic hyperkalemic T-wave elevation."
            
        summary_html = f"""
        <div style="background-color: #f8fafc; border-radius: 12px; border: 2px solid {badge_color}; padding: 18px; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 26px; font-weight: 800; color: {badge_color};">{icon} {pred_label.upper()}</span>
                    <p style="margin: 4px 0 0 0; color: #475569; font-size: 14px;"><b>Clinical Impression:</b> {desc}</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 14px; color: #64748b;">Model Confidence</div>
                    <div style="font-size: 24px; font-weight: 700; color: #1e293b;">{conf}%</div>
                </div>
            </div>
            <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 12px 0;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                <div style="background: #ffffff; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b;">P(Normal)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #16a34a;">{p_norm}%</div>
                </div>
                <div style="background: #ffffff; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b;">P(Hyperkalemia)</div>
                    <div style="font-size: 16px; font-weight: 700; color: #dc2626;">{p_hyp}%</div>
                </div>
                <div style="background: #ffffff; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b;">Est. Heart Rate</div>
                    <div style="font-size: 16px; font-weight: 700; color: #0284c7;">{metrics.get('estimated_bpm', 'N/A')} BPM</div>
                </div>
                <div style="background: #ffffff; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 11px; color: #64748b;">8s Windows Evaluated</div>
                    <div style="font-size: 16px; font-weight: 700; color: #0f172a;">{res['n_windows']}</div>
                </div>
            </div>
        </div>
        """
        
        prob_dict = {
            "Normal": p_norm / 100.0,
            "Hyperkalemia": p_hyp / 100.0
        }
        
        df_windows = pd.DataFrame(res["window_results"])
        fig = plot_ecg_and_timeline(res["ecg_500hz"], res["window_results"])
        
        metrics_text = f"""
        **File Analyzed:** `{filename}`
        - **Source Sampling Rate:** `{res['original_sampling_rate_hz']} Hz` (Standardized to 500 Hz)
        - **Signal Duration:** `{metrics['duration_sec']} seconds` ({metrics['total_samples']} samples)
        - **Peak-to-Peak Amplitude:** `{metrics['p2p_amplitude_mv']} mV`
        - **Risk Stratification:** `{risk}`
        """
        
        return summary_html, prob_dict, fig, df_windows, metrics_text
        
    except Exception as e:
        err_msg = f"❌ Error analyzing file `{filename}`: {str(e)}"
        return err_msg, None, None, None, None

def analyze_batch_files(file_list):
    if not file_list:
        return None, "No files uploaded."
    
    rows = []
    for f_obj in file_list:
        path = f_obj.name if hasattr(f_obj, "name") else str(f_obj)
        fname = os.path.basename(path)
        try:
            res = predictor.predict(path)
            rows.append({
                "Filename": fname,
                "Prediction": res["prediction"],
                "Confidence (%)": res["confidence"],
                "P(Normal) %": res["p_normal_pct"],
                "P(Hyperkalemia) %": res["p_hyperkalemia_pct"],
                "Risk Stratification": res["risk_level"],
                "Duration (s)": res["metrics"]["duration_sec"],
                "Heart Rate (BPM)": res["metrics"].get("estimated_bpm", "N/A"),
                "Windows": res["n_windows"]
            })
        except Exception as e:
            rows.append({
                "Filename": fname,
                "Prediction": f"Error: {str(e)}",
                "Confidence (%)": 0.0,
                "P(Normal) %": 0.0,
                "P(Hyperkalemia) %": 0.0,
                "Risk Stratification": "Failed",
                "Duration (s)": 0,
                "Heart Rate (BPM)": "N/A",
                "Windows": 0
            })
            
    df_batch = pd.DataFrame(rows)
    return df_batch, f"Successfully processed {len(rows)} files."

with gr.Blocks(title="1D ECG Hyperkalemia Classifier") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 24px 10px; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; border-radius: 12px; margin-bottom: 20px;">
        <h1 style="color: #38bdf8; margin: 0; font-size: 28px; font-weight: 800;">🫀 1D ECG Hyperkalemia AI Diagnostic System</h1>
        <p style="color: #94a3b8; margin-top: 8px; font-size: 15px;">Deep Learning 1D-CNN + Attention Model for Automated Hyperkalemia Classification from Single-Lead ECG .txt Tracings</p>
    </div>
    """)
    
    with gr.Tabs():
        with gr.TabItem("🔬 Single Patient Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload ECG .txt Recording", file_types=[".txt", ".csv"])
                    analyze_btn = gr.Button("🔍 Run Hyperkalemia Analysis", variant="primary", size="lg")
                    metrics_output = gr.Markdown()
                    
                with gr.Column(scale=2):
                    summary_output = gr.HTML()
                    label_output = gr.Label(num_top_classes=2, label="Classification Probabilities")
                    plot_output = gr.Plot(label="ECG Tracing & Window Diagnostics")
                    
                    with gr.Accordion("📋 Detailed 8-Second Window Breakdown", open=False):
                        table_output = gr.DataFrame(headers=["window_idx", "start_time_sec", "end_time_sec", "p_normal", "p_hyperkalemia", "prediction"])

            analyze_btn.click(
                fn=analyze_single_file,
                inputs=[file_input],
                outputs=[summary_output, label_output, plot_output, table_output, metrics_output]
            )
            file_input.change(
                fn=analyze_single_file,
                inputs=[file_input],
                outputs=[summary_output, label_output, plot_output, table_output, metrics_output]
            )

        with gr.TabItem("📊 Batch Patient Processing"):
            gr.Markdown("### Upload multiple `.txt` ECG files for automated batch triage")
            batch_files = gr.File(label="Upload Multiple Patient Files", file_count="multiple", file_types=[".txt", ".csv"])
            batch_btn = gr.Button("⚡ Process Batch", variant="primary")
            batch_status = gr.Markdown()
            batch_table = gr.DataFrame(label="Batch Classification Results")
            
            batch_btn.click(
                fn=analyze_batch_files,
                inputs=[batch_files],
                outputs=[batch_table, batch_status]
            )

        with gr.TabItem("ℹ️ Model & Technical Specifications"):
            gr.Markdown("""
            ### 📐 Model Architecture & Clinical Pipeline
            
            - **Model Type**: 1D Convolutional Neural Network with 1D-Attention Mechanism.
            - **Input Shape**: `(4000, 1)` corresponding to **8 seconds** of single-lead ECG signal sampled at **500 Hz**.
            - **Feature Extractor**: 3x Conv1D layers (filters: 64, 128, 256) with LeakyReLU activations, Batch Normalization, and Dropout.
            - **Attention Block**: Custom 1D Attention scoring to focus on characteristic ECG morphological landmarks (T-wave amplitude, peaking, and symmetry).
            - **Decision Threshold**: Softmax probabilities for **Normal** vs **Hyperkalemia** (Threshold = 0.50).
            
            #### 🔄 Signal Preprocessing Pipeline:
            1. **Header Parsing**: Automatic detection of sampling rate (1000 Hz / 500 Hz) and sensor resolution (16-bit ADC counts vs mV).
            2. **Unit Conversion**: Automatic Cardiobanplux / Plux transfer function applied to raw counts:
               $$\\text{ECG (mV)} = \\left(\\frac{\\text{ADC}}{2^{16}} - 0.5\\right) \\times \\left(\\frac{V_{cc}}{\\text{Gain}}\\right) \\times 1000$$
            3. **Resampling**: Anti-aliased decimation / resampling from 1000 Hz source to 500 Hz target.
            4. **Filtering**: 0.5 Hz 4th-order Butterworth high-pass filter for baseline wander removal.
            5. **Standard Scaling**: Normalization using global training set distribution parameters.
            """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
