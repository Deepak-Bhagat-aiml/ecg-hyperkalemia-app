---
title: 1D ECG Hyperkalemia Classifier
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🫀 1D ECG Hyperkalemia Classifier Web App

A cloud-ready Web Application for automated detection and classification of **Hyperkalemia vs. Normal** from 1D single-lead ECG recordings (`.txt` files).

## 🌟 Key Features
- **Instant .txt File Processing**: Drag and drop any raw OpenSignals / Cardiobanplux `.txt` or generic single-lead ECG text/CSV file.
- **Built-in Signal Conditioning**: Automatic sampling rate detection (1000 Hz / 500 Hz), ADC unit conversion to millivolts (mV), and 0.5 Hz Butterworth baseline wander filtering.
- **Attention-Guided 1D-CNN**: Evaluates 8-second sliding windows (4000 samples @ 500 Hz) using deep feature maps and attention scoring.
- **Interactive Visualizations**: View the full preprocessed ECG trace and window-by-window confidence timeline.
- **Batch Processing**: Upload multiple patient records simultaneously and export a diagnostic summary table.

## 🚀 Cloud Deployment (Share with Anyone)

To run this app on the cloud so **anyone can access it from any laptop without connecting to your PC**:

### Method 1: Automated 1-Click Script
Run in terminal:
```bash
python deploy_hf.py
```
Enter your free Hugging Face token, and the app will be live at `https://huggingface.co/spaces/<your-username>/ecg-hyperkalemia-detector`.

### Method 2: Manual Upload to Hugging Face Spaces (Free)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select **Gradio** as the Space SDK.
3. Upload all files from this folder (`app.py`, `hyperkalemia_engine.py`, `final_hyperkalemia_model.keras`, `training_scaler.pkl`, `requirements.txt`, `sample_data/`).
4. Hugging Face builds and hosts your app 24/7 on high-speed cloud servers for free. Share the URL with anyone!
