import os
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
import joblib

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

try:
    import keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        keras = None

DEFAULT_SCALER_MEAN = 4.90279664e-05
DEFAULT_SCALER_SCALE = 0.13823536

HF_ONNX_URL = "https://huggingface.co/Deepak932/hyperkalemia-1d-ecg-model/resolve/main/final_hyperkalemia_model.onnx"
HF_SCALER_URL = "https://huggingface.co/Deepak932/hyperkalemia-1d-ecg-model/resolve/main/training_scaler.pkl"

class HyperkalemiaPredictor:
    def __init__(self, model_path=None, scaler_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "final_hyperkalemia_model.onnx")
        if scaler_path is None:
            scaler_path = os.path.join(base_dir, "training_scaler.pkl")

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.session = None
        self.input_name = None
        self.keras_model = None
        self.scaler = None
        self._load_model_and_scaler()

    def _download_if_missing(self, file_path, url):
        if not os.path.exists(file_path):
            print(f"Downloading {os.path.basename(file_path)} from Cloud Storage ({url})...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {os.path.basename(file_path)} successfully!")

    def _load_model_and_scaler(self):
        self._download_if_missing(self.model_path, HF_ONNX_URL)
        self._download_if_missing(self.scaler_path, HF_SCALER_URL)

        if onnxruntime is not None and self.model_path.endswith(".onnx"):
            print(f"Loading ONNX model from {self.model_path}...")
            self.session = onnxruntime.InferenceSession(self.model_path)
            self.input_name = self.session.get_inputs()[0].name
            print("ONNX Inference Engine ready!")
        elif keras is not None:
            keras_path = self.model_path.replace(".onnx", ".keras")
            if os.path.exists(keras_path):
                self.keras_model = keras.models.load_model(keras_path)
                print("Keras model loaded as fallback!")
        else:
            raise RuntimeError("Neither onnxruntime nor keras is available to load the model.")

        if os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                print("Scaler loaded successfully from file!")
            except Exception as e:
                print(f"Warning: Failed to load scaler via joblib ({e}). Using exact fallback scaler...")
                self.scaler = None
        else:
            self.scaler = None

    def parse_header(self, filepath_or_buffer):
        sampling_rate = 1000
        n_bits = 16
        columns = ["nSeq", "ECG"]

        try:
            if isinstance(filepath_or_buffer, str):
                with open(filepath_or_buffer, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [f.readline() for _ in range(15)]
            else:
                pos = filepath_or_buffer.tell()
                lines = [filepath_or_buffer.readline().decode("utf-8", errors="ignore") for _ in range(15)]
                filepath_or_buffer.seek(pos)

            for line in lines:
                if line.startswith("#") and line.strip().startswith("# {"):
                    meta = json.loads(line.strip().lstrip("#").strip())
                    dev_key = list(meta.keys())[0]
                    info = meta[dev_key]
                    columns = info.get("column", columns)
                    sampling_rate = info.get("sampling rate", 1000)
                    res = info.get("resolution", [16])
                    n_bits = res[0] if isinstance(res, list) and len(res) > 0 else 16
                    break
                if line.strip() == "# EndOfHeader":
                    break
        except Exception as e:
            print(f"Header parsing fallback used: {e}")

        return columns, float(sampling_rate), int(n_bits)

    def load_ecg_signal(self, file_path_or_buffer, vcc=3.0, gain=1000.0):
        columns, fs_declared, n_bits = self.parse_header(file_path_or_buffer)

        try:
            df = pd.read_csv(file_path_or_buffer, sep=r"\s+|,|\t", engine="python", comment="#", header=None)
        except Exception:
            if isinstance(file_path_or_buffer, str):
                df = pd.read_csv(file_path_or_buffer, comment="#", delim_whitespace=True, header=None)
            else:
                file_path_or_buffer.seek(0)
                df = pd.read_csv(file_path_or_buffer, comment="#", delim_whitespace=True, header=None)

        df = df.dropna()
        if df.shape[1] >= 2:
            c0 = df.iloc[:, 0].values.astype(float)
            ecg_raw = df.iloc[:, 1].values.astype(float)
        else:
            c0 = np.arange(len(df))
            ecg_raw = df.iloc[:, 0].values.astype(float)

        is_timestamp_col = "timestamp" in [str(c).lower() for c in columns]
        max_abs = np.nanmax(np.abs(ecg_raw))
        if is_timestamp_col or (max_abs < 50.0):
            ecg_mv = ecg_raw
            dt = np.diff(c0[:20000])
            dt = dt[dt > 0]
            fs = 1000.0 / np.median(dt) if len(dt) > 0 else fs_declared
        else:
            ecg_mv = ((ecg_raw / (2**n_bits)) - 0.5) * (vcc / gain) * 1000.0
            fs = fs_declared

        return ecg_mv.astype(np.float64), float(fs)

    def preprocess_signal(self, ecg_mv, fs_source, target_fs=500.0, highpass_cutoff=0.5):
        if abs(fs_source - target_fs) > 1e-6:
            factor = fs_source / target_fs
            if float(factor).is_integer():
                ecg_resampled = sp_signal.decimate(ecg_mv, int(factor))
            else:
                n_target = int(round(len(ecg_mv) * target_fs / fs_source))
                ecg_resampled = sp_signal.resample(ecg_mv, n_target)
        else:
            ecg_resampled = ecg_mv

        b, a = sp_signal.butter(4, highpass_cutoff / (0.5 * target_fs), btype="high")
        ecg_filtered = sp_signal.filtfilt(b, a, ecg_resampled)
        return ecg_filtered

    def make_windows(self, ecg_500hz, window=4000, stride=2000):
        n = len(ecg_500hz)
        starts = list(range(0, n - window + 1, stride))
        if not starts:
            seg = np.pad(ecg_500hz, (0, max(0, window - n)), mode="edge")[:window]
            return seg[None, :, None], np.array([0.0])

        segs = np.stack([ecg_500hz[s:s+window] for s in starts], axis=0)
        time_offsets = np.array([s / 500.0 for s in starts])
        return segs[:, :, None], time_offsets

    def scale_windows(self, windows):
        n_win, win_len, _ = windows.shape
        flat = windows.reshape(-1, 1)
        if self.scaler is not None:
            flat_scaled = self.scaler.transform(flat)
        else:
            flat_scaled = (flat - DEFAULT_SCALER_MEAN) / DEFAULT_SCALER_SCALE
        return flat_scaled.reshape(n_win, win_len, 1).astype(np.float32)

    def compute_signal_metrics(self, ecg_500hz):
        duration_sec = len(ecg_500hz) / 500.0
        p2p_amplitude = float(np.nanmax(ecg_500hz) - np.nanmin(ecg_500hz))

        try:
            peaks, _ = sp_signal.find_peaks(ecg_500hz, distance=180, prominence=0.25 * (p2p_amplitude or 1.0))
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / 500.0
                mean_rr = np.median(rr_intervals)
                bpm = round(60.0 / mean_rr, 1) if mean_rr > 0 else None
            else:
                bpm = None
        except Exception:
            bpm = None

        return {
            "duration_sec": round(duration_sec, 2),
            "p2p_amplitude_mv": round(p2p_amplitude, 3),
            "estimated_bpm": bpm,
            "total_samples": len(ecg_500hz)
        }

    def predict(self, file_path_or_buffer, window=4000, stride=2000):
        ecg_mv, fs_orig = self.load_ecg_signal(file_path_or_buffer)
        ecg_500 = self.preprocess_signal(ecg_mv, fs_orig)
        windows, time_offsets = self.make_windows(ecg_500, window=window, stride=stride)

        windows_scaled = self.scale_windows(windows)
        
        if self.session is not None:
            probs = self.session.run(None, {self.input_name: windows_scaled})[0]
        else:
            probs = self.keras_model.predict(windows_scaled, verbose=0)

        avg_probs = np.mean(probs, axis=0)
        p_normal = float(avg_probs[0])
        p_hyper = float(avg_probs[1])

        prediction_label = "Hyperkalemia" if p_hyper >= 0.50 else "Normal"
        confidence = max(p_normal, p_hyper)

        if prediction_label == "Hyperkalemia":
            if p_hyper >= 0.85:
                risk_level = "High Risk (Marked Elevation / Peaked T-waves)"
            elif p_hyper >= 0.65:
                risk_level = "Moderate Risk (Hyperkalemia Probable)"
            else:
                risk_level = "Mild / Borderline Elevation"
        else:
            if p_normal >= 0.80:
                risk_level = "Normal ECG Pattern"
            else:
                risk_level = "Borderline Normal"

        metrics = self.compute_signal_metrics(ecg_500)

        window_results = []
        for i in range(len(probs)):
            w_norm = float(probs[i, 0])
            w_hyp = float(probs[i, 1])
            window_results.append({
                "window_idx": i + 1,
                "start_time_sec": round(float(time_offsets[i]), 2),
                "end_time_sec": round(float(time_offsets[i]) + 8.0, 2),
                "p_normal": round(w_norm * 100.0, 2),
                "p_hyperkalemia": round(w_hyp * 100.0, 2),
                "prediction": "Hyperkalemia" if w_hyp >= 0.50 else "Normal"
            })

        return {
            "prediction": prediction_label,
            "confidence": round(confidence * 100.0, 2),
            "p_normal_pct": round(p_normal * 100.0, 2),
            "p_hyperkalemia_pct": round(p_hyper * 100.0, 2),
            "risk_level": risk_level,
            "n_windows": len(windows),
            "original_sampling_rate_hz": fs_orig,
            "metrics": metrics,
            "ecg_500hz": ecg_500,
            "window_results": window_results
        }
