import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
from tabulate import tabulate
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# =========================
# 🔧 PATHS
# =========================
BASE_DIR = "D:/Android/DiplomApp"
cnn_path = os.path.join(BASE_DIR, "best_cnn_model.h5")
rcnn_path = os.path.join(BASE_DIR, "final_rcnn_model.h5")
lstm_path = os.path.join(BASE_DIR, "lstm_model.keras")
rf_path = os.path.join(BASE_DIR, "random_forest_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
wav2vec_path = os.path.join(BASE_DIR, "wav2vec_model.pth")
audio_folder = os.path.join(BASE_DIR, "freetts")  # Папка с аудио

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 🔧 ATTENTION (CNN/RCNN/LSTM)
# =========================
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform")
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer="zeros")

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


# =========================
# 📥 LOAD KERAS MODELS
# =========================
def load_model(path):
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False
        )
    except Exception as e:
        print(f"Error loading Keras model {path}: {e}")
        return None


cnn = load_model(cnn_path)
rcnn = load_model(rcnn_path)
lstm = load_model(lstm_path)

rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


# =========================
# 🧠 WAV2VEC MODEL
# =========================
wav2vec_model = None
wav2vec_processor = None

try:
    # Пробуем загрузить как модель HuggingFace (стандартный Wav2Vec2)
    # Если файл .pth содержит state_dict, нужно сначала создать модель
    wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model.load_state_dict(torch.load(wav2vec_path, map_location=DEVICE))
    wav2vec_model.to(DEVICE)
    wav2vec_model.eval()
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("Wav2Vec model loaded successfully (HuggingFace).")
except Exception as e:
    print(f"Could not load Wav2Vec via HuggingFace: {e}")
    print("Attempting to load as a custom PyTorch model...")
    try:
        # Если кастомная модель, здесь нужно определить класс
        # Пока оставляем None, чтобы скрипт не падал
        pass
    except Exception as e2:
        print(f"Failed to load Wav2Vec: {e2}")


# =========================
# 🎧 CNN FEATURES (40x200)
# =========================
def mfcc_cnn(file, max_len=200):
    audio, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc[..., np.newaxis]


# =========================
# 🎧 RCNN FEATURES (40x300x3)
# =========================
def mfcc_rcnn(file, max_len=300):
    audio, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]

    return np.stack([mfcc, mfcc, mfcc], axis=-1)


# =========================
# 🎧 LSTM FEATURES (300x40)
# =========================
def mfcc_lstm(file, max_len=300):
    audio, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.T


# =========================
# 🧠 RF FEATURES (≈280)
# =========================
def rf_features(file):
    audio, sr = librosa.load(file, sr=16000)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel)

    feats = []

    for f in [mfcc, delta, delta2]:
        feats.extend(np.mean(f, axis=1))
        feats.extend(np.std(f, axis=1))

    feats.extend(np.mean(log_mel, axis=1))
    feats.extend(np.std(log_mel, axis=1))

    return np.array(feats)


# =========================
# 🎧 WAV2VEC FEATURES
# =========================
def predict_wav2vec(file):
    if wav2vec_model is None or wav2vec_processor is None:
        return None
    try:
        audio, sr = librosa.load(file, sr=16000)
        inputs = wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        
        # Предполагаем, что выход 0 - REAL, 1 - SPOOF
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return float(probs[0][1])  # Вероятность SPOOF
    except Exception as e:
        print(f"Wav2Vec prediction error: {e}")
        return None


# =========================
# 🎨 OUTPUT FORMAT
# =========================
def fmt(model, prob):
    if prob is None:
        return [model, "SKIP", ""]
    label = "SPOOF" if prob > 0.5 else "REAL"
    return [model, f"{prob:.3f}", label]


# =========================
# 🚀 RUN
# =========================
files = [f for f in os.listdir(audio_folder)
         if f.endswith((".mp3", ".wav", ".flac"))]

if not files:
    print(f"No audio files found in {audio_folder}")
else:
    for f in tqdm(files):
        path = os.path.join(audio_folder, f)

        print("\nFILE:", f)
        results = []

        # ================= CNN =================
        try:
            if cnn:
                x = mfcc_cnn(path)
                p = float(cnn.predict(np.expand_dims(x, 0), verbose=0)[0][0])
                results.append(fmt("CNN", p))
            else:
                results.append(["CNN", "NO MODEL", ""])
        except Exception as e:
            results.append(["CNN", "ERROR", str(e)[:20]])

        # ================= RCNN =================
        try:
            if rcnn:
                x = mfcc_rcnn(path)
                p = float(rcnn.predict(np.expand_dims(x, 0), verbose=0)[0][0])
                results.append(fmt("RCNN", p))
            else:
                results.append(["RCNN", "NO MODEL", ""])
        except Exception as e:
            results.append(["RCNN", "ERROR", str(e)[:20]])

        # ================= LSTM =================
        try:
            if lstm:
                x = mfcc_lstm(path)
                p = float(lstm.predict(np.expand_dims(x, 0), verbose=0)[0][0])
                results.append(fmt("LSTM", p))
            else:
                results.append(["LSTM", "NO MODEL", ""])
        except Exception as e:
            results.append(["LSTM", "ERROR", str(e)[:20]])

        # ================= RF =================
        try:
            if rf:
                x = rf_features(path)

                EXPECTED_DIM = scaler.n_features_in_ if scaler is not None else 280

                if len(x) < EXPECTED_DIM:
                    x = np.pad(x, (0, EXPECTED_DIM - len(x)))
                else:
                    x = x[:EXPECTED_DIM]

                x = x.reshape(1, -1)

                if scaler:
                    x = scaler.transform(x)

                if hasattr(rf, "predict_proba"):
                    p = rf.predict_proba(x)[0][1]
                else:
                    p = float(rf.predict(x)[0])

                results.append(fmt("RF", p))
            else:
                results.append(["RF", "NO MODEL", ""])
        except Exception as e:
            results.append(["RF", "ERROR", str(e)[:20]])

        # ================= WAV2VEC =================
        p_wav2vec = predict_wav2vec(path)
        results.append(fmt("Wav2Vec", p_wav2vec))

        # ================= PRINT =================
        print(tabulate(
            results,
            headers=["MODEL", "PROB", "RESULT"],
            tablefmt="fancy_grid"
        ))
        print("=" * 70)
