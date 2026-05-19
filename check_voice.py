import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import joblib
import torch
import torch.nn as nn
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 🔧 ATTENTION LAYER
# =========================
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="glorot_uniform")
        self.b = self.add_weight(shape=(input_shape[1], 1), initializer="zeros")

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

# =========================
# 📥 LOAD MODELS
# =========================
def load_keras_model(path):
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False
        )
    except Exception as e:
        print(f"[!] Ошибка загрузки {path}: {e}")
        return None

print("Загрузка моделей...")
cnn = load_keras_model(cnn_path)
rcnn = load_keras_model(rcnn_path)
lstm = load_keras_model(lstm_path)

rf = None
scaler = None
try:
    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    print("RF и Scaler загружены.")
except Exception as e:
    print(f"[!] Ошибка загрузки RF/Scaler: {e}")

# Wav2Vec
wav2vec_model = None
wav2vec_processor = None
try:
    wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model.load_state_dict(torch.load(wav2vec_path, map_location=DEVICE))
    wav2vec_model.to(DEVICE)
    wav2vec_model.eval()
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("Wav2Vec загружен (HuggingFace).")
except Exception as e:
    print(f"[!] Wav2Vec не загружен: {e}")

# =========================
# 🎧 FEATURE FUNCTIONS
# =========================
def load_audio(path):
    audio, sr = librosa.load(path, sr=16000)
    return audio, sr

def mfcc_cnn(audio, max_len=200):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc[..., np.newaxis]

def mfcc_rcnn(audio, max_len=300):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return np.stack([mfcc, mfcc, mfcc], axis=-1)

def mfcc_lstm(audio, max_len=300):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0), (0, max_len - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

def rf_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
    log_mel = librosa.power_to_db(mel)
    
    feats = []
    for f in [mfcc, delta, delta2]:
        feats.extend(np.mean(f, axis=1))
        feats.extend(np.std(f, axis=1))
    feats.extend(np.mean(log_mel, axis=1))
    feats.extend(np.std(log_mel, axis=1))
    return np.array(feats)

# =========================
# 🚀 PREDICTION FUNCTIONS
# =========================
def predict_cnn(audio):
    if cnn is None: return None
    x = mfcc_cnn(audio)
    return float(cnn.predict(np.expand_dims(x, 0), verbose=0)[0][0])

def predict_rcnn(audio):
    if rcnn is None: return None
    x = mfcc_rcnn(audio)
    return float(rcnn.predict(np.expand_dims(x, 0), verbose=0)[0][0])

def predict_lstm(audio):
    if lstm is None: return None
    x = mfcc_lstm(audio)
    return float(lstm.predict(np.expand_dims(x, 0), verbose=0)[0][0])

def predict_rf(audio):
    if rf is None: return None
    x = rf_features(audio)
    EXPECTED_DIM = scaler.n_features_in_ if scaler is not None else 280
    if len(x) < EXPECTED_DIM:
        x = np.pad(x, (0, EXPECTED_DIM - len(x)))
    else:
        x = x[:EXPECTED_DIM]
    x = x.reshape(1, -1)
    if scaler:
        x = scaler.transform(x)
    if hasattr(rf, "predict_proba"):
        return float(rf.predict_proba(x)[0][1])
    return float(rf.predict(x)[0])

def predict_wav2vec(audio):
    if wav2vec_model is None or wav2vec_processor is None:
        return None
    try:
        inputs = wav2vec_processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return float(probs[0][1])  # SPOOF prob
    except Exception as e:
        print(f"    [!] Wav2Vec error: {e}")
        return None

# =========================
# 🎨 RESULT FORMATTING
# =========================
def get_result(prob):
    if prob is None: return "N/A", "---"
    label = "SPOOF" if prob > 0.5 else "REAL"
    return label, f"{prob:.3f}"

# =========================
# 🖥️ MAIN MENU
# =========================
def main():
    print("=" * 50)
    print("   VOICE FAKE DETECTION SYSTEM")
    print("=" * 50)
    
    # 1. Get file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Введите путь к аудиофайлу: ").strip().strip('"')
    
    if not os.path.exists(file_path):
        print("[!] Файл не найден.")
        return
    
    print(f"\nЗагрузка аудио: {file_path}")
    try:
        audio, sr = load_audio(file_path)
        duration = len(audio) / sr
        print(f"    Длительность: {duration:.2f} сек, SR: {sr}Hz")
    except Exception as e:
        print(f"[!] Ошибка загрузки аудио: {e}")
        return
    
    # 2. Select mode
    print("\nВыберите режим проверки:")
    print("  1. Быстрая (Enhanced RF)")
    print("  2. Долгая (все 5 моделей)")
    print("  3. Выбрать модель (1 из 5)")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    results = {}
    
    if choice == "1":
        # Fast: Only RF
        print("\n[⚡] Быстрая проверка (RF)...")
        p = predict_rf(audio)
        label, prob_str = get_result(p)
        print(f"\n✅ РЕЗУЛЬТАТ: {label}")
        print(f"   Вероятность (SPOOF): {prob_str}")
        
    elif choice == "2":
        # Full: All 5 models
        print("\n[🔍] Долгая проверка (5 моделей)...\n")
        
        models = {
            "CNN": predict_cnn,
            "RCNN": predict_rcnn,
            "LSTM": predict_lstm,
            "RF": predict_rf,
            "Wav2Vec": predict_wav2vec
        }
        
        for name, func in models.items():
            print(f"   Проверка {name}...", end=" ")
            p = func(audio)
            label, prob_str = get_result(p)
            results[name] = p
            print(f"{label} ({prob_str})")
        
        # Average result
        valid_probs = [p for p in results.values() if p is not None]
        if valid_probs:
            avg_prob = np.mean(valid_probs)
            final_label = "SPOOF" if avg_prob > 0.5 else "REAL"
            print("\n" + "=" * 50)
            print(f"✅ ИТОГОВЫЙ РЕЗУЛЬТАТ: {final_label}")
            print(f"   Средняя вероятность (SPOOF): {avg_prob:.3f}")
            print(f"   Моделей использовано: {len(valid_probs)}")
            print("=" * 50)
        
    elif choice == "3":
        # Select specific model
        print("\nВыберите модель:")
        print("  1. CNN")
        print("  2. RCNN")
        print("  3. LSTM")
        print("  4. RF (Random Forest)")
        print("  5. Wav2Vec")
        
        model_choice = input("Ваш выбор (1-5): ").strip()
        
        model_map = {
            "1": ("CNN", predict_cnn),
            "2": ("RCNN", predict_rcnn),
            "3": ("LSTM", predict_lstm),
            "4": ("RF", predict_rf),
            "5": ("Wav2Vec", predict_wav2vec)
        }
        
        if model_choice in model_map:
            name, func = model_map[model_choice]
            print(f"\n[🔍] Проверка моделью {name}...")
            p = func(audio)
            label, prob_str = get_result(p)
            print(f"\n✅ РЕЗУЛЬТАТ: {label}")
            print(f"   Модель: {name}")
            print(f"   Вероятность (SPOOF): {prob_str}")
        else:
            print("[!] Неверный выбор.")
    else:
        print("[!] Неверный выбор.")

if __name__ == "__main__":
    main()
