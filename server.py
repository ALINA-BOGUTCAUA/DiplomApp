import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import io
import tempfile

app = Flask(__name__)
CORS(app)

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
# 📥 LOAD MODELS
# =========================
def load_model(path):
    try:
        return tf.keras.models.load_model(
            path,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False
        )
    except Exception as e:
        print(f"[!] Error loading {path}: {e}")
        return None

print("Loading models...")
cnn = load_model(cnn_path)
rcnn = load_model(rcnn_path)
lstm = load_model(lstm_path)

rf = None
scaler = None
try:
    rf = joblib.load(rf_path) if os.path.exists(rf_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    print("RF and Scaler loaded.")
except Exception as e:
    print(f"[!] Error loading RF/Scaler: {e}")

# Wav2Vec
wav2vec_model = None
wav2vec_processor = None
try:
    wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base")
    wav2vec_model.load_state_dict(torch.load(wav2vec_path, map_location=DEVICE))
    wav2vec_model.to(DEVICE)
    wav2vec_model.eval()
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    print("Wav2Vec loaded (HuggingFace).")
except Exception as e:
    print(f"[!] Wav2Vec not loaded: {e}")

# =========================
# 🎧 FEATURE FUNCTIONS
# =========================
def load_audio_from_bytes(file_bytes, sr=16000):
    import soundfile as sf
    try:
        with io.BytesIO(file_bytes) as buf:
            audio, sr = librosa.load(buf, sr=sr)
        return audio, sr
    except:
        # Fallback: save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        audio, sr = librosa.load(tmp_path, sr=sr)
        os.unlink(tmp_path)
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
        return float(probs[0][1])
    except Exception as e:
        print(f"Wav2Vec error: {e}")
        return None

# =========================
# 🚀 FLASK ROUTES
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        mode = request.form.get('mode', 'quick')
        model_name = request.form.get('model_name', 'CNN')
        
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Load audio
        audio, sr = load_audio_from_bytes(audio_bytes)
        
        result = {}
        results_list = []
        
        if mode == 'quick':
            # Only CNN
            prob = predict_cnn(audio)
            if prob is not None:
                label = "SPOOF" if prob > 0.5 else "REAL"
                return jsonify({
                    "is_fake": bool(prob > 0.5),
                    "confidence": prob,
                    "model_used": "CNN",
                    "result_text": label,
                    "prob": prob
                })
            else:
                return jsonify({"error": "CNN model not available"}), 500
        
        elif mode == 'full':
            # All 5 models
            models = {
                "CNN": predict_cnn,
                "RCNN": predict_rcnn,
                "LSTM": predict_lstm,
                "RF": predict_rf,
                "Wav2Vec": predict_wav2vec
            }
            
            probs = []
            model_names = []
            
            for name, func in models.items():
                try:
                    p = func(audio)
                    if p is not None:
                        probs.append(p)
                        model_names.append(name)
                except Exception as e:
                    print(f"Error in {name}: {e}")
            
            if probs:
                avg_prob = np.mean(probs)
                label = "SPOOF" if avg_prob > 0.5 else "REAL"
                return jsonify({
                    "is_fake": bool(avg_prob > 0.5),
                    "confidence": float(avg_prob),
                    "model_used": "All models: " + ", ".join(model_names),
                    "result_text": label,
                    "prob": float(avg_prob),
                    "all_probs": {name: p for name, p in zip(model_names, probs)}
                })
            else:
                return jsonify({"error": "No models available"}), 500
        
        elif mode == 'model':
            # Specific model
            print(f"[DEBUG] Requested model: {model_name}")
            predict_func = None
            if model_name == "CNN":
                predict_func = predict_cnn
                print(f"[DEBUG] CNN model loaded: {cnn is not None}")
            elif model_name == "RCNN":
                predict_func = predict_rcnn
                print(f"[DEBUG] RCNN model loaded: {rcnn is not None}")
            elif model_name == "LSTM":
                predict_func = predict_lstm
                print(f"[DEBUG] LSTM model loaded: {lstm is not None}")
            elif model_name == "RF":
                predict_func = predict_rf
                print(f"[DEBUG] RF model loaded: {rf is not None}")
            elif model_name == "Wav2Vec":
                predict_func = predict_wav2vec
                print(f"[DEBUG] Wav2Vec model loaded: {wav2vec_model is not None}")
            
            if predict_func:
                try:
                    prob = predict_func(audio)
                    print(f"[DEBUG] {model_name} prediction result: {prob}")
                    if prob is not None:
                        label = "SPOOF" if prob > 0.5 else "REAL"
                        return jsonify({
                            "is_fake": bool(prob > 0.5),
                            "confidence": prob,
                            "model_used": model_name,
                            "result_text": label,
                            "prob": prob
                        })
                    else:
                        return jsonify({"error": f"{model_name} model not available"}), 500
                except Exception as e:
                    print(f"[ERROR] Error predicting with {model_name}: {e}")
                    return jsonify({"error": f"Error with {model_name}: {str(e)}"}), 500
            else:
                print(f"[DEBUG] Invalid model name: {model_name}")
                return jsonify({"error": f"Invalid model name: {model_name}"}), 400
        
        return jsonify({"error": "Invalid mode"}), 400
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            "CNN": cnn is not None,
            "RCNN": rcnn is not None,
            "LSTM": lstm is not None,
            "RF": rf is not None,
            "Wav2Vec": wav2vec_model is not None
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("   VOICE FAKE DETECTION SERVER")
    print("=" * 50)
    print("Starting server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
