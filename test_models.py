import os
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import io

MODEL_PATH = "app/src/main/assets/"

def load_audio(filename):
    try:
        path = os.path.join(".", "freetts", filename)
        print(f"Loading: {path}")
        
        full_path = os.path.abspath(path)
        print(f"Full path: {full_path}")
        print(f"Exists: {os.path.exists(full_path)}")
        
        with open(full_path, 'rb') as f:
            audio_data = f.read()
        
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        print(f"Samples: {len(samples)}")
        
        if len(samples) > 16000:
            samples = samples[:16000]
        elif len(samples) < 16000:
            samples = np.pad(samples, (0, 16000 - len(samples)))
        
        return samples
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

def compute_mel_spectrogram(audio, n_mels=40, n_fft=512, hop_length=160):
    n_frames = len(audio) // hop_length
    
    spec = []
    for i in range(min(n_frames, 200)):
        frame = audio[i * hop_length:(i + 1) * hop_length]
        fft = np.fft.rfft(frame, n_fft)
        magnitude = np.abs(fft)[:n_fft // 2 + 1]
        
        mel_basis = np.random.random((n_mels, n_fft // 2 + 1)) * 0.01
        mel = np.dot(mel_basis, magnitude)
        mel = np.log(mel + 1e-10)
        spec.append(mel)
    
    while len(spec) < 200:
        spec.append(np.zeros(n_mels))
    
    spec = np.array(spec[:200]).T
    spec = spec.reshape(1, 40, 200, 1)
    return spec.astype(np.float32)

def test_cnn_model(model_path, audio_data):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_data = compute_mel_spectrogram(audio_data).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        return float(output[0][0])
    except Exception as e:
        print(f"CNN error: {e}")
        return 0.5

def test_random_forest(audio_data):
    try:
        audio = np.array(audio_data)
        
        sum_val = np.sum(audio)
        sum_squares = np.sum(audio ** 2)
        zero_count = np.sum(np.abs(audio) < 0.001)
        
        variance = (sum_squares / len(audio)) - ((sum_val / len(audio)) ** 2)
        std_dev = np.sqrt(max(variance, 0))
        zero_ratio = zero_count / len(audio)
        
        is_suspicious = variance < 0.001 or std_dev > 0.3 or zero_ratio > 0.5
        if is_suspicious:
            confidence = 0.3 + min(std_dev, 0.3)
        else:
            confidence = 0.7 - min(zero_ratio * 0.3, 0.3)
        
        return np.clip(confidence, 0, 1)
    except:
        return 0.5

def main():
    files = [f for f in os.listdir("freetts") if f.endswith('.mp3')]
    
    print("=" * 60)
    print(f"Тестирование {len(files)} файлов")
    print("=" * 60)
    
    results = []
    
    for i, filename in enumerate(files[:14], 1):
        print(f"\n[{i}/{len(files)}] {filename[:25]}")
        
        audio = load_audio(filename)
        if audio is None:
            continue
        
        conf_cnn = test_cnn_model(MODEL_PATH + "cnn_model.tflite", audio)
        is_fake_cnn = conf_cnn > 0.5
        print(f"  CNN: {conf_cnn:.3f} ({'FAKE' if is_fake_cnn else 'REAL'})")
        
        conf_rf = test_random_forest(audio)
        is_fake_rf = conf_rf < 0.5
        print(f"  RF: {conf_rf:.3f} ({'FAKE' if is_fake_rf else 'REAL'})")
        
        fake_count = (1 if is_fake_cnn else 0) + (1 if is_fake_rf else 0)
        result = "FAKE" if fake_count >= 1 else "REAL"
        
        results.append((filename, result, fake_count))
        print(f"  => {result}")
    
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print("=" * 60)
    
    fake_total = sum(1 for _, r, _ in results if r == "FAKE")
    real_total = sum(1 for _, r, _ in results if r == "REAL")
    
    print(f"Всего: {len(results)}")
    print(f"FAKE: {fake_total} ({fake_total/len(results)*100:.1f}%)")
    print(f"REAL: {real_total} ({real_total/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()