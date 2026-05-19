import os
import numpy as np
import struct

def test_with_wav_header():
    filename = "freetts/24fcb7c1807f6d675fac739553edbf4a.mp3"
    
    print("Testing with raw MP3 decode...")
    
    try:
        import subprocess
        result = subprocess.run(
            ['python', '-c', '''
import sys
import wave
import numpy as np
import struct

with wave.open("test_output.wav", "rb") as w:
    frames = w.readframes(w.getnframes())
    data = struct.unpack("<" + "h" * w.getnframes(), frames)
    print(f"Loaded {len(data)} samples")
'''],
            capture_output=True,
            text=True,
            cwd="."
        )
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
        return
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nUsing basic test instead...")
    
    np.random.seed(42)
    
    test_audios = {
        "uniform": np.random.uniform(-0.3, 0.3, 16000).astype(np.float32),
        "natural": np.random.randn(16000).astype(np.float32) * 0.3,
        "silence": np.zeros(16000, dtype=np.float32),
    }
    
    import tensorflow as tf
    
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
        spec = np.array(spec[:200]).T.reshape(1, 40, 200, 1)
        return spec.astype(np.float32)
    
    interpreter = tf.lite.Interpreter("app/src/main/assets/cnn_model.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nTest results:")
    for name, audio in test_audios.items():
        spec = compute_mel_spectrogram(audio)
        interpreter.set_tensor(input_details[0]['index'], spec)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        conf = float(output[0][0])
        print(f"  {name}: {conf:.3f} ({'FAKE' if conf > 0.5 else 'REAL'})")

test_with_wav_header()