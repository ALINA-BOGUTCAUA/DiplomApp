import numpy as np
import tensorflow as tf

MODEL_PATH = "app/src/main/assets/"

def test_all_models():
    # Load models
    cnn = tf.lite.Interpreter(MODEL_PATH + "cnn_model.tflite")
    cnn.allocate_tensors()
    cnn_in = cnn.get_input_details()[0]
    cnn_out = cnn.get_output_details()[0]
    print(f"CNN input shape: {cnn_in['shape']}")
    
    lstm = tf.lite.Interpreter(MODEL_PATH + "modelLSTM.tflite")
    lstm.allocate_tensors()
    lstm_in = lstm.get_input_details()[0]
    lstm_out = lstm.get_output_details()[0]
    print(f"LSTM input shape: {lstm_in['shape']}")
    
    rcnn = tf.lite.Interpreter(MODEL_PATH + "rcnn_model.tflite")
    rcnn.allocate_tensors()
    rcnn_in = rcnn.get_input_details()[0]
    rcnn_out = rcnn.get_output_details()[0]
    print(f"RCNN input shape: {rcnn_in['shape']}")
    
    # Test data
    audio = np.random.randn(16000).astype(np.float32)
    
    print("\n" + "="*50)
    print("Testing different input formats:")
    print("="*50)
    
    # Format 1: Raw [1, 16000]
    print("\n[1] Raw [1,16000]:")
    try:
        inp = np.array([audio])
        cnn.set_tensor(cnn_in['index'], inp)
        cnn.invoke()
        out = cnn.get_tensor(cnn_out['index'])
        print(f"  CNN: {out[0]}")
    except Exception as e:
        print(f"  CNN error: {e}")
    
    # Test LSTM
    try:
        lstm.set_tensor(lstm_in['index'], inp)
        lstm.invoke()
        out = lstm.get_tensor(lstm_out['index'])
        print(f"  LSTM: {out[0]}")
    except Exception as e:
        print(f"  LSTM error: {e}")

    # Format 2: Mel spectrogram [1, 40, 200, 1]
    print("\n[2] Mel [1,40,200,1]:")
    try:
        # Create mel-like spectrogram
        mel = np.random.randn(1, 40, 200, 1).astype(np.float32) * 0.1
        cnn.set_tensor(cnn_in['index'], mel)
        cnn.invoke()
        out = cnn.get_tensor(cnn_out['index'])
        print(f"  CNN: {out[0]}")
    except Exception as e:
        print(f"  CNN error: {e}")
    
    # Format 3: Different shapes
    print("\n[3] Testing LSTM with different shapes:")
    for shape in [[1, 16000], [1, 80, 200, 1], [1, 1, 16000, 1]]:
        try:
            test_input = np.random.randn(*shape).astype(np.float32)
            lstm.set_tensor(lstm_in['index'], test_input)
            lstm.invoke()
            out = lstm.get_tensor(lstm_out['index'])
            print(f"  Shape {shape}: {out[0]}")
        except Exception as e:
            print(f"  Shape {shape}: {e}")

test_all_models()