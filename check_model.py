import tensorflow as tf
import numpy as np

model_path = "app/src/main/assets/cnn_model.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("CNN Model:")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# Test with correct shape
correct_shape = input_details[0]['shape']
test_input = np.random.random(correct_shape).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
print(f"  Output sample: {output[0]}")
print(f"  Expected format: [real_prob, fake_prob]")