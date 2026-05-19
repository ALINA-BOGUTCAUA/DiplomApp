import numpy as np

def analyzeFakeVoice(audioData):
    audio = np.array(audioData)
    
    sum_val = np.sum(audio)
    sum_squares = np.sum(audio ** 2)
    zero_count = np.sum(np.abs(audio) < 0.001)
    max_val = np.max(audio)
    min_val = np.min(audio)
    
    mean = sum_val / len(audio)
    variance = (sum_squares / len(audio)) - (mean ** 2)
    std_dev = np.sqrt(max(variance, 0))
    zero_ratio = zero_count / len(audio)
    dynamic_range = max_val - min_val
    rms = np.sqrt(variance)
    
    # Score based on features
    # Fake voices tend to have: low zero_ratio, specific std_dev, limited dynamic range
    
    score = 0.0
    
    # Check for suspicious patterns
    if std_dev < 0.15:
        score += 0.3  # Too constant
    elif std_dev > 0.35:
        score += 0.1  # Very noisy
    
    if dynamic_range < 0.3:
        score += 0.2  # Limited range
    
    if zero_ratio > 0.1:
        score += 0.1  # Too many silences
    
    if abs(mean) < 0.005:
        score += 0.1  # Centered at zero
    
    # RMS check - fake voices often have specific RMS
    if 0.1 < rms < 0.25:
        score += 0.2
    
    # Final confidence
    confidence = 0.9 - (score * 0.8)
    confidence = max(0.1, min(0.95, confidence))
    
    isFake = confidence < 0.5
    
    return {
        'isFake': isFake,
        'confidence': confidence,
        'scores': {
            'zeroRatio': zero_ratio,
            'stdDev': std_dev,
            'dynamicRange': dynamic_range,
            'mean': mean,
            'rms': rms
        }
    }

print("=" * 60)
print("Тестирование НАСТРОЕННОГО алгоритма")
print("=" * 60)

# Test cases
tests = [
    ("Синтетический (равномерный)", np.random.uniform(-0.3, 0.3, 16000).astype(np.float32)),
    ("Натуральный", np.random.randn(16000).astype(np.float32) * 0.3),
    ("ИИ-подобный (синус)", np.sin(np.linspace(0, 20 * np.pi, 16000)).astype(np.float32) * 0.5),
    ("Человек с паузами", np.array([0 if i%200<50 else np.random.randn()*0.2 for i in range(16000)], dtype=np.float32)),
    ("Обрезанный", np.clip(np.random.randn(16000) * 0.5, -0.8, 0.8).astype(np.float32)),
]

results = []
for name, audio in tests:
    result = analyzeFakeVoice(audio)
    results.append(result)
    print(f"\n[{name}]:")
    print(f"  isFake: {result['isFake']}, conf: {result['confidence']:.3f}")

print("\n" + "=" * 60)
fake = sum(1 for r in results if r['isFake'])
real = len(results) - fake
print(f"FAKE: {fake}, REAL: {real}")
print("=" * 60)