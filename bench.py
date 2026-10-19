import time
import numpy as np

def benchmark():
    rng = np.random.default_rng(42)
    # 5000 векторов размерности 384
    a = rng.normal(size=(5000, 384)).astype("float32")
    q = rng.normal(size=384).astype("float32")
    
    # Нормализуем для косинусного сходства
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    q /= np.linalg.norm(q)
    
    samples = []
    for _ in range(100):
        start = time.perf_counter()
        # Считаем скалярное произведение (матрица на вектор)
        scores = a @ q
        # Находим топ-10
        np.argpartition(scores, -10)[-10:]
        samples.append((time.perf_counter() - start) * 1000)
        
    p50, p95 = np.percentile(samples, [50, 95])
    print(f"Exact NumPy (n=5000, d=384): p50 = {p50:.3f} ms, p95 = {p95:.3f} ms")

if __name__ == "__main__":
    benchmark()
