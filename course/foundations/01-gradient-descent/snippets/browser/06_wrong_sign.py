import numpy as np

distance = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = np.array([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

w, b = 0.0, 0.0
lr = 0.01
for step in range(5):
    error = w * distance + b - minutes
    print(f"step {step}: loss {np.mean(error ** 2):6.1f}")
    w = w + lr * np.mean(2 * error * distance)  # oops: + instead of -
    b = b + lr * np.mean(2 * error)
