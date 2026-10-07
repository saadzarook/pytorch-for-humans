import numpy as np

distance = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = np.array([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

w, b = 0.0, 0.0  # start with a terrible guess
lr = 0.1  # learning rate = step size
losses = []

for step in range(200):
    error = w * distance + b - minutes  # 1. how wrong are we?
    loss = np.mean(error ** 2)
    losses.append(loss)
    if step % 25 == 0:  # report BEFORE stepping, so loss, w and b all describe the same moment
        print(f"step {step:3d}   loss {loss:8.3f}   w {w:6.3f}   b {b:6.3f}")

    grad_w = np.mean(2 * error * distance)  # 2. which way is uphill?
    grad_b = np.mean(2 * error)

    w = w - lr * grad_w  # 3. take a step downhill
    b = b - lr * grad_b

print(f"\nLearned: delivery takes {b:.2f} min + {w:.2f} min per km")

# Sanity check against NumPy's exact least-squares answer
w_best, b_best = np.polyfit(distance, minutes, 1)
print(f"Exact:   delivery takes {b_best:.2f} min + {w_best:.2f} min per km")
