import numpy as np

# A chai stall's data: hours of rain that day -> cups of chai sold
rain_hours = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5])
cups = np.array([41.0, 46.0, 47.0, 53.0, 54.0, 60.0, 61.0, 67.0, 70.0, 77.0])

w, b = 0.0, 0.0
lr = 0.1

for step in range(1000):
    error = w * rain_hours + b - cups

    # TODO 1: the gradients (copy the pattern from the delivery example)
    grad_w = np.mean(2 * error * rain_hours)
    grad_b = np.mean(2 * error)

    # TODO 2: step downhill
    w = w - lr * grad_w
    b = b - lr * grad_b

print(f"Every hour of rain sells about {w:.1f} extra cups, on top of {b:.1f} cups on a dry day.")
