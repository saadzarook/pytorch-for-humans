import numpy as np

distance = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = np.array([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])


def loss_fn(w, b):
    return np.mean((w * distance + b - minutes) ** 2)


w, b = 3.0, 5.0
nudge = 0.001

# 1) Feel the slope with your feet: nudge w a tiny bit each way, see how the loss changes.
slope_w = (loss_fn(w + nudge, b) - loss_fn(w - nudge, b)) / (2 * nudge)
slope_b = (loss_fn(w, b + nudge) - loss_fn(w, b - nudge)) / (2 * nudge)
print(f"Nudging:  slope for w = {slope_w:.3f},  slope for b = {slope_b:.3f}")

# 2) The calculus shortcut gives the same numbers, without any nudging.
error = w * distance + b - minutes
grad_w = np.mean(2 * error * distance)
grad_b = np.mean(2 * error)
print(f"Formula:  slope for w = {grad_w:.3f},  slope for b = {grad_b:.3f}")

# Negative slope = "loss goes DOWN if you increase this", so increase it.
for name, s in [("w", grad_w), ("b", grad_b)]:
    direction = "increase" if s < 0 else "decrease"
    print(f"Downhill for {name}: {direction} it")
