import numpy as np

distance = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = np.array([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

w, b = 0.0, 0.0
lr = 0.5  # way too big for this data (the nerd note on the valley shows why)

for step in range(1000):
    error = w * distance + b - minutes
    loss = np.mean(error ** 2)
    if step % 25 == 0 or not np.isfinite(loss):
        print(f"step {step:3d}   loss {loss:.3g}")
    if not np.isfinite(loss):
        print("\ninf = the loss got too big for the computer to store.")
        print("Keep going and it turns into nan ('not a number'). Training is dead.")
        print("This is fine. (It is not fine.)")
        break
    w = w - lr * np.mean(2 * error * distance)
    b = b - lr * np.mean(2 * error)
