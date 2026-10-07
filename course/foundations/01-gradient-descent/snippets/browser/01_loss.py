import numpy as np

# Pizza delivery data: how far away you live (km) -> how long it took (minutes)
distance = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = np.array([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

# Our guess: "3 minutes per km, plus 5 minutes to cook the pizza"
w = 3.0  # slope: extra minutes for every km
b = 5.0  # intercept: minutes even if you live next door

predicted = w * distance + b
errors = predicted - minutes
loss = np.mean(errors ** 2)  # mean squared error: square each miss, then average

for d, p, m in zip(distance[:4], predicted[:4], minutes[:4]):
    print(f"{d} km: guessed {p:5.1f} min, really {m:5.1f} min, off by {p - m:+.1f}")
print("...")
print(f"Loss (mean squared error) = {loss:.2f}")
print("Change w and b above and run again. Can you get the loss below 1?")
