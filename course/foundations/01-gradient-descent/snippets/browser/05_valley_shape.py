import numpy as np

distance = np.array([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])

# How fast the slope changes in each direction of the (w, b) bowl.
# For mean squared error on a line, this "curvature" matrix (the Hessian) is:
H = 2 * np.array([[np.mean(distance ** 2), np.mean(distance)],
                  [np.mean(distance), 1.0]])

# Its eigenvalues are the curvature along the bowl's two main directions.
gentle, steep = np.linalg.eigvalsh(H)  # returned smallest first
print(f"Curvature across the valley (steep):  {steep:.2f}")
print(f"Curvature along the valley (gentle):  {gentle:.3f}")
print(f"So it's about {steep / gentle:.0f}x steeper across than along.")

# A step overshoots and grows (explodes) once lr * curvature > 2.
print(f"Biggest learning rate that doesn't explode: 2 / {steep:.2f} = {2 / steep:.3f}")
