import torch

w = torch.tensor(3.0, requires_grad=True)
loss = (w - 2) ** 2
loss.backward()

try:
    w -= 0.1 * w.grad  # forgot `with torch.no_grad():`
except RuntimeError as e:
    print("RuntimeError:", e)

with torch.no_grad():  # the fix
    w -= 0.1 * w.grad
print("With no_grad it works: w =", round(w.item(), 3))
