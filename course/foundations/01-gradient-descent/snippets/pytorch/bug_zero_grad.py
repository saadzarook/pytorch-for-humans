import torch

w = torch.tensor(3.0, requires_grad=True)

for i in range(3):
    loss = (w - 2) ** 2  # slope at w=3 is 2 * (3 - 2) = 2
    loss.backward()  # ...but we never call zero_grad()
    print(f"after backward #{i + 1}: w.grad = {w.grad.item()}")

# The true slope is 2 every time. PyTorch ADDS new gradients to old ones,
# so without zeroing, the "slope" keeps growing: 2, 4, 6...
