import torch

distance = torch.tensor([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = torch.tensor([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)

loss = ((w * distance + b - minutes) ** 2).mean()
loss.backward()

# Our hand-written NumPy formula from earlier, for comparison:
error = (w * distance + b - minutes).detach()
print(f"autograd:  w.grad = {w.grad.item():.3f}   b.grad = {b.grad.item():.3f}")
print(f"by hand:   grad_w = {(2 * error * distance).mean().item():.3f}   grad_b = {(2 * error).mean().item():.3f}")
