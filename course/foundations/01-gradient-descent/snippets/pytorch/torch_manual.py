import torch

distance = torch.tensor([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = torch.tensor([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(200):
    loss = ((w * distance + b - minutes) ** 2).mean()
    loss.backward()  # autograd computes the slopes for us

    with torch.no_grad():  # the update itself shouldn't be tracked
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()  # reset for next time
    b.grad.zero_()

print(f"w = {w.item():.3f}, b = {b.item():.3f}")
