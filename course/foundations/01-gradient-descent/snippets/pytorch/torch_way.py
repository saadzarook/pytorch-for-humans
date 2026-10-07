import torch

# Same pizza data as before, now as PyTorch tensors
distance = torch.tensor([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = torch.tensor([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

# requires_grad=True: "PyTorch, please track how the loss depends on these"
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# The optimizer does the "w = w - lr * grad" step for us
optimizer = torch.optim.SGD([w, b], lr=0.1)

for step in range(200):
    predicted = w * distance + b
    loss = ((predicted - minutes) ** 2).mean()
    if step % 40 == 0:  # report BEFORE stepping, so loss, w and b all describe the same moment
        print(f"step {step:3d}   loss {loss.item():8.3f}   w {w.item():6.3f}   b {b.item():6.3f}")

    optimizer.zero_grad()  # 1. wipe the slopes from last step
    loss.backward()  # 2. compute fresh slopes: fills w.grad and b.grad
    optimizer.step()  # 3. step downhill: w -= lr * w.grad, b -= lr * b.grad

print(f"\nLearned: delivery takes {b.item():.2f} min + {w.item():.2f} min per km")
