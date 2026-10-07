import torch

distance = torch.tensor([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = torch.tensor([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.5)  # too big!

for step in range(200):
    loss = ((w * distance + b - minutes) ** 2).mean()
    if step % 10 == 0 or torch.isnan(loss):
        print(f"step {step:3d}   loss {loss.item():.3g}")
    if torch.isnan(loss):
        print("\nnan: training is dead. Lower the learning rate and start again.")
        break
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
