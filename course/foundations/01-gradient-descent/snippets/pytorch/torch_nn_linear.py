import torch

torch.manual_seed(0)  # nn.Linear starts from random values; this makes them repeatable

distance = torch.tensor([0.5, 0.8, 1.0, 1.3, 1.5, 1.9, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6])
minutes = torch.tensor([11.7, 11.2, 13.5, 13.0, 16.4, 17.8, 18.4, 21.9, 20.9, 23.2, 25.3, 24.7])

model = torch.nn.Linear(1, 1)  # one input (distance) -> one output (minutes): literally w * x + b
loss_fn = torch.nn.MSELoss()  # mean squared error, built in
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = distance.unsqueeze(1)  # layers expect one row per example: shape (12,) -> (12, 1)
y = minutes.unsqueeze(1)

for step in range(300):
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"nn.Linear learned: w = {model.weight.item():.3f}, b = {model.bias.item():.3f}")
