import math

import torch


def loss_fn(p):
    # A curved valley: its floor follows y = 0.6·sin(x), and the lowest point is (0, 0)
    x, y = p[0], p[1]
    return 0.05 * x**2 + 1.2 * (y - 0.6 * torch.sin(x)) ** 2


start = [-5.2, 2.2]


def race(lr, max_steps=500):
    """Race three optimizers from the same start. Returns each one's path and result."""
    paths, results = {}, {}
    for name in ["SGD", "Momentum", "Adam"]:
        # float64: the same number precision as the browser sim
        p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        if name == "SGD":
            opt = torch.optim.SGD([p], lr=lr)
        elif name == "Momentum":
            opt = torch.optim.SGD([p], lr=lr, momentum=0.9)
        else:
            opt = torch.optim.Adam([p], lr=lr)

        path, result = [list(start)], "never arrived"
        for step in range(1, max_steps + 1):
            opt.zero_grad()
            loss_fn(p).backward()
            opt.step()
            x, y = p.detach().tolist()
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x) > 40 or abs(y) > 40:
                result = f"blew up at step {step}"
                break
            path.append([x, y])
            if loss_fn(p.detach()) < 0.01:
                result = f"reached the minimum in {step} steps"
                break
        paths[name], results[name] = path, result
    return paths, results


paths, results = race(lr=0.1)
for name, result in results.items():
    print(f"lr 0.1   {name:9s} {result}")

_, results_big = race(lr=1.0)
for name, result in results_big.items():
    print(f"lr 1.0   {name:9s} {result}")
