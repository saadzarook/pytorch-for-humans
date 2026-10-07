# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PyTorch for Humans 01: Gradient Descent 🏔️
#
# **Foundations · Lesson 1.** You need basic Python. You do *not* need calculus.
#
# This notebook is the "real PyTorch" half of lesson 1 of **PyTorch for Humans**, a free course
# that explains the *why*. The [interactive web version]({{COURSE_URL}}foundations/gradient-descent/)
# has sims you can poke, and Python you can edit right on the page.
#
# **How to use this notebook:** the charts below are interactive even on the read-only page
# (hover, zoom, press ▶). To *run* the code, click **Copy & Edit** in the top right. It runs top
# to bottom on Kaggle's free CPU, with no files to upload and no internet needed.
#
# **The plan:** see it → build it from scratch (NumPy) → use the real tool (PyTorch).

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display

print("torch", torch.__version__, "| numpy", np.__version__)

# %% [markdown]
# ## 1. 🤔 Why should I care?
#
# You order a pizza. The app says **"arriving in 19 minutes"**. How does it know?
#
# A computer looked at past deliveries and learned a rule. Let's shrink it down to
# **12 deliveries**: how far away the customer was, and how long it took. The simplest rule is
# a straight line:
#
# ```
# delivery time = w × distance + b
# ```
#
# - **`w`** (slope): extra minutes per kilometre.
# - **`b`** (intercept): minutes even if you live next door (cooking, boxing, finding the scooter keys).
#
# `w` and `b` are **parameters**: knobs. Our job is to find good settings for them.
#
# Two knobs? Easy to eyeball. Modern AI models have *billions* of knobs. We need a recipe
# that works for any number of them, with no human fiddling. That recipe is **gradient descent**,
# and some souped-up version of it trains nearly every neural network you've heard of.
#
# ### What does "good" mean? The loss
#
# The **loss** is one number for how wrong the knobs are. Lower is better, like golf.
# We'll use **mean squared error**: for each delivery, take the miss (predicted − actual),
# square it (so misses in both directions count, and big misses hurt a lot more), then average.
#
# ```
# loss = average of (predicted − actual)²
# ```
#
# Here's the data, and the loss for one guess:

# %%
# include: snippets/browser/01_loss.py

# %% [markdown]
# Drag the slider to try different slopes `w` (with `b` fixed at 8). Watch the loss:
# it's lowest when the line hugs the points.

# %%
def loss_at(w, b, x=distance, y=minutes):  # data bound now, so later cells can't change it
    return np.mean((w * x + b - y) ** 2)


ws = np.round(np.arange(0, 10.01, 0.5), 2)
b_fixed = 8.0
xs = np.array([0, 4])

fig = go.Figure()
fig.add_trace(go.Scatter(x=distance, y=minutes, mode="markers", marker=dict(size=9), name="deliveries"))
fig.add_trace(go.Scatter(x=xs, y=ws[0] * xs + b_fixed, mode="lines", line=dict(width=3), name="our line"))
steps = [
    dict(
        method="update",
        label=f"{w:g}",
        args=[
            {"y": [minutes, w * xs + b_fixed]},
            {"title": f"w = {w:g}, b = {b_fixed:g}  →  loss = {loss_at(w, b_fixed):.2f}"},
        ],
    )
    for w in ws
]
fig.update_layout(
    title=f"w = {ws[0]:g}, b = {b_fixed:g}  →  loss = {loss_at(ws[0], b_fixed):.2f}",
    sliders=[dict(active=0, currentvalue=dict(prefix="w = "), steps=steps, pad=dict(t=40))],
    xaxis=dict(title="distance (km)", range=[0, 4]),
    yaxis=dict(title="minutes", range=[0, 45]),
    height=450,
    margin=dict(l=40, r=20, t=50, b=40),
)
fig.show()

# %% [markdown]
# ## 2. 🌫️ The analogy: walking downhill in thick fog
#
# You're lost on a hillside in fog so thick you can't see your shoes. You want to reach the
# village in the valley. You can't see it, but you can feel the ground under your feet. So you:
#
# 1. **Feel** which way the ground tilts down.
# 2. **Take a step** that way.
# 3. **Repeat** until the ground is flat.
#
# | In the fog | In machine learning |
# | --- | --- |
# | Where you're standing | The knob values `w`, `b` |
# | Your altitude | The loss |
# | The tilt under your feet | The **slope** / **gradient** |
# | Your stride length | The **learning rate** |
# | The valley floor | The best knobs (lowest loss) |
#
# ### Feeling the slope: just nudge it
#
# The slope answers: *"If I nudge this knob up a tiny bit, does the loss go up or down, and how fast?"*
# We can measure that directly, then compare with the calculus shortcut:

# %%
# include: snippets/browser/02_slope.py

# %% [markdown]
# They agree: calculus is a shortcut for nudging. (Mathematicians call the slope the
# **derivative**; with several knobs, the list of slopes is the **gradient**.)
#
# ### The update rule
#
# The gradient points uphill, so we step the other way:
#
# ```
# new w = old w − learning_rate × slope
# ```
#
# - Slope positive → subtract → move left. Downhill ✅
# - Slope negative → subtract a negative → move right. Downhill ✅
# - Steep → big step. Flat → tiny step. It slows down by itself near the bottom.
#
# **Learning rate = stride length.** Baby steps take forever. Giant leaps overshoot the valley
# and land you on the opposite hillside, possibly higher than before.

# %% [markdown]
# ## 3. 🎮 See it: the ball rolling downhill
#
# One knob, loss = `(w − 2)²`, starting at `w = −2`. Same start, four learning rates.
# Press ▶ under the animation.

# %%
def descend_1d(df, w0, lr, steps):
    path = [w0]
    for _ in range(steps):
        path.append(path[-1] - lr * df(path[-1]))
    return np.array(path)


bowl = lambda w: (w - 2) ** 2
bowl_slope = lambda w: 2 * (w - 2)
configs = [("🐢 too slow (lr=0.02)", 0.02), ("👌 just right (lr=0.3)", 0.3),
           ("🏓 bouncy (lr=0.9)", 0.9), ("🔥 chaos (lr=1.05)", 1.05)]
N_STEPS = 30
paths = [descend_1d(bowl_slope, -2.0, lr, N_STEPS) for _, lr in configs]

grid = np.linspace(-3, 7, 200)
fig_anim, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True, dpi=72)  # low dpi keeps the notebook small
balls = []
for ax, (title, _) in zip(axes, configs):
    ax.plot(grid, bowl(grid), lw=2)
    ax.set_title(title.split(" ", 1)[1], fontsize=10)
    ax.set_xlim(-3, 7)
    ax.set_ylim(-1, 27)
    ax.set_xlabel("w")
    (ball,) = ax.plot([], [], "o", ms=11, color="tab:red")
    balls.append(ball)
axes[0].set_ylabel("loss")
fig_anim.tight_layout()


def draw_frame(i):
    for ball, path in zip(balls, paths):
        ball.set_data([path[i]], [bowl(path[i])])
    return balls


anim = animation.FuncAnimation(fig_anim, draw_frame, frames=N_STEPS + 1, interval=250, blit=True)
plt.close(fig_anim)  # show only the animation, not an extra static copy
display(HTML(anim.to_jshtml()))

# %% [markdown]
# What you should see:
#
# - **Too slow** creeps along and is nowhere near the bottom after 30 steps.
# - **Just right** gets there in a handful of steps. The steps shrink as the slope flattens.
# - **Bouncy** overshoots every time, but the bounces shrink.
# - **Chaos** overshoots by *more* each time and flies off the chart. 💥
#
# 🤓 **Nerd note:** for this curve, each step multiplies the distance to the bottom by
# `(1 − 2 × lr)`. Below `lr = 1` that shrinks, at exactly `lr = 1` you bounce forever, and above it
# every bounce grows. Real loss curves are messier, so in practice you find a good learning rate
# by experimenting.
#
# Here's the loss after each step for all four (log scale, so you can see both "tiny" and "huge"):

# %%
fig = go.Figure()
for (title, _), path in zip(configs, paths):
    fig.add_trace(go.Scatter(y=bowl(path), mode="lines+markers", name=title))
fig.update_layout(title="Loss after each step", xaxis_title="step", yaxis_title="loss (log scale)",
                  yaxis_type="log", height=400, margin=dict(l=40, r=20, t=50, b=40))
fig.show()

# %% [markdown]
# ### Plot twist: bumpy hills (local minima)
#
# Real loss landscapes can have more than one valley. Gradient descent only feels the ground
# under its feet, so it settles in whichever valley it rolls into first.

# %%
bumpy = lambda w: 0.06 * w**4 - 0.75 * w**2 + 0.45 * w + 4
bumpy_slope = lambda w: 0.24 * w**3 - 1.5 * w + 0.45

grid = np.linspace(-4.4, 4.4, 300)
fig = go.Figure(go.Scatter(x=grid, y=bumpy(grid), mode="lines", name="loss curve", line=dict(width=3)))
for label, w0, lr in [("start 3.5, lr 0.05 → stuck", 3.5, 0.05), ("start −0.5, lr 0.05 → deep valley", -0.5, 0.05),
                      ("start 4.0, lr 0.4 → leaps the hill", 4.0, 0.4)]:
    path = descend_1d(bumpy_slope, w0, lr, 80)
    fig.add_trace(go.Scatter(x=path, y=bumpy(path), mode="lines+markers", name=label, marker=dict(size=6)))
    print(f"{label:40s} ended at w = {path[-1]:+.3f}, loss = {bumpy(path[-1]):.3f}")
fig.update_layout(title="Same algorithm, different starting points", xaxis_title="w", yaxis_title="loss",
                  height=420, margin=dict(l=40, r=20, t=50, b=40))
fig.show()

# %% [markdown]
# - **Local minimum:** lowest point *in the neighbourhood*: the right-hand valley, where the first path got stuck.
# - **Global minimum:** lowest point *anywhere*: the deeper left-hand valley the other two paths found (compare their final losses above).
#
# Fixes: try different starting points, or (sometimes) bigger steps. For big neural networks this
# turns out to be much less scary than the 1D picture suggests, and researchers are still working
# out exactly why. Learn the idea; don't lose sleep over it yet.

# %% [markdown]
# ### Two knobs: the bowl from above
#
# With `w` *and* `b`, the loss is a 3D bowl. From above it looks like a hiking map: each contour
# line is one "height" of loss. Press ▶ to watch gradient descent (lr = 0.1, starting from
# `w = 0, b = 0`) walk into the valley while the line fits the data.

# %%
def descent_2d(lr, steps, w0=0.0, b0=0.0, x=distance, y=minutes):  # data bound now, see loss_at
    w, b = w0, b0
    path = [(w, b)]
    for _ in range(steps):
        error = w * x + b - y
        w, b = w - lr * np.mean(2 * error * x), b - lr * np.mean(2 * error)
        path.append((w, b))
    return np.array(path)


path2d = descent_2d(lr=0.1, steps=150)
W, B = np.meshgrid(np.linspace(-1, 11, 120), np.linspace(-4, 20, 120))
L = np.mean((W[..., None] * distance + B[..., None] - minutes) ** 2, axis=-1)

frame_ids = list(range(0, 30)) + list(range(30, 151, 5))
fig = make_subplots(rows=1, cols=2, subplot_titles=("The data and the current line", "Loss landscape from above"))
fig.add_trace(go.Scatter(x=distance, y=minutes, mode="markers", marker=dict(size=8), name="deliveries"), row=1, col=1)
fig.add_trace(go.Scatter(x=xs, y=path2d[0, 0] * xs + path2d[0, 1], mode="lines", line=dict(width=3), name="line"), row=1, col=1)
fig.add_trace(go.Contour(x=W[0], y=B[:, 0], z=np.log10(L), colorscale="Blues", showscale=False,
                         contours=dict(coloring="heatmap"), name="log10(loss)"), row=1, col=2)
fig.add_trace(go.Scatter(x=path2d[:1, 0], y=path2d[:1, 1], mode="lines+markers", line=dict(color="orangered"),
                         marker=dict(size=5), name="path"), row=1, col=2)
fig.frames = [
    go.Frame(
        data=[
            go.Scatter(x=xs, y=path2d[i, 0] * xs + path2d[i, 1]),
            go.Scatter(x=path2d[: i + 1, 0], y=path2d[: i + 1, 1]),
        ],
        traces=[1, 3],
        name=str(i),
        layout=go.Layout(title_text=f"step {i}: w = {path2d[i, 0]:.2f}, b = {path2d[i, 1]:.2f}, loss = {loss_at(*path2d[i]):.2f}"),
    )
    for i in frame_ids
]
fig.update_layout(
    title_text=f"step 0: w = 0.00, b = 0.00, loss = {loss_at(0, 0):.2f}",
    height=460,
    margin=dict(l=40, r=20, t=80, b=40),
    showlegend=False,
    updatemenus=[dict(type="buttons", x=0, y=-0.12, xanchor="left", buttons=[
        dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
        dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
    ])],
)
fig.update_xaxes(title_text="distance (km)", range=[0, 4], row=1, col=1)
fig.update_yaxes(title_text="minutes", range=[0, 32], row=1, col=1)
fig.update_xaxes(title_text="slope w", range=[-1, 11], row=1, col=2)
fig.update_yaxes(title_text="intercept b", range=[-4, 20], row=1, col=2)
fig.show()

# %% [markdown]
# Notice the path **zig-zags** at first. The valley is long and narrow: steep across, gentle along.
# The steep direction limits how big a step you can take, so progress along the gentle direction
# is slow. Later lessons fix this with **feature scaling** and smarter optimizers (**momentum**, **Adam**).
#
# 🤓 **Nerd note: exactly how lopsided is the valley?** The **Hessian** is a small table of how fast
# the slope itself changes in each direction (the bowl's *curvature*). Its two **eigenvalues** are the
# curvature straight across the valley and straight along it:

# %%
# include: snippets/browser/05_valley_shape.py

# %% [markdown]
# About **{{facts.valley.ratio}}× steeper across than along**. Once `lr × curvature` is bigger than 2,
# each step overshoots by more than it started with, so the biggest learning rate that doesn't
# explode is **2 / {{facts.valley.steep}} ≈ {{facts.valley.maxLr}}**. On the website, the "zig-zag" preset
# (lr = {{facts.presets.zigzag}}) survives, and "chaos" (lr = {{facts.presets.chaos}}) explodes.

# %% [markdown]
# ## 4. 🛠️ Build it from scratch (NumPy)
#
# Every loop does three things: **(1)** how wrong are we? **(2)** which way is uphill?
# **(3)** step downhill. We print `loss`, `w` and `b` *before* stepping, so all three describe
# the same moment.

# %%
# include: snippets/browser/03_descent.py

# %%
fig = go.Figure(go.Scatter(y=losses, mode="lines", line=dict(width=3)))
fig.update_layout(title="Loss after each step (from scratch)", xaxis_title="step", yaxis_title="loss (log scale)",
                  yaxis_type="log", height=360, margin=dict(l=40, r=20, t=50, b=40))
fig.show()

# %% [markdown]
# 🎉 Our ten lines of NumPy found the same line as the exact formula: about
# **{{facts.bestFit.bShort}} minutes + {{facts.bestFit.wShort}} minutes per km**.
#
# **When do we stop?** When the loss stops improving: once the curve goes flat, extra steps just
# burn time. Here, 200 steps is plenty. (Later you'll make this automatic, with *epochs* and
# *early stopping*.)
#
# ### 🎛️ Bonus (after Copy & Edit): a live learning-rate slider
#
# This one needs a running notebook. On the read-only Kaggle page it shows nothing interactive.
# After **Copy & Edit**, run it and drag the slider.

# %%
try:
    import ipywidgets as widgets

    def show_run(lr=0.1, steps=50):
        path = descent_2d(lr=lr, steps=steps)
        run_losses = [loss_at(pw, pb) for pw, pb in path]
        plt.figure(figsize=(6, 3))
        plt.plot(run_losses)
        plt.yscale("log")
        plt.xlabel("step")
        plt.ylabel("loss (log scale)")
        plt.title(f"lr = {lr:.3g}: final loss {run_losses[-1]:.3g}")
        plt.show()

    widgets.interact(
        show_run,
        lr=widgets.FloatLogSlider(value=0.1, base=10, min=-3, max=-0.5, step=0.05),
        steps=widgets.IntSlider(value=50, min=5, max=300, step=5),
    )
except ImportError:
    print("ipywidgets isn't installed here, so skip this bonus. Everything else works without it.")

# %% [markdown]
# ## 5. 🔥 The PyTorch way
#
# For a straight line we could work out the slope formulas by hand. For a neural network with
# millions of knobs, nobody does that. PyTorch's **autograd** computes every slope automatically.
#
# | PyTorch | What it means | Fog version |
# | --- | --- | --- |
# | `requires_grad=True` | Track how the loss depends on this knob | Strap a tilt sensor to your boot |
# | `loss.backward()` | Compute every tracked knob's slope, store it in `.grad` | Feel the ground |
# | `optimizer.step()` | `knob = knob − lr × knob.grad` for every knob | Take the step |
# | `optimizer.zero_grad()` | Wipe old slopes before computing new ones | Forget the last spot's tilt |
#
# First: proof that autograd gets the same slopes as our formula.

# %%
# include: snippets/pytorch/torch_grad_check.py

# %% [markdown]
# Same numbers. Now the training loop the way you'll write it for the rest of your PyTorch life:

# %%
# include: snippets/pytorch/torch_way.py

# %% [markdown]
# Same answer as our NumPy version: it's the exact same algorithm. PyTorch did the calculus
# and the bookkeeping.
#
# **What is `optimizer.step()` doing?** Nothing magic. This is equivalent:

# %%
# include: snippets/pytorch/torch_manual.py

# %% [markdown]
# 💡 **Why "SGD"?** It stands for *stochastic* gradient descent: normally it's fed a small random
# batch of data each step. We give it all 12 points every time, so here it's plain gradient descent.
#
# ### 👀 Sneak peek: the same model, the "neural network" way
#
# Real PyTorch code wraps knobs in **layers**. `nn.Linear(1, 1)` is literally `w × x + b`,
# starting from small random values instead of zero. You'll meet this properly in a later lesson.

# %%
# include: snippets/pytorch/torch_nn_linear.py

# %% [markdown]
# ## 6. 🔧 "Bro, why is it broken?"
#
# ### 💥 Loss goes to `inf`, then `nan`
#
# **Cause:** learning rate too big. Every step overshoots more than the last.

# %%
# include: snippets/pytorch/bug_nan_torch.py

# %% [markdown]
# 🐶☕🔥 *"This is fine."* That's the comic where a dog calmly sips coffee while the room burns.
# Very much your training loop, cheerfully printing `nan`.
#
# **Fix:** divide the learning rate by 10 and try again. Once it's stable, nudge it back up.
#
# ### 🧺 Forgot `optimizer.zero_grad()`
#
# PyTorch **adds** new gradients to whatever is already in `.grad`. No error, just silently wrong slopes:

# %%
# include: snippets/pytorch/bug_zero_grad.py

# %% [markdown]
# Like a shopping cart you never empty. (Adding is useful for some advanced tricks, which is why
# PyTorch does it.) **Rule: `zero_grad()` every step.**
#
# ### 🚫 `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.`
#
# You updated a tracked knob without `torch.no_grad()`:

# %%
# include: snippets/pytorch/bug_inplace.py

# %% [markdown]
# ### 📈 Loss goes *up*, steadily
#
# You wrote `+` instead of `−` and built gradient **ascent**. Congrats, you're maximising wrongness:

# %%
# include: snippets/browser/06_wrong_sign.py

# %% [markdown]
# **Fix:** it's minus. Always minus.
#
# ### 🐢 Loss barely moves · 🏓 loss bounces
#
# Learning rate too small (raise it 3–10×), or a bit too big (lower it a little). The loss curve
# tells you which: you want a fast drop that then flattens out.

# %% [markdown]
# ## 7. 🏆 Mini challenge
#
# A chai stall tracked **cups sold** vs **hours of rain**. Fill in the two TODOs (gradients,
# then the downhill step), then run the check cell after it.

# %%
# include: snippets/browser/challenge_starter.py

# %%
# include: snippets/browser/challenge_check.py

# %% [markdown]
# <details>
# <summary><b>Spoiler: one possible solution</b> (try first!)</summary>
#
# ```python
# grad_w = np.mean(2 * error * rain_hours)
# grad_b = np.mean(2 * error)
# w = w - lr * grad_w
# b = b - lr * grad_b
# ```
# </details>
#
# The cells below run that solution through the same checker, so this notebook proves it works:

# %%
# include: snippets/browser/challenge_solution.py

# %%
# include: snippets/browser/challenge_check.py

# %%
assert challenge_passed, "the reference solution should pass the checker"

# %% [markdown]
# ## 🧠 Quick quiz
#
# Answer in your head, then click to reveal.
#
# **1. At your current `w` the slope is negative. What does gradient descent do to `w`?**
# <details><summary>Answer</summary>It <b>increases</b> <code>w</code>. A negative slope means the loss falls as <code>w</code> grows, and <code>w − lr × (negative)</code> is <code>w + something</code>.</details>
#
# **2. The learning rate is fixed, but the steps shrink near the bottom. Why?**
# <details><summary>Answer</summary>The slope gets flatter, and step = learning rate × slope. Near the bottom the slope is ≈ 0, so the step is ≈ 0.</details>
#
# **3. Your loss goes from hundreds, to billions, to `inf`, to `nan`. Most likely cause?**
# <details><summary>Answer</summary>Learning rate too big. (Too small makes the loss fall slowly; it never explodes.)</details>
#
# **4. Which line actually changes `w` and `b`: `loss.backward()`, `optimizer.step()`, or `optimizer.zero_grad()`?**
# <details><summary>Answer</summary><code>optimizer.step()</code>. <code>backward()</code> only computes slopes into <code>.grad</code>; <code>zero_grad()</code> resets those slopes.</details>
#
# **5. The ball settles in a small dip, but a deeper valley exists elsewhere. Where did it stop?**
# <details><summary>Answer</summary>At a <b>local minimum</b>. Try other starting points.</details>

# %% [markdown]
# ## 😂 Meme recap
#
# **Expanding brain** (each level supposedly more enlightened than the last):
#
# 1. 🧠 Guess `w` and `b` by hand
# 2. 🧠✨ Try random values, keep the best
# 3. 🧠💡 Feel the slope, step downhill, repeat
# 4. 🌌🧠🌌 `loss.backward(); optimizer.step()` on a billion knobs
#
# **Nope / yep** (the two-panel meme where you reject one thing and point approvingly at another):
#
# | 🙅 nope | 👉😎 yep |
# | --- | --- |
# | `lr = 1.05` and vibes | `lr = 0.3` and watching the loss chart |
#
# ## 📌 TL;DR
#
# - **Loss** = one number for how wrong the model is. Lower is better.
# - **Slope / gradient** = which way is uphill, and how steep.
# - **Update rule:** `w = w − learning_rate × slope`. The minus means "go downhill".
# - **Learning rate** = stride. Too small → slow. Too big → bouncy, then `nan`.
# - **Stop** when the loss stops improving.
# - **Local minima** exist. Try different starting points.
# - **PyTorch loop:** `zero_grad()` → loss → `backward()` → `step()`. Every. Single. Time.
#
# ## 📚 Further reading
#
# - PyTorch: [A Gentle Introduction to `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html)
# - d2l.ai: [Linear regression from scratch](https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html), [Gradient descent](https://d2l.ai/chapter_optimization/gd.html)
# - 3Blue1Brown: [Gradient descent, how neural networks learn](https://www.3blue1brown.com/lessons/gradient-descent)
# - Andrej Karpathy: [Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) and [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
# - Distill: [Why Momentum Really Works](https://distill.pub/2017/momentum/) (the zig-zag problem, beautifully explained)
