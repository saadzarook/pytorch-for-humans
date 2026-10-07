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
# # PyTorch for Humans __NUM__: __TITLE__
#
# This notebook is the "real PyTorch" half of a lesson in **PyTorch for Humans**. The
# [interactive web version]({{COURSE_URL}}__TIER__/__SLUG__/) has sims and in-browser Python.
#
# Charts are interactive even on the read-only page. To run the code, click **Copy & Edit**.
# Must run top to bottom on Kaggle CPU, with no internet and no external files.

# %%
import numpy as np
import torch
import plotly.graph_objects as go

torch.manual_seed(0)
print("torch", torch.__version__)

# %% [markdown]
# ## 1. 🤔 Why should I care?
# TODO: same story and voice as the web lesson, adapted for a notebook.
#
# Lesson code goes in snippets/ and is pulled in with an include cell (below). The
# build replaces it with the file, and checks its output matches the website's.

# %%
# include: snippets/pytorch/example.py
