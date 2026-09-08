# CS229 — Machine Learning from Scratch

My solutions to the problem sets of [Stanford's CS229](http://cs229.stanford.edu/) (Machine Learning), implemented in **pure NumPy** — no ML framework. The goal: re-derive every algorithm from the math before ever calling a library.

## What's implemented

- **PS1 — Supervised learning:** logistic regression (Newton's method), Gaussian Discriminant Analysis, Poisson regression, locally weighted linear regression
- **PS2 — Classifiers & theory:** logistic regression convergence, perceptron, kernelized SVM, naive Bayes spam filter
- **PS3 — Deep learning & unsupervised:** neural network with manual backpropagation, Gaussian Mixture Models (EM, semi-supervised)
- **PS4 — Advanced:** Independent Component Analysis, reinforcement learning (cartpole)

Each `problem-sets/PSn/src/` folder contains the implementations; `output/` holds generated predictions and decision-boundary plots.

## Materials

Course notes, slides and problem statements are **not** redistributed here — they belong to Stanford and are available on the [official course page](http://cs229.stanford.edu/) and the [lecture videos on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).

## Setup

```bash
conda env create -f problem-sets/environment.yml
```

---

*Part of my "from scratch" learning philosophy: understand the mathematics before using the framework.*
