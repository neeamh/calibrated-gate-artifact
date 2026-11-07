"""Utility icon for the main flowchart: one flat 2D panel.

Patients as particles in two classes (blue = not readmitted, orange = readmitted), with the
decision boundary of a logistic regression fitted to them and each side of the boundary lightly
tinted with the class the model predicts there. Message: a model learns to separate outcomes
from these rows.

Usage:
    python make_utility_icon.py                   # -> utility_icon.svg / .pdf, transparent
    python make_utility_icon.py --points 900 --seed 3 --out my_icon
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BLUE = '#1f6f8b'
ORANGE = '#c96a1b'
INK = '#3a3f45'
GRID = '#dde1e5'
L = 3.2


def sample(n, rng):
    """Two overlapping classes, separated along a diagonal."""
    cov = np.array([[.8, .2], [.2, .8]])
    half = n // 2
    neg = rng.multivariate_normal([-.95, -.75], cov, size=half)
    pos = rng.multivariate_normal([.95, .75], cov, size=n - half)
    X = np.vstack([neg, pos])
    y = np.r_[np.zeros(half), np.ones(n - half)]
    keep = (np.abs(X) < L - .1).all(axis=1)
    return X[keep], y[keep]


def fit_logistic(X, y, steps=25):
    """Newton-Raphson logistic regression with an intercept (numpy only)."""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-A @ w))
        H = A.T @ (A * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(w))
        w += np.linalg.solve(H, A.T @ (y - p))
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=1400)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', default='utility_icon')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, y = sample(args.points, rng)
    w = fit_logistic(X, y)

    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(2.4, 2.4))

    # Light tint of the predicted class on each side of the boundary.
    g = np.linspace(-L, L, 400)
    GX, GY = np.meshgrid(g, g)
    side = (w[0] + w[1] * GX + w[2] * GY) > 0
    ax.contourf(GX, GY, side.astype(float), levels=[-.5, .5, 1.5], colors=[BLUE, ORANGE], alpha=.07, zorder=0)

    for cls, color in ((0, BLUE), (1, ORANGE)):
        pts = X[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=4, alpha=.65, linewidths=0, zorder=2)

    # Decision boundary w0 + w1 x + w2 y = 0.
    xs = np.linspace(-L, L, 200)
    ys = -(w[0] + w[1] * xs) / w[2]
    ok = np.abs(ys) <= L
    ax.plot(xs[ok], ys[ok], color=INK, lw=1.8, zorder=3)

    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect('equal')
    ax.set_xticks(np.arange(-3, 4)); ax.set_yticks(np.arange(-3, 4))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(True, color=GRID, linewidth=.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.savefig(f'{args.out}.svg', bbox_inches='tight', pad_inches=.02, transparent=True)
    fig.savefig(f'{args.out}.pdf', bbox_inches='tight', pad_inches=.02, transparent=True)
    print(f'wrote {args.out}.svg and {args.out}.pdf')


if __name__ == '__main__':
    main()
