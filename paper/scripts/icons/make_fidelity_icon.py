"""Fidelity icon for the main flowchart: a joint distribution drawn as particles on the floor of a
3D box, with each variable's marginal distribution standing on a back wall.

Two panels by default: real (correlated variables) and candidate (same marginals, correlation
destroyed). This is the shuffled-predictor failure from the paper's introduction: every column
looks right, only the joint structure is wrong.

Usage:
    python make_fidelity_icon.py                  # teal (real) panel, no label -> fidelity_icon.svg / .pdf
    python make_fidelity_icon.py --panels both --titles   # real and candidate side by side
    python make_fidelity_icon.py --points 2500 --seed 7 --out my_icon
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

TEAL = '#1f6f8b'       # "real" everywhere in the paper
ORANGE = '#c96a1b'     # "candidate", used only for this distinction
INK = '#3a3f45'        # marginal curves: identical in both panels, so they stay neutral
BAR = '#9aa1a9'        # marginal histogram bars
GRID = '#dde1e5'
L = 3.2                # half-width of the box, in standard deviations


def sample(n, rho, rng):
    cov = np.array([[1., rho], [rho, 1.]])
    return rng.multivariate_normal([0., 0.], cov, size=n)


def density(pts, rho):
    d = 1 - rho ** 2
    x, y = pts[:, 0], pts[:, 1]
    return np.exp(-(x ** 2 - 2 * rho * x * y + y ** 2) / (2 * d)) / (2 * np.pi * np.sqrt(d))


def panel(ax, pts, rho, color, title, wall_height):
    pts = pts[(np.abs(pts) < L).all(axis=1)]
    # Particles on the floor, coloured from pale to full by local density, densest drawn last.
    dens = density(pts, rho)
    order = np.argsort(dens)
    cmap = LinearSegmentedColormap.from_list('c', [tuple(.25 + .75 * c for c in to_rgb(color)), color])
    ax.scatter(pts[order, 0], pts[order, 1], np.zeros(len(order)), c=dens[order] / dens.max(),
               cmap=cmap, s=1.2, alpha=.7, linewidths=0, depthshade=False, zorder=2)

    # Marginals on the back walls: histogram bars plus the N(0, 1) curve.
    grid = np.linspace(-L, L, 300)
    curve = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    scale = wall_height / curve.max()
    edges = np.linspace(-L, L, 25)
    centers = (edges[:-1] + edges[1:]) / 2
    for axis, wall in ((0, 'back'), (1, 'left')):
        counts, _ = np.histogram(pts[:, axis], bins=edges, density=True)
        for c, h in zip(centers, counts * scale):
            xs, ys = (c, L) if wall == 'back' else (-L, c)
            ax.plot([xs, xs], [ys, ys], [0, h], color=BAR, lw=1.1, solid_capstyle='butt', zorder=3)
        if wall == 'back':
            ax.plot(grid, np.full_like(grid, L), curve * scale, color=INK, lw=1.4, zorder=4)
        else:
            ax.plot(np.full_like(grid, -L), grid, curve * scale, color=INK, lw=1.4, zorder=4)

    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0, wall_height * 1.15)
    ax.set_box_aspect((1, 1, .55))
    ax.view_init(elev=24, azim=-58)
    ticks = np.arange(-3, 4)
    ax.set_xticks(ticks); ax.set_yticks(ticks); ax.set_zticks(np.linspace(0, wall_height, 4))
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_fill(False)
        a.pane.set_edgecolor(GRID)
        a.line.set_color(GRID)
        a._axinfo['grid'].update(color=GRID, linewidth=.5)
        a._axinfo['tick'].update(inward_factor=0., outward_factor=0.)
        a.set_tick_params(color=(0, 0, 0, 0))
    ax.grid(True)
    if title:
        ax.set_title(title, fontsize=9, y=.93, color=INK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panels', choices=('both', 'real', 'candidate'), default='real')
    parser.add_argument('--rho', type=float, default=-.8, help='correlation of the real panel')
    parser.add_argument('--points', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', default='fidelity_icon')
    parser.add_argument('--titles', action='store_true', help='label each panel')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    real = sample(args.points, args.rho, rng)
    # Candidate: shuffle one column of the real sample. Marginals are exactly the same; correlation is gone.
    candidate = real.copy()
    candidate[:, 1] = rng.permutation(candidate[:, 1])
    specs = {'real': (real, args.rho, TEAL), 'candidate': (candidate, 0., ORANGE)}
    names = ['real', 'candidate'] if args.panels == 'both' else [args.panels]

    plt.rcParams.update({'font.family': 'serif', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(2.6 * len(names), 2.6))
    for i, name in enumerate(names):
        ax = fig.add_subplot(1, len(names), i + 1, projection='3d', computed_zorder=False)
        pts, rho, color = specs[name]
        panel(ax, pts, rho, color, name if args.titles else None, wall_height=.6)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=-.08)
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(f'{args.out}.svg', bbox_inches='tight', pad_inches=.02, transparent=True)
    fig.savefig(f'{args.out}.pdf', bbox_inches='tight', pad_inches=.02, transparent=True)
    print(f'wrote {args.out}.svg and {args.out}.pdf')


if __name__ == '__main__':
    main()
