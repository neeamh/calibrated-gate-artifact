"""Privacy icon for the main flowchart: one flat 2D panel, same style as make_utility_icon.py.

Blue = training patients, orange = released rows. Most released rows sit among the training
patients at a normal distance, but one is an exact copy of a training record: it is circled, and
a dashed ring marks the "closer than real patients are to each other" radius around it.
Message: could a training patient be found in the release?

Usage:
    python make_privacy_icon.py                   # -> privacy_icon.svg / .pdf, transparent
    python make_privacy_icon.py --points 45 --seed 3 --out my_icon
"""
import argparse

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyBboxPatch  # noqa: E402

BLUE = '#1f6f8b'
ORANGE = '#c96a1b'
INK = '#3a3f45'
RED = '#c0392b'
GRID = '#dde1e5'
L = 3.2


def padlock(ax, center, size, color, open_=True):
    """Padlock drawn from patches (no font needed). An open shackle means the record is exposed."""
    cx, cy = center
    w, h = size, .78 * size
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={.18 * size}',
                                facecolor=color, edgecolor='none', zorder=7))
    # Shackle: an arch on top; when open, it is lifted and swung to the left.
    sw, lift = .62 * w, (.32 * size if open_ else 0.)
    sx = cx - (.36 * w if open_ else 0.)
    ax.add_patch(Arc((sx, cy + h / 2 + lift), sw, 1.1 * size, theta1=0, theta2=180, lw=2.2, color=color, zorder=6))
    ax.plot([sx - sw / 2, sx - sw / 2], [cy + h / 2 + lift, cy + h / 2 - .02], color=color, lw=2.2, zorder=6,
            solid_capstyle='butt')
    if not open_:
        ax.plot([sx + sw / 2] * 2, [cy + h / 2, cy + h / 2 + lift], color=color, lw=2.2, zorder=6)
    else:
        ax.plot([sx + sw / 2] * 2, [cy + h / 2 + lift, cy + h / 2 + lift - .12 * size], color=color, lw=2.2, zorder=6)
    # Keyhole.
    ax.add_patch(Circle((cx, cy + .05 * h), .1 * size, color='white', zorder=8))
    ax.plot([cx, cx], [cy + .05 * h, cy - .25 * h], color='white', lw=1.6, zorder=8, solid_capstyle='round')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', type=int, default=32, help='points per group')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', default='privacy_icon')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cov = np.array([[1.3, .3], [.3, 1.3]])
    train = rng.multivariate_normal([0, 0], cov, size=args.points)
    release = rng.multivariate_normal([0, 0], cov, size=args.points)
    train = train[(np.abs(train) < L - .3).all(axis=1)]
    release = release[(np.abs(release) < L - .3).all(axis=1)]

    radius = .42  # drawing size of the rings, in data units

    # Pick a training record away from the centre and away from the edges, and copy it into the release.
    r = np.linalg.norm(train, axis=1)
    candidates = np.where((r > 1.0) & (r < 2.0))[0]
    leaked = train[candidates[0]].copy()
    # Keep the neighbourhood clear of other points so the copy reads at icon size.
    train = np.vstack([train[np.linalg.norm(train - leaked, axis=1) > 3 * radius], leaked])
    release = np.vstack([release[np.linalg.norm(release - leaked, axis=1) > 3 * radius], leaked])

    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.scatter(train[:, 0], train[:, 1], s=30, color=BLUE, alpha=.85, linewidths=0, zorder=2)
    ax.scatter(release[:, 0], release[:, 1], s=26, marker='D', color=ORANGE, alpha=.85, linewidths=0, zorder=3)

    # The copied record: orange diamond exactly on a blue dot, circled, with the proximity radius.
    # Draw the copied pair so both are visible: a larger blue dot under the orange diamond.
    ax.scatter([leaked[0]], [leaked[1]], s=85, color=BLUE, linewidths=0, zorder=3.5)
    ax.scatter([leaked[0]], [leaked[1]], s=26, marker='D', color=ORANGE, linewidths=0, zorder=3.6)
    ax.add_patch(Circle(leaked, 2.0 * radius, fill=False, ls=(0, (2.5, 2)), lw=1.3, color=RED, zorder=4))
    ax.add_patch(Circle(leaked, radius, fill=False, lw=1.6, color=RED, zorder=5))
    # Open padlock just outside the dashed ring, toward the emptier side of the panel.
    direction = leaked / (np.linalg.norm(leaked) + 1e-9)
    lock_at = leaked + direction * 2.0 * radius * 1.9
    lock_at = np.clip(lock_at, -L + .5, L - .6)
    padlock(ax, lock_at, .55, RED, open_=True)

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
