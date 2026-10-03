#!/usr/bin/env python3
"""Create one streamline figure for each configured Reynolds number."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from cavity_io import (
    latest_result,
    load_config,
    primary_vortex_center,
    read_structured_result,
    select_reynolds,
)


FIGURE_NUMBER = {100: 4, 400: 5, 1000: 6, 5000: 7}


def plot_case(result, reynolds: int, output: Path, dpi: int) -> None:
    speed = np.hypot(result.u, result.v)
    maximum = max(float(np.nanmax(speed)), 1.0e-12)
    line_width = 0.35 + 1.45 * speed / maximum
    vortex_x, vortex_y, _ = primary_vortex_center(result)

    figure, axis = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    stream = axis.streamplot(
        result.x,
        result.y,
        result.u,
        result.v,
        color=speed,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=maximum),
        density=2.3,
        linewidth=line_width,
        arrowsize=0.75,
        minlength=0.03,
    )
    axis.plot(vortex_x, vortex_y, "+", color="#d1495b", markersize=11, markeredgewidth=2)
    axis.text(
        vortex_x + 0.025,
        vortex_y,
        "primary vortex",
        color="#8e2434",
        fontsize=9,
        va="center",
    )
    axis.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="black", linewidth=1.1)
    axis.annotate(
        "moving lid",
        xy=(0.72, 0.972),
        xytext=(0.28, 0.972),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5},
        ha="center",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(f"Lid-driven cavity streamlines, Re = {reynolds}", pad=14)
    colorbar = figure.colorbar(stream.lines, ax=axis, shrink=0.86, pad=0.04)
    colorbar.set_label("Speed, $|\\mathbf{u}|$")
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "benchmark_config.json")
    parser.add_argument("--cases-root", type=Path, default=root / "cases")
    parser.add_argument("--output", type=Path, default=root / "validation" / "figures")
    parser.add_argument("--re", type=int, nargs="+", help="plot only selected Reynolds numbers")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    selected = select_reynolds(config, args.re)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for reynolds in selected:
        result_path = latest_result(args.cases_root.resolve() / f"Re{reynolds}")
        result = read_structured_result(result_path)
        number = FIGURE_NUMBER.get(reynolds)
        if number is None:
            raise SystemExit(f"No requested figure number is assigned to Re={reynolds}")
        output = output_dir / f"figure_{number:02d}_streamlines_Re{reynolds}.png"
        plot_case(result, reynolds, output, args.dpi)
        print(f"Re={reynolds}: wrote {output}")


if __name__ == "__main__":
    main()
