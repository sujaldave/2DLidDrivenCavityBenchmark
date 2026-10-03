#!/usr/bin/env python3
"""Compare centerlines and primary-vortex locations with Ghia et al."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cavity_io import (
    centerline_profiles,
    error_metrics,
    latest_result,
    load_config,
    primary_vortex_center,
    read_primary_vortex_references,
    read_reference_profile,
    read_structured_result,
    select_reynolds,
    write_csv,
)


COLORS = {100: "#1f77b4", 400: "#ff7f0e", 1000: "#2ca02c", 5000: "#d62728"}


def plot_centerlines(records: list[dict], figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    handles = []
    for record in records:
        reynolds = record["reynolds"]
        color = COLORS.get(reynolds)
        line = axis.plot(record["u_full"], record["y_full"], color=color, linewidth=1.8)[0]
        axis.plot(
            record["u_reference"],
            record["y_reference"],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
        )
        handles.append(Line2D([], [], color=color, linewidth=2, label=f"Re = {reynolds}"))
    handles.append(
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", color="black", label="Ghia et al.")
    )
    axis.set_xlabel("Horizontal velocity, u")
    axis.set_ylabel("Vertical coordinate, y")
    axis.set_title("Vertical centerline velocity: $u(0.5,y)$")
    axis.grid(True, alpha=0.25)
    axis.legend(handles=handles, frameon=False, ncol=2)
    figure.savefig(figures_dir / "figure_01_u_vertical_centerline.png", dpi=240, facecolor="white")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    handles = []
    for record in records:
        reynolds = record["reynolds"]
        color = COLORS.get(reynolds)
        axis.plot(record["x_full"], record["v_full"], color=color, linewidth=1.8)
        axis.plot(
            record["x_reference"],
            record["v_reference"],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.2,
        )
        handles.append(Line2D([], [], color=color, linewidth=2, label=f"Re = {reynolds}"))
    handles.append(
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", color="black", label="Ghia et al.")
    )
    axis.set_xlabel("Horizontal coordinate, x")
    axis.set_ylabel("Vertical velocity, v")
    axis.set_title("Horizontal centerline velocity: $v(x,0.5)$")
    axis.grid(True, alpha=0.25)
    axis.legend(handles=handles, frameon=False, ncol=2)
    figure.savefig(figures_dir / "figure_02_v_horizontal_centerline.png", dpi=240, facecolor="white")
    plt.close(figure)


def plot_primary_vortex(records: list[dict], figures_dir: Path) -> None:
    reynolds = np.asarray([record["reynolds"] for record in records])
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
    for axis, coordinate, label in ((axes[0], "x", "$x_c$"), (axes[1], "y", "$y_c$")):
        axis.plot(
            reynolds,
            [record[f"vortex_reference_{coordinate}"] for record in records],
            "o--",
            color="black",
            label="Ghia et al.",
        )
        axis.plot(
            reynolds,
            [record[f"vortex_simulated_{coordinate}"] for record in records],
            "s-",
            color="#d1495b",
            label="svMultiPhysics",
        )
        axis.set_xscale("log")
        axis.set_xticks(reynolds, labels=[str(value) for value in reynolds])
        axis.set_xlabel("Reynolds number")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    axes[0].set_title("Primary-vortex horizontal location")
    axes[1].set_title("Primary-vortex vertical location")
    axes[0].legend(frameon=False)
    figure.suptitle("Effect of Reynolds number on the primary-vortex center")
    figure.savefig(figures_dir / "figure_03_primary_vortex_center.png", dpi=240, facecolor="white")
    plt.close(figure)


def relative_result_path(path: Path, cases_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(cases_root.resolve()))
    except ValueError:
        return path.name


def validate(args: argparse.Namespace) -> bool:
    config = load_config(args.config.resolve())
    selected = select_reynolds(config, args.re)
    limits = config["validation"]["relative_l2_limits"]
    vortex_limit = float(config["validation"]["primary_vortex_distance_limit"])
    literature = args.literature.resolve()
    cases_root = args.cases_root.resolve()
    output_root = args.output.resolve()
    figures_dir = output_root / "figures"
    data_dir = output_root / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    vortex_references = read_primary_vortex_references(literature / "ghia_vortex_centers.csv")
    plot_records: list[dict] = []
    metric_rows: list[dict] = []
    all_passed = True

    for reynolds in selected:
        result_path = latest_result(cases_root / f"Re{reynolds}")
        print(f"Re={reynolds}: reading {result_path}")
        result = read_structured_result(result_path)
        y_full, u_full, x_full, v_full = centerline_profiles(result)
        y_reference, u_reference = read_reference_profile(
            literature / "ghia_table_I_u.csv", "y", reynolds
        )
        x_reference, v_reference = read_reference_profile(
            literature / "ghia_table_II_v.csv", "x", reynolds
        )
        u_simulated = np.interp(y_reference, y_full, u_full)
        v_simulated = np.interp(x_reference, x_full, v_full)
        u_metrics = error_metrics(u_reference, u_simulated)
        v_metrics = error_metrics(v_reference, v_simulated)
        vortex_x, vortex_y, vortex_psi = primary_vortex_center(result)
        reference_x, reference_y = vortex_references[reynolds]
        vortex_distance = math.hypot(vortex_x - reference_x, vortex_y - reference_y)

        u_limit = float(limits[str(reynolds)]["u"])
        v_limit = float(limits[str(reynolds)]["v"])
        passed = (
            u_metrics["relative_l2"] <= u_limit
            and v_metrics["relative_l2"] <= v_limit
            and vortex_distance <= vortex_limit
        )
        all_passed = all_passed and passed
        metric_rows.append(
            {
                "reynolds": reynolds,
                "result_file": relative_result_path(result_path, cases_root),
                "result_step": result.step,
                "u_absolute_l2": u_metrics["absolute_l2"],
                "u_relative_l2": u_metrics["relative_l2"],
                "u_rmse": u_metrics["rmse"],
                "u_max_abs_error": u_metrics["max_abs_error"],
                "u_relative_l2_limit": u_limit,
                "v_absolute_l2": v_metrics["absolute_l2"],
                "v_relative_l2": v_metrics["relative_l2"],
                "v_rmse": v_metrics["rmse"],
                "v_max_abs_error": v_metrics["max_abs_error"],
                "v_relative_l2_limit": v_limit,
                "primary_vortex_x_ghia": reference_x,
                "primary_vortex_y_ghia": reference_y,
                "primary_vortex_x_svmp": vortex_x,
                "primary_vortex_y_svmp": vortex_y,
                "primary_vortex_distance": vortex_distance,
                "primary_vortex_distance_limit": vortex_limit,
                "status": "PASS" if passed else "FAIL",
            }
        )

        rows = []
        for coordinate, reference, computed in zip(y_reference, u_reference, u_simulated):
            rows.append(
                {
                    "component": "u",
                    "coordinate": float(coordinate),
                    "literature": float(reference),
                    "svMultiPhysics": float(computed),
                    "absolute_error": abs(float(computed - reference)),
                }
            )
        for coordinate, reference, computed in zip(x_reference, v_reference, v_simulated):
            rows.append(
                {
                    "component": "v",
                    "coordinate": float(coordinate),
                    "literature": float(reference),
                    "svMultiPhysics": float(computed),
                    "absolute_error": abs(float(computed - reference)),
                }
            )
        write_csv(data_dir / f"Re{reynolds}_centerline_comparison.csv", rows)
        plot_records.append(
            {
                "reynolds": reynolds,
                "y_full": y_full,
                "u_full": u_full,
                "x_full": x_full,
                "v_full": v_full,
                "y_reference": y_reference,
                "u_reference": u_reference,
                "x_reference": x_reference,
                "v_reference": v_reference,
                "vortex_reference_x": reference_x,
                "vortex_reference_y": reference_y,
                "vortex_simulated_x": vortex_x,
                "vortex_simulated_y": vortex_y,
                "vortex_psi": vortex_psi,
            }
        )
        print(
            f"  u relative L2={u_metrics['relative_l2']:.5f} (limit {u_limit:.5f}); "
            f"v relative L2={v_metrics['relative_l2']:.5f} (limit {v_limit:.5f}); "
            f"vortex distance={vortex_distance:.5f} (limit {vortex_limit:.5f}) "
            f"=> {'PASS' if passed else 'FAIL'}"
        )

    write_csv(output_root / "metrics.csv", metric_rows)
    summary = {
        "benchmark": "2D lid-driven cavity",
        "reference": "Ghia, Ghia, and Shin (1982)",
        "status": "PASS" if all_passed else "FAIL",
        "reynolds_numbers": selected,
        "metrics_file": "metrics.csv",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_centerlines(plot_records, figures_dir)
    plot_primary_vortex(plot_records, figures_dir)
    print(f"Validation status: {summary['status']}; outputs written to {output_root}")
    return all_passed


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "benchmark_config.json")
    parser.add_argument("--cases-root", type=Path, default=root / "cases")
    parser.add_argument("--literature", type=Path, default=root / "literature")
    parser.add_argument("--output", type=Path, default=root / "validation")
    parser.add_argument("--re", type=int, nargs="+", help="validate only selected Reynolds numbers")
    parser.add_argument("--no-fail", action="store_true", help="write reports but do not fail on thresholds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passed = validate(args)
    if not passed and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
