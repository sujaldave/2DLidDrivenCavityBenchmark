#!/usr/bin/env python3
"""Run generated cavity cases with svMultiPhysics, one case at a time."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from cavity_io import latest_result, load_config, select_reynolds


def executable(value: str, label: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if not candidate.is_file():
            raise FileNotFoundError(f"{label} executable does not exist: {candidate}")
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(f"Could not find {label} executable {value!r} on PATH")
    return resolved


def has_result(case_dir: Path) -> bool:
    try:
        latest_result(case_dir)
    except FileNotFoundError:
        return False
    return True


def run_and_log(command: list[str], case_dir: Path, dry_run: bool) -> int:
    rendered = shlex.join(command)
    print(f"[{case_dir.name}] {rendered}")
    if dry_run:
        return 0
    log_path = case_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=case_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_solver = os.environ.get("SVMP_BIN", "svmultiphysics")
    default_mpi = os.environ.get("MPIEXEC", "mpirun")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "benchmark_config.json")
    parser.add_argument("--cases-root", type=Path, default=root / "cases")
    parser.add_argument("--solver", default=default_solver, help="svMultiPhysics executable")
    parser.add_argument("--mpi-exec", default=default_mpi, help="MPI launcher")
    parser.add_argument("--np", type=int, help="MPI processes; defaults to benchmark_config.json")
    parser.add_argument("--re", type=int, nargs="+", help="run only selected Reynolds numbers")
    parser.add_argument("--rerun", action="store_true", help="run even when a result_*.vtu exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    parser.add_argument(
        "--continue-on-error", action="store_true", help="attempt later cases after a solver failure"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    selected = select_reynolds(config, args.re)
    process_count = args.np if args.np is not None else int(config.get("default_processes", 1))
    if process_count < 1:
        raise SystemExit("--np must be at least 1")

    try:
        solver = executable(args.solver, "solver")
        mpi = executable(args.mpi_exec, "MPI") if process_count > 1 else None
    except FileNotFoundError as exc:
        if not args.dry_run:
            raise SystemExit(str(exc)) from exc
        solver = args.solver
        mpi = args.mpi_exec if process_count > 1 else None

    failures: list[int] = []
    for reynolds in selected:
        case_dir = args.cases_root.resolve() / f"Re{reynolds}"
        solver_xml = case_dir / "solver.xml"
        if not solver_xml.is_file():
            raise SystemExit(f"Missing {solver_xml}; run generate_cases.py first")
        if has_result(case_dir) and not args.rerun:
            print(f"[Re{reynolds}] result already exists; use --rerun to run again")
            continue
        command = [solver, "solver.xml"]
        if process_count > 1:
            assert mpi is not None
            command = [mpi, "-n", str(process_count), *command]
        return_code = run_and_log(command, case_dir, args.dry_run)
        if return_code:
            failures.append(reynolds)
            print(f"[Re{reynolds}] solver exited with status {return_code}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if failures:
        raise SystemExit(f"Solver failed for Reynolds number(s): {failures}")


if __name__ == "__main__":
    main()
