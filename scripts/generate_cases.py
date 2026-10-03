#!/usr/bin/env python3
"""Generate cavity meshes, boundary surfaces, case folders, and solver XML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, TextIO

from cavity_io import case_map, load_config, select_reynolds


def chunks(values: Iterable[object], width: int = 12):
    row: list[str] = []
    for value in values:
        row.append(str(value))
        if len(row) == width:
            yield " ".join(row)
            row = []
    if row:
        yield " ".join(row)


def write_array(stream: TextIO, values: Iterable[object], indent: str = "        ") -> None:
    for row in chunks(values):
        stream.write(f"{indent}{row}\n")


def node_id(i: int, j: int, n: int) -> int:
    return j * n + i + 1


def cell_id(i: int, j: int, n: int) -> int:
    return j * (n - 1) + i + 1


def write_volume_mesh(path: Path, n: int, length: float) -> None:
    number_of_points = n * n
    number_of_cells = (n - 1) * (n - 1)
    spacing = length / (n - 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n')
        stream.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        stream.write("  <UnstructuredGrid>\n")
        stream.write(
            f'    <Piece NumberOfPoints="{number_of_points}" NumberOfCells="{number_of_cells}">\n'
        )
        stream.write("      <PointData>\n")
        stream.write('        <DataArray type="Int32" Name="GlobalNodeID" format="ascii">\n')
        write_array(stream, range(1, number_of_points + 1))
        stream.write("        </DataArray>\n      </PointData>\n")
        stream.write("      <CellData>\n")
        stream.write('        <DataArray type="Int32" Name="GlobalElementID" format="ascii">\n')
        write_array(stream, range(1, number_of_cells + 1))
        stream.write("        </DataArray>\n      </CellData>\n")
        stream.write("      <Points>\n")
        stream.write(
            '        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">\n'
        )
        for j in range(n):
            for i in range(n):
                stream.write(f"          {i * spacing:.16g} {j * spacing:.16g} 0\n")
        stream.write("        </DataArray>\n      </Points>\n")
        stream.write("      <Cells>\n")
        stream.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        connectivity = (
            value
            for j in range(n - 1)
            for i in range(n - 1)
            for value in (
                node_id(i, j, n) - 1,
                node_id(i + 1, j, n) - 1,
                node_id(i + 1, j + 1, n) - 1,
                node_id(i, j + 1, n) - 1,
            )
        )
        write_array(stream, connectivity)
        stream.write("        </DataArray>\n")
        stream.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        write_array(stream, (4 * value for value in range(1, number_of_cells + 1)))
        stream.write("        </DataArray>\n")
        stream.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        write_array(stream, (9 for _ in range(number_of_cells)))
        stream.write("        </DataArray>\n      </Cells>\n")
        stream.write("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n")


def boundary_definition(name: str, n: int, length: float):
    spacing = length / (n - 1)
    if name == "bottom":
        nodes = [(i, 0) for i in range(n)]
        parents = [cell_id(i, 0, n) for i in range(n - 1)]
    elif name == "top":
        nodes = [(i, n - 1) for i in range(n)]
        parents = [cell_id(i, n - 2, n) for i in range(n - 1)]
    elif name == "left":
        nodes = [(0, j) for j in range(n)]
        parents = [cell_id(0, j, n) for j in range(n - 1)]
    elif name == "right":
        nodes = [(n - 1, j) for j in range(n)]
        parents = [cell_id(n - 2, j, n) for j in range(n - 1)]
    else:
        raise ValueError(f"Unknown boundary {name!r}")
    points = [(i * spacing, j * spacing, 0.0) for i, j in nodes]
    global_ids = [node_id(i, j, n) for i, j in nodes]
    return points, global_ids, parents


def write_boundary(path: Path, name: str, n: int, length: float) -> None:
    points, global_ids, parent_ids = boundary_definition(name, n, length)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write('<?xml version="1.0"?>\n')
        stream.write('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        stream.write("  <PolyData>\n")
        stream.write(
            f'    <Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="{n - 1}" '
            'NumberOfStrips="0" NumberOfPolys="0">\n'
        )
        stream.write("      <PointData>\n")
        stream.write('        <DataArray type="Int32" Name="GlobalNodeID" format="ascii">\n')
        write_array(stream, global_ids)
        stream.write("        </DataArray>\n      </PointData>\n")
        stream.write("      <CellData>\n")
        stream.write('        <DataArray type="Int32" Name="GlobalElementID" format="ascii">\n')
        write_array(stream, parent_ids)
        stream.write("        </DataArray>\n      </CellData>\n")
        stream.write("      <Points>\n")
        stream.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for x, y, z in points:
            stream.write(f"          {x:.16g} {y:.16g} {z:.16g}\n")
        stream.write("        </DataArray>\n      </Points>\n")
        stream.write("      <Lines>\n")
        stream.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        write_array(stream, (value for i in range(n - 1) for value in (i, i + 1)))
        stream.write("        </DataArray>\n")
        stream.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        write_array(stream, (2 * value for value in range(1, n)))
        stream.write("        </DataArray>\n      </Lines>\n")
        stream.write("    </Piece>\n  </PolyData>\n</VTKFile>\n")


def solver_xml(case: dict, geometry: dict) -> str:
    reynolds = int(case["reynolds"])
    density = float(geometry["density"])
    lid_velocity = float(geometry["lid_velocity"])
    length = float(geometry["length"])
    viscosity = density * lid_velocity * length / reynolds
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
    <Number_of_time_steps>{int(case["number_of_steps"])}</Number_of_time_steps>
    <Time_step_size>{float(case["time_step"]):.12g}</Time_step_size>
    <Spectral_radius_of_infinite_time_step>0.5</Spectral_radius_of_infinite_time_step>
    <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
    <Increment_in_saving_VTK_files>{int(case["save_every"])}</Increment_in_saving_VTK_files>
    <Start_saving_after_time_step>1</Start_saving_after_time_step>
    <Increment_in_saving_restart_files>{int(case["restart_every"])}</Increment_in_saving_restart_files>
    <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
    <Verbose>1</Verbose>
    <Warning>1</Warning>
    <Debug>0</Debug>
  </GeneralSimulationParameters>

  <Add_mesh name="cavity">
    <Mesh_file_path>mesh/mesh-complete.mesh.vtu</Mesh_file_path>
    <Add_face name="left"><Face_file_path>mesh/mesh-surfaces/left.vtp</Face_file_path></Add_face>
    <Add_face name="right"><Face_file_path>mesh/mesh-surfaces/right.vtp</Face_file_path></Add_face>
    <Add_face name="bottom"><Face_file_path>mesh/mesh-surfaces/bottom.vtp</Face_file_path></Add_face>
    <Add_face name="top"><Face_file_path>mesh/mesh-surfaces/top.vtp</Face_file_path></Add_face>
  </Add_mesh>

  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>15</Max_iterations>
    <Tolerance>1e-6</Tolerance>
    <Density>{density:.12g}</Density>
    <Viscosity model="Constant"><Value>{viscosity:.12g}</Value></Viscosity>
    <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
      <Traction>true</Traction>
      <Vorticity>true</Vorticity>
      <Divergence>true</Divergence>
    </Output>
    <Output type="Volume_integral"><Divergence>true</Divergence></Output>
    <LS type="NS">
      <Linear_algebra type="fsils"><Preconditioner>fsils</Preconditioner></Linear_algebra>
      <Max_iterations>15</Max_iterations>
      <NS_GM_max_iterations>500</NS_GM_max_iterations>
      <NS_CG_max_iterations>300</NS_CG_max_iterations>
      <Tolerance>0.4</Tolerance>
      <NS_GM_tolerance>1e-3</NS_GM_tolerance>
      <NS_CG_tolerance>0.02</NS_CG_tolerance>
      <Krylov_space_dimension>50</Krylov_space_dimension>
    </LS>
    <Add_BC name="left"><Type>Dir</Type><Value>0.0</Value></Add_BC>
    <Add_BC name="right"><Type>Dir</Type><Value>0.0</Value></Add_BC>
    <Add_BC name="bottom"><Type>Dir</Type><Value>0.0</Value></Add_BC>
    <Add_BC name="top">
      <Type>Dir</Type><Value>{lid_velocity:.12g}</Value>
      <Effective_direction>(1, 0)</Effective_direction>
      <Zero_out_perimeter>false</Zero_out_perimeter>
    </Add_BC>
    <Add_BC name="top">
      <Type>Dir</Type><Value>0.0</Value>
      <Effective_direction>(0, 1)</Effective_direction>
      <Zero_out_perimeter>false</Zero_out_perimeter>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
'''


def generate(config_path: Path, output_root: Path, requested: list[int] | None) -> None:
    config = load_config(config_path)
    cases = case_map(config)
    selected = select_reynolds(config, requested)
    geometry = config["geometry"]
    length = float(geometry["length"])
    for reynolds in selected:
        case = cases[reynolds]
        n = int(case["grid_points"])
        if n < 3 or n % 2 == 0:
            raise ValueError(f"Re={reynolds}: grid_points must be odd and at least 3")
        case_dir = output_root / f"Re{reynolds}"
        surface_dir = case_dir / "mesh" / "mesh-surfaces"
        print(f"Re={reynolds}: generating {n}x{n} points and {(n - 1)}x{(n - 1)} Q1 quads")
        write_volume_mesh(case_dir / "mesh" / "mesh-complete.mesh.vtu", n, length)
        for boundary in ("left", "right", "bottom", "top"):
            write_boundary(surface_dir / f"{boundary}.vtp", boundary, n, length)
        (case_dir / "solver.xml").write_text(solver_xml(case, geometry), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "benchmark_config.json")
    parser.add_argument("--output", type=Path, default=root / "cases")
    parser.add_argument("--re", type=int, nargs="+", help="generate only selected Reynolds numbers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.config.resolve(), args.output.resolve(), args.re)


if __name__ == "__main__":
    main()
