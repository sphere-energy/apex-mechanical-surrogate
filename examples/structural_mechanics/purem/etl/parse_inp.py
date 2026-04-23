# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Abaqus .inp file parser.

Handles the sections needed to extract mesh topology, shell properties,
loading conditions, and file includes from Abaqus input decks.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InpData:
    """Parsed contents of an Abaqus .inp file."""

    nodes: dict[int, np.ndarray] = field(default_factory=dict)
    """node_id -> xyz array of shape [3]"""

    elements: list[tuple[int, list[int]]] = field(default_factory=list)
    """(elem_id, [n1, n2, n3, n4]) for shell quad elements"""

    element_sets: dict[str, list[int]] = field(default_factory=dict)
    """elset_name -> list of element IDs"""

    node_sets: dict[str, list[int]] = field(default_factory=dict)
    """nset_name -> list of node IDs"""

    includes: list[str] = field(default_factory=list)
    """Relative paths from *INCLUDE directives"""

    shell_thickness: Optional[float] = None
    """Shell section thickness in mm"""

    velocity_bcs: dict[str, dict[int, float]] = field(default_factory=dict)
    """nset_name -> {dof: velocity}"""

    friction_coefficient: Optional[float] = None

    step_duration: Optional[float] = None
    """Total simulation time in seconds"""


def _strip_comment(line: str) -> str:
    """Remove Abaqus inline comments (** prefix lines already skipped)."""
    return line.split("**")[0].strip()


def parse_inp(filepath: str) -> InpData:
    """
    Parse an Abaqus .inp file and return structured data.

    Only parses the named file (does not recurse into *INCLUDE files).
    """
    data = InpData()
    section = None
    elset_name = None
    nset_name = None
    current_elements: list[tuple[int, list[int]]] = []
    pending_shell_elset = None
    pending_velocity_nset = None
    pending_velocity_dof_start = None
    pending_velocity_dof_end = None

    with open(filepath, "r", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            # Skip blank lines and full-line comments
            if not line or line.startswith("**"):
                continue

            stripped = _strip_comment(line)
            upper = stripped.upper()

            # Keyword lines
            if upper.startswith("*"):
                parts = [p.strip() for p in stripped.split(",")]
                keyword = parts[0].upper()

                if keyword == "*NODE":
                    section = "NODE"
                    nset_name = None
                    for p in parts[1:]:
                        if p.upper().startswith("NSET="):
                            nset_name = p.split("=", 1)[1].strip()
                    continue

                elif keyword == "*ELEMENT":
                    section = "ELEMENT"
                    elset_name = None
                    for p in parts[1:]:
                        if p.upper().startswith("ELSET="):
                            elset_name = p.split("=", 1)[1].strip()
                    if elset_name and elset_name not in data.element_sets:
                        data.element_sets[elset_name] = []
                    continue

                elif keyword == "*ELSET":
                    section = "ELSET"
                    elset_name = None
                    for p in parts[1:]:
                        if p.upper().startswith("ELSET="):
                            elset_name = p.split("=", 1)[1].strip()
                    if elset_name and elset_name not in data.element_sets:
                        data.element_sets[elset_name] = []
                    continue

                elif keyword == "*NSET":
                    section = "NSET"
                    nset_name = None
                    for p in parts[1:]:
                        if p.upper().startswith("NSET="):
                            nset_name = p.split("=", 1)[1].strip()
                    if nset_name and nset_name not in data.node_sets:
                        data.node_sets[nset_name] = []
                    continue

                elif keyword == "*SHELL SECTION":
                    section = "SHELL_SECTION"
                    for p in parts[1:]:
                        if p.upper().startswith("ELSET="):
                            pending_shell_elset = p.split("=", 1)[1].strip()
                    continue

                elif keyword == "*INCLUDE":
                    for p in parts[1:]:
                        if p.upper().startswith("INPUT="):
                            inc_path = p.split("=", 1)[1].strip()
                            data.includes.append(inc_path)
                    section = None
                    continue

                elif keyword == "*FRICTION":
                    section = "FRICTION"
                    continue

                elif keyword == "*DYNAMIC":
                    section = "DYNAMIC"
                    # *DYNAMIC, EXPLICIT
                    # next data line is ", duration" or "dt, duration"
                    continue

                elif keyword == "*BOUNDARY":
                    # Check for TYPE=VELOCITY
                    is_velocity = any(
                        "TYPE=VELOCITY" in p.upper() for p in parts[1:]
                    )
                    section = "VELOCITY_BC" if is_velocity else "BOUNDARY"
                    pending_velocity_nset = None
                    continue

                elif keyword == "*STEP":
                    section = "STEP"
                    continue

                elif keyword == "*END STEP":
                    section = None
                    continue

                else:
                    section = None
                    continue

            # Data lines
            if section == "NODE":
                vals = [v.strip() for v in stripped.split(",")]
                if len(vals) >= 4:
                    try:
                        node_id = int(vals[0])
                        xyz = np.array([float(vals[1]), float(vals[2]), float(vals[3])])
                        data.nodes[node_id] = xyz
                        if nset_name:
                            data.node_sets.setdefault(nset_name, []).append(node_id)
                    except ValueError:
                        pass

            elif section == "ELEMENT":
                vals = [v.strip() for v in stripped.split(",")]
                if len(vals) >= 5:
                    try:
                        elem_id = int(vals[0])
                        node_ids = [int(v) for v in vals[1:] if v]
                        data.elements.append((elem_id, node_ids))
                        if elset_name:
                            data.element_sets[elset_name].append(elem_id)
                    except ValueError:
                        pass

            elif section == "ELSET":
                vals = [v.strip() for v in stripped.split(",")]
                for v in vals:
                    if v and elset_name:
                        try:
                            data.element_sets[elset_name].append(int(v))
                        except ValueError:
                            # Could be another elset name reference — skip
                            pass

            elif section == "NSET":
                vals = [v.strip() for v in stripped.split(",")]
                for v in vals:
                    if v and nset_name:
                        try:
                            data.node_sets[nset_name].append(int(v))
                        except ValueError:
                            pass

            elif section == "SHELL_SECTION":
                # First data line is: thickness [, integration_points]
                vals = [v.strip() for v in stripped.split(",")]
                try:
                    data.shell_thickness = float(vals[0])
                except ValueError:
                    pass
                section = None

            elif section == "FRICTION":
                try:
                    data.friction_coefficient = float(stripped.split(",")[0].strip())
                except ValueError:
                    pass
                section = None

            elif section == "DYNAMIC":
                # Format: [min_dt], duration
                vals = [v.strip() for v in stripped.split(",")]
                try:
                    if len(vals) >= 2:
                        data.step_duration = float(vals[1])
                    elif len(vals) == 1 and vals[0]:
                        data.step_duration = float(vals[0])
                except ValueError:
                    pass
                section = None

            elif section == "VELOCITY_BC":
                # Format: nset, dof_start[, dof_end], value
                vals = [v.strip() for v in stripped.split(",")]
                if len(vals) >= 3:
                    try:
                        nset = vals[0]
                        dof_start = int(vals[1])
                        if len(vals) == 3:
                            dof_end = dof_start
                            value = float(vals[2])
                        else:
                            dof_end = int(vals[2])
                            value = float(vals[3])
                        if nset not in data.velocity_bcs:
                            data.velocity_bcs[nset] = {}
                        for dof in range(dof_start, dof_end + 1):
                            data.velocity_bcs[nset][dof] = value
                    except (ValueError, IndexError):
                        pass

    data.elements = current_elements if current_elements else data.elements
    return data


# ---------------------------------------------------------------------------
# Filename metadata parsing
# ---------------------------------------------------------------------------

_VARIANT_RE = re.compile(
    r"B_(?P<W>[\d.]+)x(?P<H>[\d.]+)x(?P<t>[\d.]+)"
    r"_L(?P<L>[\d.]+)"
    r"_R(?P<R>[\d.]+)"
    r"_H(?P<Hn>[\d.]+)n(?P<n>[\d.]+)"
    r"_Rh(?P<rh_int>\d+)d(?P<rh_dec>\d+)"
    r"_t(?P<t2>[\d.]+)"
)


def parse_variant_params(name: str) -> dict[str, float]:
    """
    Extract geometry parameters from a variant filename/stem.

    Example: 'B_160x40x1_L1000_R2_H8n10_Rh2d0_t1' ->
        {'width_mm': 160.0, 'height_mm': 40.0, 'thickness_mm': 1.0,
         'length_mm': 1000.0, 'punch_radius_mm': 2.0,
         'hole_height_mm': 8.0, 'hole_n': 10.0, 'rh': 2.0}
    """
    m = _VARIANT_RE.search(name)
    if m is None:
        return {}
    g = m.groupdict()
    rh = float(g["rh_int"]) + float(g["rh_dec"]) / (10 ** len(g["rh_dec"]))
    return {
        "width_mm": float(g["W"]),
        "height_mm": float(g["H"]),
        "section_thickness_mm": float(g["t"]),
        "length_mm": float(g["L"]),
        "punch_radius_mm": float(g["R"]),
        "hole_height_mm": float(g["Hn"]),
        "hole_n": float(g["n"]),
        "rh": rh,
    }


# ---------------------------------------------------------------------------
# Mesh-topology helpers
# ---------------------------------------------------------------------------

def build_edges_from_elements(
    elements: list[tuple[int, list[int]]],
    node_id_to_idx: dict[int, int],
) -> np.ndarray:
    """
    Build undirected edge array [E, 2] from quad/shell element connectivity.

    Each quad element (n0, n1, n2, n3) contributes 4 edges: n0-n1, n1-n2, n2-n3, n3-n0.
    Edges are deduplicated and stored with src < dst.

    Args:
        elements: list of (elem_id, [node_id, ...])
        node_id_to_idx: mapping from Abaqus node ID to 0-indexed position in the array

    Returns:
        edges: int64 array [E, 2] with 0-indexed node indices
    """
    edge_set: set[tuple[int, int]] = set()
    for _eid, nids in elements:
        n = len(nids)
        for i in range(n):
            a = node_id_to_idx.get(nids[i])
            b = node_id_to_idx.get(nids[(i + 1) % n])
            if a is not None and b is not None and a != b:
                edge_set.add((min(a, b), max(a, b)))
    if not edge_set:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(edge_set), dtype=np.int64)


def find_mesh_file(
    include_path: str,
    reference_dir: str,
    search_dirs: list[str],
) -> Optional[str]:
    """
    Locate a mesh file referenced by *INCLUDE.

    Search order:
    1. Relative to the directory containing the master .inp
    2. Each directory in search_dirs (basename match)

    Returns the absolute path if found, else None.
    """
    # 1. Relative to reference directory
    candidate = os.path.normpath(os.path.join(reference_dir, include_path))
    if os.path.isfile(candidate):
        return candidate

    # 2. Basename match in search dirs
    basename = os.path.basename(include_path)
    for d in search_dirs:
        candidate = os.path.join(d, basename)
        if os.path.isfile(candidate):
            return candidate

    return None
