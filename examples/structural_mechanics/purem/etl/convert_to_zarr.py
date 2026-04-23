#!/usr/bin/env python3
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
ETL: Convert Abaqus crash simulation outputs to Zarr format.

Each simulation is identified by a master_variant_*.inp file alongside matching
parquet output files. The ETL:
  1. Parses the master .inp to find included mesh files and loading conditions
  2. Parses the deformable structure mesh for node coordinates and element topology
  3. Reads nodal_output.parquet and pivots to [T, N, 3] positions
  4. Reads stress_output.parquet, node-averages stresses, computes von Mises
  5. Writes a Zarr store per simulation
  6. Writes/updates global_features.json in the output directory

Usage:
    python convert_to_zarr.py \\
        --raw_dirs /path/to/sim_outputs [dir2 ...] \\
        --mesh_dirs /path/to/mesh_files [dir2 ...] \\
        --output_dir /path/to/zarr_output \\
        [--num_workers N]
"""

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import zarr

# Allow running from the etl/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from etl.parse_inp import (
    InpData,
    build_edges_from_elements,
    find_mesh_file,
    parse_inp,
    parse_variant_params,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation case discovery
# ---------------------------------------------------------------------------

_MASTER_RE = re.compile(r"^(master_variant_(B_.+))\.inp$")
_PARQUET_SUFFIX = {
    "nodal": "_all_output_nodal_output.parquet",
    "stress": "_all_output_stress_output.parquet",
    "history": "_all_output_history_output.parquet",
}


def find_sim_cases(raw_dirs: list[str]) -> list[dict]:
    """
    Find all simulation cases in the given directories.

    A case is a master_variant_*.inp file with at least a nodal parquet alongside it.
    """
    cases = []
    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            log.warning(f"Skipping missing directory: {raw_dir}")
            continue
        for fname in sorted(os.listdir(raw_dir)):
            m = _MASTER_RE.match(fname)
            if m is None:
                continue
            full_stem = m.group(1)   # e.g. master_variant_B_160x40x1_...
            variant = m.group(2)     # e.g. B_160x40x1_...
            master_path = os.path.join(raw_dir, fname)
            parquets = {
                k: os.path.join(raw_dir, f"{full_stem}{v}")
                for k, v in _PARQUET_SUFFIX.items()
            }
            nodal_path = parquets["nodal"]
            if not os.path.isfile(nodal_path):
                log.warning(f"Missing nodal parquet for {variant}, skipping")
                continue
            cases.append(
                {
                    "variant": variant,
                    "master_path": master_path,
                    "master_dir": raw_dir,
                    "parquets": {k: v for k, v in parquets.items() if os.path.isfile(v)},
                }
            )
    return cases


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _is_rigid_include(include_path: str) -> bool:
    """Heuristic: paths referencing 'Rigid' are the punch mesh, not the deformable structure."""
    return "rigid" in os.path.basename(include_path).lower()


def load_deformable_mesh(
    master_data: InpData,
    master_dir: str,
    mesh_dirs: list[str],
) -> tuple[InpData, str]:
    """
    Find and parse the deformable structure mesh referenced by the master .inp.

    Returns (InpData, resolved_path).
    Raises FileNotFoundError if the mesh cannot be located.
    """
    for inc in master_data.includes:
        if _is_rigid_include(inc):
            continue
        resolved = find_mesh_file(inc, master_dir, mesh_dirs)
        if resolved is not None:
            log.debug(f"  Found deformable mesh: {resolved}")
            return parse_inp(resolved), resolved
    raise FileNotFoundError(
        f"Could not find deformable structure mesh in includes {master_data.includes}. "
        f"Searched dirs: {[master_dir] + mesh_dirs}"
    )


# ---------------------------------------------------------------------------
# Parquet processing
# ---------------------------------------------------------------------------

def _load_nodal(path: str, deformable_ids: set[int]) -> pd.DataFrame:
    """Load and filter nodal output to deformable nodes only."""
    df = pd.read_parquet(path, columns=["Frame", "NodeLabel", "U1", "U2", "U3"])
    return df[df["NodeLabel"].isin(deformable_ids)].copy()


def nodal_to_mesh_pos(
    nodal_df: pd.DataFrame,
    initial_xyz: np.ndarray,
    node_id_to_idx: dict[int, int],
) -> np.ndarray:
    """
    Convert nodal displacement parquet to absolute position array.

    Args:
        nodal_df: filtered nodal output DataFrame (Frame, NodeLabel, U1, U2, U3)
        initial_xyz: [N, 3] initial coordinates for deformable nodes (0-indexed)
        node_id_to_idx: mapping Abaqus node ID → 0-based index

    Returns:
        mesh_pos: float32 [T, N, 3] absolute positions
    """
    frames = sorted(nodal_df["Frame"].unique())
    T = len(frames)
    N = len(node_id_to_idx)

    displacements = np.zeros((T, N, 3), dtype=np.float32)
    for fi, frame_id in enumerate(frames):
        frame_df = nodal_df[nodal_df["Frame"] == frame_id]
        idx_arr = frame_df["NodeLabel"].map(node_id_to_idx).values
        displacements[fi, idx_arr, 0] = frame_df["U1"].values.astype(np.float32)
        displacements[fi, idx_arr, 1] = frame_df["U2"].values.astype(np.float32)
        displacements[fi, idx_arr, 2] = frame_df["U3"].values.astype(np.float32)

    # initial_xyz[idx] is the reference position for each node
    mesh_pos = initial_xyz[np.newaxis, :, :] + displacements  # broadcast [1,N,3] + [T,N,3]
    return mesh_pos


def _load_stress(path: str, deformable_elem_ids: set[int]) -> pd.DataFrame:
    """Load and filter stress output to deformable elements only."""
    df = pd.read_parquet(
        path,
        columns=["Frame", "ElementLabel", "IntPoint", "S11", "S22", "S33", "S12", "S13", "S23"],
    )
    return df[df["ElementLabel"].isin(deformable_elem_ids)].copy()


def stress_to_node(
    stress_df: pd.DataFrame,
    elements: list[tuple[int, list[int]]],
    elem_id_to_idx: dict[int, int],
    node_id_to_idx: dict[int, int],
    components: list[str],
) -> dict[str, np.ndarray]:
    """
    Node-average element stresses.

    For each node, averages the stress from all elements sharing that node.
    Shell elements produce plane-stress (S33=S13=S23=0 by construction).

    Returns:
        dict mapping component name → float32 [T, N]
    Also includes 'von_mises' key.
    """
    frames = sorted(stress_df["Frame"].unique())
    T = len(frames)
    N = len(node_id_to_idx)

    # Build element → sorted list of 0-indexed node indices
    # and node → list of 0-indexed element indices
    elem_nodes: dict[int, list[int]] = {}  # 0-indexed elem → [0-indexed nodes]
    node_elems: dict[int, list[int]] = {}  # 0-indexed node → [0-indexed elems]

    for eid, nids in elements:
        eidx = elem_id_to_idx.get(eid)
        if eidx is None:
            continue
        node_idxs = [node_id_to_idx[n] for n in nids if n in node_id_to_idx]
        elem_nodes[eidx] = node_idxs
        for nidx in node_idxs:
            node_elems.setdefault(nidx, []).append(eidx)

    # For each frame, average elements at integration points, then average to nodes
    result = {c: np.zeros((T, N), dtype=np.float32) for c in components}

    for fi, frame_id in enumerate(frames):
        frame_df = stress_df[stress_df["Frame"] == frame_id]

        # Average integration points per element → per-element stress
        elem_stress = (
            frame_df.groupby("ElementLabel")[components].mean()
        )
        # Map to 0-indexed arrays
        stress_arr = np.zeros((len(elem_id_to_idx), len(components)), dtype=np.float32)
        for eid, row in elem_stress.iterrows():
            eidx = elem_id_to_idx.get(int(eid))
            if eidx is not None:
                for ci, c in enumerate(components):
                    stress_arr[eidx, ci] = row[c]

        # Node-average
        for nidx, eidxs in node_elems.items():
            for ci, c in enumerate(components):
                result[c][fi, nidx] = stress_arr[eidxs, ci].mean()

    # Von Mises (plane stress: S33=S13=S23≈0)
    s11 = result["S11"]
    s22 = result["S22"]
    s12 = result["S12"]
    result["von_mises"] = np.sqrt(
        np.clip(s11**2 - s11 * s22 + s22**2 + 3.0 * s12**2, a_min=0.0, a_max=None)
    ).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# Global features extraction
# ---------------------------------------------------------------------------

def extract_global_features(
    variant: str,
    master_data: InpData,
    mesh_data: InpData,
) -> dict[str, float]:
    """Build per-simulation global feature dict from filename and .inp contents."""
    params = parse_variant_params(variant)

    # Loading conditions from master .inp
    punch_velocity = None
    for nset, dofs in master_data.velocity_bcs.items():
        # DOF 2 = Y velocity (loading direction)
        if 2 in dofs and dofs[2] != 0.0:
            punch_velocity = abs(dofs[2])
            break

    feats: dict[str, float] = dict(params)
    if punch_velocity is not None:
        feats["punch_velocity_mm_per_s"] = punch_velocity
    if master_data.friction_coefficient is not None:
        feats["friction_coefficient"] = master_data.friction_coefficient
    if master_data.step_duration is not None:
        feats["step_duration_s"] = master_data.step_duration
    if mesh_data.shell_thickness is not None:
        feats["shell_thickness_mm"] = mesh_data.shell_thickness

    return feats


# ---------------------------------------------------------------------------
# Per-case conversion
# ---------------------------------------------------------------------------

STRESS_COMPONENTS = ["S11", "S22", "S33", "S12", "S13", "S23"]


def convert_case(
    case: dict,
    mesh_dirs: list[str],
    output_dir: str,
) -> tuple[str, dict]:
    """
    Convert one simulation case to a Zarr store.

    Returns (variant_name, global_features_dict).
    """
    variant = case["variant"]
    zarr_name = f"{variant}.zarr"
    zarr_path = os.path.join(output_dir, zarr_name)

    if os.path.exists(zarr_path):
        log.info(f"  [skip] Zarr store already exists: {zarr_name}")
        # Still return features for JSON
        gf_path = os.path.join(output_dir, "global_features.json")
        if os.path.isfile(gf_path):
            with open(gf_path) as f:
                all_gf = json.load(f)
            return variant, all_gf.get(zarr_name, {})

    log.info(f"Processing: {variant}")

    # 1. Parse master .inp
    master_data = parse_inp(case["master_path"])

    # 2. Find and parse deformable mesh
    mesh_data, _ = load_deformable_mesh(
        master_data, case["master_dir"], mesh_dirs
    )
    if not mesh_data.nodes:
        raise ValueError(f"No nodes found in deformable mesh for {variant}")
    if not mesh_data.elements:
        raise ValueError(f"No elements found in deformable mesh for {variant}")

    # 3. Build sequential node mapping (deformable nodes only)
    deform_ids_sorted = sorted(mesh_data.nodes.keys())
    node_id_to_idx = {nid: idx for idx, nid in enumerate(deform_ids_sorted)}
    N = len(deform_ids_sorted)

    initial_xyz = np.zeros((N, 3), dtype=np.float32)
    for nid, idx in node_id_to_idx.items():
        initial_xyz[idx] = mesh_data.nodes[nid].astype(np.float32)

    # 4. Build edges
    edges = build_edges_from_elements(mesh_data.elements, node_id_to_idx)
    log.info(f"  Nodes: {N:,}  Elements: {len(mesh_data.elements):,}  Edges: {len(edges):,}")

    # 5. Static feature: thickness (per-node, constant)
    t = mesh_data.shell_thickness if mesh_data.shell_thickness is not None else 1.0
    thickness = np.full(N, t, dtype=np.float32)

    # 6. Parse nodal output → mesh_pos [T, N, 3]
    deform_id_set = set(deform_ids_sorted)
    nodal_df = _load_nodal(case["parquets"]["nodal"], deform_id_set)
    mesh_pos = nodal_to_mesh_pos(nodal_df, initial_xyz, node_id_to_idx)
    T = mesh_pos.shape[0]
    log.info(f"  Frames: {T}  mesh_pos shape: {mesh_pos.shape}")

    # 7. Parse stress output → node-averaged [T, N] per component
    stress_arrays: dict[str, np.ndarray] = {}
    if "stress" in case["parquets"]:
        elem_ids_sorted = sorted(e[0] for e in mesh_data.elements)
        elem_id_to_idx = {eid: idx for idx, eid in enumerate(elem_ids_sorted)}
        deform_elem_set = set(elem_ids_sorted)

        stress_df = _load_stress(case["parquets"]["stress"], deform_elem_set)
        stress_arrays = stress_to_node(
            stress_df,
            mesh_data.elements,
            elem_id_to_idx,
            node_id_to_idx,
            ["S11", "S22", "S12"],  # S33=S13=S23=0 for plane-stress shells
        )
        log.info(f"  Stress arrays written: {list(stress_arrays.keys())}")
    else:
        log.warning(f"  No stress parquet for {variant} — skipping stress arrays")

    # 8. Write Zarr store (compatible with zarr v2 and v3)
    store = zarr.open(zarr_path, mode="w")
    store["mesh_pos"] = mesh_pos.astype(np.float32)
    store["edges"] = edges.astype(np.int64)
    store["thickness"] = thickness.astype(np.float32)
    for name, arr in stress_arrays.items():
        store[name] = arr.astype(np.float32)

    log.info(f"  Wrote Zarr store: {zarr_path}")

    # 9. Global features
    global_feats = extract_global_features(variant, master_data, mesh_data)
    return zarr_name, global_feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Abaqus simulation outputs to Zarr stores for PhysicsNemo training."
    )
    parser.add_argument(
        "--raw_dirs",
        nargs="+",
        required=True,
        help="Directories containing master_variant_*.inp + parquet files",
    )
    parser.add_argument(
        "--mesh_dirs",
        nargs="+",
        default=[],
        help="Additional directories to search for referenced mesh .inp files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for Zarr stores and global_features.json",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cases = find_sim_cases(args.raw_dirs)
    if not cases:
        log.error("No simulation cases found. Check --raw_dirs.")
        sys.exit(1)
    log.info(f"Found {len(cases)} simulation case(s)")

    # Load existing global features if present
    gf_path = os.path.join(args.output_dir, "global_features.json")
    if os.path.isfile(gf_path):
        with open(gf_path) as f:
            all_global_features: dict[str, dict] = json.load(f)
    else:
        all_global_features = {}

    if args.num_workers > 1:
        futures = {}
        with ProcessPoolExecutor(max_workers=args.num_workers) as exe:
            for case in cases:
                fut = exe.submit(
                    convert_case, case, args.mesh_dirs, args.output_dir
                )
                futures[fut] = case["variant"]
        for fut in as_completed(futures):
            variant = futures[fut]
            try:
                zarr_name, gf = fut.result()
                all_global_features[zarr_name] = gf
            except Exception as exc:
                log.error(f"Failed to process {variant}: {exc}")
    else:
        for case in cases:
            try:
                zarr_name, gf = convert_case(case, args.mesh_dirs, args.output_dir)
                all_global_features[zarr_name] = gf
            except Exception as exc:
                log.error(f"Failed to process {case['variant']}: {exc}")

    # Write updated global features
    with open(gf_path, "w") as f:
        json.dump(all_global_features, f, indent=2)
    log.info(f"Wrote global features: {gf_path} ({len(all_global_features)} entries)")


if __name__ == "__main__":
    main()
