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
Reader for Purem Abaqus Zarr stores.

Each .zarr store contains:
  mesh_pos    float32 [T, N, 3]    temporal absolute positions
  edges       int64   [E, 2]       undirected edge list (0-indexed)
  thickness   float32 [N]          static per-node shell thickness
  S11         float32 [T, N]       node-averaged stress (time-series)
  S22         float32 [T, N]
  S12         float32 [T, N]
  von_mises   float32 [T, N]

This reader is compatible with CrashBaseDataset from crash/datapipe.py.
2D time-series arrays [T, N] are expanded to per-timestep point_data entries
{name}_t0, {name}_t1, ... so that CrashBaseDataset._get_dynamic_feature() works
without modification.
"""

import json
import os
import re

import numpy as np
import zarr


def _natural_key(name: str) -> list:
    return [int(s) if s.isdigit() else s.lower() for s in re.findall(r"\d+|\D+", name)]


def find_zarr_stores(base_data_dir: str) -> list[str]:
    """Return sorted list of .zarr store paths in the given directory."""
    if not os.path.isdir(base_data_dir):
        return []
    stores = [
        os.path.join(base_data_dir, f)
        for f in os.listdir(base_data_dir)
        if f.endswith(".zarr") and os.path.isdir(os.path.join(base_data_dir, f))
    ]
    return sorted(stores, key=lambda p: _natural_key(os.path.basename(p)))


def load_purem_zarr_store(zarr_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load a Purem Zarr store.

    Returns:
        mesh_pos: float64 [T, N, 3]
        edges: int64 [E, 2]
        point_data: dict with static arrays [N] and expanded time-series {name}_t{i} [N]
    """
    store = zarr.open(zarr_path, mode="r")

    if "mesh_pos" not in store:
        raise KeyError(f"'mesh_pos' not found in {zarr_path}")
    if "edges" not in store:
        raise KeyError(f"'edges' not found in {zarr_path}")

    mesh_pos = np.array(store["mesh_pos"][:], dtype=np.float64)  # [T, N, 3]
    edges = np.array(store["edges"][:], dtype=np.int64)          # [E, 2]

    T = mesh_pos.shape[0]
    N = mesh_pos.shape[1]

    point_data: dict[str, np.ndarray] = {}
    for name in store.keys():
        if name in ("mesh_pos", "edges"):
            continue
        arr = np.array(store[name][:])
        if arr.ndim == 1:
            # Static feature [N]
            if len(arr) != N:
                raise ValueError(
                    f"Static array '{name}' length {len(arr)} != N={N} in {zarr_path}"
                )
            point_data[name] = arr.astype(np.float32)
        elif arr.ndim == 2 and arr.shape[0] == T and arr.shape[1] == N:
            # Time-series [T, N]: expand to per-timestep entries
            for t in range(T):
                point_data[f"{name}_t{t}"] = arr[t].astype(np.float32)
        else:
            raise ValueError(
                f"Array '{name}' has unexpected shape {arr.shape} in {zarr_path}. "
                f"Expected [N={N}] or [T={T}, N={N}]."
            )

    return mesh_pos, edges, point_data


def process_purem_data(
    data_dir: str,
    num_samples: int,
    global_features_filepath: str | None = None,
    logger=None,
):
    """
    Load Purem Zarr stores from a directory.

    Returns:
        srcs: list of source node arrays for graph edges
        dsts: list of destination node arrays for graph edges
        point_data_all: list of dicts with 'coords' [T,N,3] and optional point_data fields
        global_features_all: list of per-sample global feature dicts
    """
    zarr_stores = find_zarr_stores(data_dir)
    if not zarr_stores:
        msg = f"No .zarr stores found in: {data_dir}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    # Load global features index (keyed by zarr store basename).
    # Only loaded when explicitly requested; empty dicts otherwise so that
    # CrashBaseDataset's global_features validation is not triggered.
    all_gf: dict[str, dict] = {}
    if global_features_filepath and os.path.isfile(global_features_filepath):
        with open(global_features_filepath) as f:
            all_gf = json.load(f)

    srcs, dsts = [], []
    point_data_all = []
    global_features_all = []

    processed = 0
    for zarr_path in zarr_stores:
        if processed >= num_samples:
            break
        basename = os.path.basename(zarr_path)
        if logger:
            logger.info(f"Loading: {basename}")

        try:
            mesh_pos, edges, point_data = load_purem_zarr_store(zarr_path)
        except Exception as exc:
            if logger:
                logger.error(f"Error loading {basename}: {exc}")
            raise

        # Validate edge indices
        N = mesh_pos.shape[1]
        if edges.size > 0:
            if edges.min() < 0 or edges.max() >= N:
                raise ValueError(
                    f"Edge indices out of bounds [0, {N-1}] in {zarr_path}"
                )

        src, dst = edges.T if edges.ndim == 2 and len(edges) > 0 else (
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        )
        srcs.append(src)
        dsts.append(dst)

        # Pack into record format expected by CrashBaseDataset
        record = {"coords": mesh_pos, "point_data": point_data}
        point_data_all.append(record)

        # Global features for this store
        gf = all_gf.get(basename, {})
        global_features_all.append(gf)

        processed += 1

    if logger:
        logger.info(f"Loaded {processed} Purem Zarr store(s)")

    return srcs, dsts, point_data_all, global_features_all


class Reader:
    """
    Reader for Purem Abaqus Zarr stores.

    Drop-in replacement for crash/zarr_reader.Reader.
    Compatible with CrashBaseDataset, CrashGraphDataset, CrashPointCloudDataset.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        data_dir: str,
        num_samples: int,
        split: str | None = None,
        global_features_filepath: str | None = None,
        logger=None,
        **kwargs,
    ):
        """
        Load Purem Zarr stores.

        Args:
            data_dir: Directory containing .zarr stores and global_features.json
            num_samples: Maximum number of samples to load
            split: Data split (unused — Zarr stores are pre-split into separate dirs)
            global_features_filepath: Path to global_features.json (default: data_dir/global_features.json)
            logger: Optional logger

        Returns:
            srcs, dsts, point_data_all, global_features_all
        """
        return process_purem_data(
            data_dir=data_dir,
            num_samples=num_samples,
            global_features_filepath=global_features_filepath,
            logger=logger,
        )
