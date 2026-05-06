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
Hydra structured config schemas for the Purem training pipeline.

Importing this module (or calling register_configs()) registers typed
dataclass schemas with the Hydra ConfigStore. Hydra then validates all
config values against these schemas at startup — wrong types and unknown
keys raise an error before any training code runs.

Usage: import this module before @hydra.main so the schemas are registered:
    import structured_configs  # noqa: F401
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainingConfig:
    # ── Required (must be set via experiment config or CLI) ──────────────────
    raw_data_dir: str = MISSING
    raw_data_dir_validation: str = MISSING
    num_time_steps: int = MISSING
    num_training_samples: int = MISSING

    # ── Optional with defaults ───────────────────────────────────────────────
    global_features_filepath: Optional[str] = None
    optimizer: str = "adam"
    optimizer_weight_decay: float = 0.0001
    num_validation_samples: int = 0
    start_lr: float = 1e-4
    end_lr: float = 3e-7
    epochs: int = 10000
    validation_freq: int = 10
    save_checkpoint_freq: int = 10
    amp: bool = True
    use_apex: bool = True
    num_dataloader_workers: int = 4
    max_workers_preprocessing: int = 64
    ckpt_path: str = "./checkpoints"

    # ── Checkpointing ────────────────────────────────────────────────────────
    top_k_checkpoints: int = 3                        # keep top-K by val loss; 0 = save every save_checkpoint_freq epochs

    # ── MLflow experiment tracking ───────────────────────────────────────────
    mlflow_tracking_uri: Optional[str] = None        # null = auto: <project_root>/mlruns/
    mlflow_run_name: Optional[str] = None             # null = experiment_name
    mlflow_run_group: Optional[str] = None            # logical grouping tag, set via CLI
    mlflow_model_registry_name: Optional[str] = None  # null = auto: experiment_name


@dataclass
class PointCloudDatapipeConfig:
    _target_: str = "datapipe.CrashPointCloudDataset"
    _convert_: str = "all"
    data_dir: str = MISSING
    global_features_filepath: Optional[str] = None
    num_samples: int = MISSING
    num_steps: int = MISSING
    sample_type: str = "all_time_steps"
    static_features: List[str] = field(default_factory=list)
    dynamic_features: List[str] = field(default_factory=list)
    dynamic_targets: List[str] = field(default_factory=list)
    global_features: Optional[List[str]] = None
    stats_dir: str = "stats"
    dt: float = 0.025


@dataclass
class GraphDatapipeConfig:
    _target_: str = "datapipe.CrashGraphDataset"
    _convert_: str = "all"
    data_dir: str = MISSING
    global_features_filepath: Optional[str] = None
    num_samples: int = MISSING
    num_steps: int = MISSING
    sample_type: str = "all_time_steps"
    static_features: List[str] = field(default_factory=list)
    dynamic_features: List[str] = field(default_factory=list)
    dynamic_targets: List[str] = field(default_factory=list)
    global_features: Optional[List[str]] = None
    stats_dir: str = "stats"
    dt: float = 0.025


@dataclass
class GeoTransolverOneShotConfig:
    _target_: str = "rollout.GeoTransolverOneShot"
    _convert_: str = "all"
    functional_dim: int = MISSING
    out_dim: int = MISSING
    geometry_dim: int = 3
    global_dim: Optional[int] = None
    slice_num: int = 128
    n_layers: int = 6
    use_te: bool = False
    time_input: bool = False
    include_local_features: bool = True
    num_time_steps: int = MISSING


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(group="training", name="base_training", node=TrainingConfig)
    cs.store(group="datapipe", name="base_point_cloud", node=PointCloudDatapipeConfig)
    cs.store(group="datapipe", name="base_graph", node=GraphDatapipeConfig)
    cs.store(group="model", name="base_geotransolver_one_shot", node=GeoTransolverOneShotConfig)


register_configs()
