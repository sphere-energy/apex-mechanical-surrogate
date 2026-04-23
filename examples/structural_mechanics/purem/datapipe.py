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
Re-exports the shared datapipe classes from crash/ so that Hydra configs
and train.py can import them uniformly as `from datapipe import ...`.
"""

import os
import sys

# Add crash/ to the path so imports resolve correctly when running from purem/
_crash_dir = os.path.join(os.path.dirname(__file__), "..", "crash")
if _crash_dir not in sys.path:
    sys.path.insert(0, _crash_dir)

from datapipe import (  # noqa: F401  (re-export)
    CrashBaseDataset,
    CrashGraphDataset,
    CrashPointCloudDataset,
    SimSample,
    simsample_collate,
)
