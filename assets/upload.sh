#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CURRENT_DIR=$(dirname $(realpath $0))

aws s3 sync $CURRENT_DIR/checkpoints s3://flashdreams/assets/checkpoints --profile team-sil-videogen --endpoint-url https://pdx.s8k.io
aws s3 sync $CURRENT_DIR/example_data s3://flashdreams/assets/example_data --profile team-sil-videogen --endpoint-url https://pdx.s8k.io
