# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3 credential-file safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from flashdreams.core.io.s3_filesystem import S3FileSystem

pytestmark = pytest.mark.ci_cpu


def test_s3_credentials_must_be_owner_only(tmp_path: Path) -> None:
    """Reject credential files readable by users other than their owner."""
    credentials = tmp_path / "credentials.json"
    credentials.write_text("{}", encoding="utf-8")
    credentials.chmod(0o640)

    with pytest.raises(PermissionError, match="chmod 600"):
        S3FileSystem(str(credentials))
