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

"""Concrete controller identity and credential bootstrap failure checks."""

import socket

import pytest
from flashdreams_controller_input import credentials

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("address", ["127.0.0.1", "10.111.69.99"])
def test_hostname_resolves_to_concrete_identity(monkeypatch, address):
    monkeypatch.setattr(credentials.socket, "gethostbyname", lambda _: address)
    assert credentials.resolve_server_address("controller.example") == address


@pytest.mark.parametrize("resolved", ["0.0.0.0", "224.0.0.1", "::1"])
def test_invalid_identity_is_rejected_before_creating_state(monkeypatch, resolved):
    monkeypatch.setattr(credentials.socket, "gethostbyname", lambda _: resolved)
    monkeypatch.setattr(
        credentials, "_state_root", lambda: pytest.fail("Must not create trust")
    )
    with pytest.raises(ValueError, match="concrete IPv4"):
        credentials.server_credentials("controller.example")


def test_unresolvable_identity_has_actionable_error(monkeypatch):
    def missing(_):
        raise socket.gaierror("unavailable")

    monkeypatch.setattr(credentials.socket, "gethostbyname", missing)
    with pytest.raises(ValueError, match="controller-host must resolve"):
        credentials.server_credentials("controller.example")


def test_missing_openssl_has_actionable_error(monkeypatch):
    def missing(*_, **__):
        raise FileNotFoundError

    monkeypatch.setattr(credentials.subprocess, "run", missing)
    with pytest.raises(ValueError, match="Required command is not installed: openssl"):
        credentials._run(["openssl", "version"])
