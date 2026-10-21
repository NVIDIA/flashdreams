# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the guarded CloudXR ARM64/64K compatibility copy."""

from __future__ import annotations

import hashlib
import stat
import struct
from pathlib import Path

import pytest
from flashdreams_controller_input import cloudxr_64k_compat as compat

pytestmark = pytest.mark.ci_cpu

_PAGE_SIZE = 0x100
_PREPARE_PAGE_SIZE = 0x2000
_FIRST_BAD_OFFSET = 0x250
_SAFE_PADDING = 0x120


def _elf_fixture() -> bytes:
    """Build a small ELF64 fixture with the CloudXR trailing-load shape."""
    data = bytearray(0x500)
    ident = bytearray(16)
    ident[:7] = b"\x7fELF\x02\x01\x01"
    phoff = 0x40
    shoff = 0x400
    compat._ELF64_HEADER.pack_into(
        data,
        0,
        bytes(ident),
        compat._DYNAMIC_OBJECT,
        compat._AARCH64_MACHINE,
        1,
        0,
        phoff,
        shoff,
        0,
        compat._ELF64_HEADER.size,
        compat._ELF64_PROGRAM_HEADER.size,
        3,
        compat._ELF64_SECTION_HEADER.size,
        2,
        0,
    )
    programs = (
        # One already aligned segment before the insertion boundary.
        (compat._PT_LOAD, 5, 0, 0, 0, 0x200, 0x240, _PAGE_SIZE),
        # Both trailing segments need +0x20 for congruence. A safe repair
        # inserts +0x120 so the first 64K-style backing page is all zeroes.
        (compat._PT_LOAD, 6, 0x250, 0x2370, 0x2370, 0x40, 0x40, 0x10),
        (compat._PT_LOAD, 6, 0x2B0, 0x23D0, 0x23D0, 0x30, 0x30, 0x10),
    )
    for index, program in enumerate(programs):
        compat._ELF64_PROGRAM_HEADER.pack_into(
            data,
            phoff + index * compat._ELF64_PROGRAM_HEADER.size,
            *program,
        )
    compat._ELF64_SECTION_HEADER.pack_into(
        data,
        shoff + compat._ELF64_SECTION_HEADER.size,
        0,
        1,
        0,
        0,
        0x2B0,
        0x20,
        0,
        0,
        1,
        0,
    )
    data[0x250:0x290] = b"A" * 0x40
    data[0x2B0:0x2E0] = b"B" * 0x30
    return bytes(data)


def _program_offsets(data: bytes) -> tuple[tuple[int, int], ...]:
    """Return load-segment file offsets and alignments."""
    layout = compat._read_elf_layout(data)
    return tuple(
        (segment.offset, segment.align)
        for segment in layout.programs
        if segment.type == compat._PT_LOAD
    )


def test_host_preflight_is_specific_to_large_page_arm64() -> None:
    assert compat.cloudxr_64k_workaround_required(machine="aarch64", page_size=65536)
    assert not compat.cloudxr_64k_workaround_required(machine="aarch64", page_size=4096)
    assert not compat.cloudxr_64k_workaround_required(machine="x86_64", page_size=65536)


def test_repair_preserves_segments_and_inserts_a_zero_backing_page() -> None:
    original = _elf_fixture()
    repaired = compat._repair_elf_load_offsets(original, _PAGE_SIZE)

    assert len(repaired) == len(original) + _SAFE_PADDING
    assert repaired[_FIRST_BAD_OFFSET : _FIRST_BAD_OFFSET + _SAFE_PADDING] == (
        b"\0" * _SAFE_PADDING
    )
    assert repaired[0x370:0x3B0] == b"A" * 0x40
    assert repaired[0x3D0:0x400] == b"B" * 0x30
    assert _program_offsets(repaired) == (
        (0, _PAGE_SIZE),
        (0x370, _PAGE_SIZE),
        (0x3D0, _PAGE_SIZE),
    )
    layout = compat._read_elf_layout(repaired)
    assert compat._misaligned_loads(layout, _PAGE_SIZE) == ()
    assert layout.shoff == 0x520
    assert layout.sections[1].offset == 0x3D0


def test_repair_rejects_a_boundary_that_intersects_existing_data() -> None:
    original = bytearray(_elf_fixture())
    first_program = 0x40
    struct.pack_into("<Q", original, first_program + 32, 0x260)
    struct.pack_into("<Q", original, first_program + 40, 0x260)

    with pytest.raises(compat.CloudXR64KCompatibilityError, match="intersects"):
        compat._repair_elf_load_offsets(bytes(original), _PAGE_SIZE)


def test_repair_rejects_a_zero_prefix_over_file_backed_data() -> None:
    original = bytearray(_elf_fixture())
    first_program = 0x40
    struct.pack_into("<Q", original, first_program + 16, 0x2200)

    with pytest.raises(compat.CloudXR64KCompatibilityError, match="file-backed"):
        compat._repair_elf_load_offsets(bytes(original), _PAGE_SIZE)


def test_repair_rejects_incompatible_permissions_over_bss() -> None:
    original = bytearray(_elf_fixture())
    first_program = 0x40
    struct.pack_into("<Q", original, first_program + 16, 0x2200)
    struct.pack_into("<Q", original, first_program + 32, 0x100)
    struct.pack_into("<Q", original, first_program + 40, 0x200)

    with pytest.raises(compat.CloudXR64KCompatibilityError, match="permissions"):
        compat._repair_elf_load_offsets(bytes(original), _PAGE_SIZE)


def test_prepare_keeps_the_vendor_binary_and_reuses_an_atomic_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "libNvStreamServer.so"
    original = _elf_fixture()
    source.write_bytes(original)
    source.chmod(0o755)
    digest = hashlib.sha256(original).hexdigest()
    repaired = compat._repair_elf_load_offsets(original, _PREPARE_PAGE_SIZE)
    monkeypatch.setattr(compat, "_SUPPORTED_PAGE_SIZE", _PREPARE_PAGE_SIZE)
    monkeypatch.setattr(
        compat,
        "_SUPPORTED_SERVER_REPAIRED_SHA256",
        {digest: hashlib.sha256(repaired).hexdigest()},
    )

    output = compat.prepare_cloudxr_64k_preload(
        tmp_path, machine="aarch64", page_size=_PREPARE_PAGE_SIZE
    )

    assert output is not None
    assert output != source
    assert source.read_bytes() == original
    assert output.read_bytes() == repaired
    assert stat.S_IMODE(output.stat().st_mode) == 0o755
    assert (
        compat.prepare_cloudxr_64k_preload(
            tmp_path, machine="aarch64", page_size=_PREPARE_PAGE_SIZE
        )
        == output
    )


def test_prepare_refuses_an_unknown_server_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "libNvStreamServer.so"
    source.write_bytes(_elf_fixture())
    monkeypatch.setattr(compat, "_SUPPORTED_PAGE_SIZE", _PREPARE_PAGE_SIZE)
    monkeypatch.setattr(compat, "_SUPPORTED_SERVER_REPAIRED_SHA256", {})

    with pytest.raises(compat.CloudXR64KCompatibilityError, match="unknown.*sha256"):
        compat.prepare_cloudxr_64k_preload(
            tmp_path, machine="aarch64", page_size=_PREPARE_PAGE_SIZE
        )

    assert list(tmp_path.iterdir()) == [source]


def test_prepare_refuses_an_unexpected_repair_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "libNvStreamServer.so"
    original = _elf_fixture()
    source.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    monkeypatch.setattr(compat, "_SUPPORTED_PAGE_SIZE", _PREPARE_PAGE_SIZE)
    monkeypatch.setattr(
        compat,
        "_SUPPORTED_SERVER_REPAIRED_SHA256",
        {digest: "0" * 64},
    )

    with pytest.raises(compat.CloudXR64KCompatibilityError, match="deterministic hash"):
        compat.prepare_cloudxr_64k_preload(
            tmp_path, machine="aarch64", page_size=_PREPARE_PAGE_SIZE
        )

    assert list(tmp_path.iterdir()) == [source]
