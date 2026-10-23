# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guarded CloudXR ELF compatibility for 64 KiB ARM64 hosts.

CloudXR 6.3's ARM64 ``libNvStreamServer.so`` contains two trailing load
segments whose file offsets are only congruent with their virtual addresses at
4 KiB.  Glibc rejects that layout on the 64 KiB kernels used by GB-class DGX
systems.  This module creates a separate, locally derived copy with corrected
file offsets.  It never changes or redistributes the installed NVIDIA binary.
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

_AARCH64_MACHINE = 183
_DYNAMIC_OBJECT = 3
_PT_LOAD = 1
_SHT_NOBITS = 8
_ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF64_PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
_ELF64_SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
_SUPPORTED_PAGE_SIZE = 65536

# Isaac Teleop 1.4.142's stable and experimental CloudXR 6.3 ARM64
# libraries.  Refuse to rewrite an unknown proprietary binary even when it
# happens to have a superficially similar ELF layout.
_SUPPORTED_SERVER_REPAIRED_SHA256 = {
    "d405da04cc21fb7e99dddc88775eb5ccccf334e9f847bd7d8dd9b56a9e2e33a7": (
        "e6f84008279104cc2ff15a8288c9364cfa136217416c33ba63b471a7e727cb9c"
    ),
    "221c442018f7fb8d0ce7d4a786ee64fcc02ae0f0ff21d7915eee7ec8708e6eb7": (
        "5ece41543fac0f66940a22d8dd26ced81d8cbe9acb070d22f10c4e98569d1cb9"
    ),
}


class CloudXR64KCompatibilityError(RuntimeError):
    """Raised when a CloudXR binary cannot be repaired safely."""


@dataclass(frozen=True, slots=True)
class _ProgramHeader:
    """Fields needed to relocate one ELF64 program segment."""

    index: int
    table_offset: int
    type: int
    flags: int
    offset: int
    vaddr: int
    filesz: int
    memsz: int
    align: int


@dataclass(frozen=True, slots=True)
class _SectionHeader:
    """Fields needed to relocate one ELF64 section."""

    index: int
    table_offset: int
    type: int
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class _ElfLayout:
    """Validated ELF64 table layout."""

    phoff: int
    shoff: int
    phentsize: int
    shentsize: int
    programs: tuple[_ProgramHeader, ...]
    sections: tuple[_SectionHeader, ...]


def _require_span(data: bytes | bytearray, offset: int, size: int, label: str) -> None:
    """Require a file-backed ELF span to fit in *data*."""
    if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
        raise CloudXR64KCompatibilityError(
            f"CloudXR ELF has an out-of-range {label}: offset={offset:#x}, "
            f"size={size:#x}, file_size={len(data):#x}"
        )


def _read_elf_layout(data: bytes | bytearray) -> _ElfLayout:
    """Parse and bounds-check the ELF tables used by the repair."""
    _require_span(data, 0, _ELF64_HEADER.size, "header")
    (
        ident,
        elf_type,
        machine,
        _version,
        _entry,
        phoff,
        shoff,
        _flags,
        ehsize,
        phentsize,
        phnum,
        shentsize,
        shnum,
        _shstrndx,
    ) = _ELF64_HEADER.unpack_from(data)
    if ident[:4] != b"\x7fELF" or ident[4:7] != b"\x02\x01\x01":
        raise CloudXR64KCompatibilityError(
            "CloudXR server is not a little-endian ELF64 binary"
        )
    if elf_type != _DYNAMIC_OBJECT or machine != _AARCH64_MACHINE:
        raise CloudXR64KCompatibilityError(
            f"CloudXR server is not an ARM64 shared object "
            f"(type={elf_type}, machine={machine})"
        )
    if ehsize != _ELF64_HEADER.size:
        raise CloudXR64KCompatibilityError(f"unsupported ELF header size {ehsize}")
    if phentsize != _ELF64_PROGRAM_HEADER.size or phnum in (0, 0xFFFF):
        raise CloudXR64KCompatibilityError("unsupported ELF64 program-header table")
    if shentsize != _ELF64_SECTION_HEADER.size or shnum == 0:
        raise CloudXR64KCompatibilityError("unsupported ELF64 section-header table")
    _require_span(data, phoff, phentsize * phnum, "program-header table")
    _require_span(data, shoff, shentsize * shnum, "section-header table")

    programs: list[_ProgramHeader] = []
    for index in range(phnum):
        table_offset = phoff + index * phentsize
        (
            segment_type,
            segment_flags,
            offset,
            vaddr,
            _paddr,
            filesz,
            memsz,
            align,
        ) = _ELF64_PROGRAM_HEADER.unpack_from(data, table_offset)
        if filesz:
            _require_span(data, offset, filesz, f"program segment {index}")
        if segment_type == _PT_LOAD:
            if filesz > memsz:
                raise CloudXR64KCompatibilityError(
                    f"ELF load segment {index} has filesz greater than memsz"
                )
            if align not in (0, 1) and align & (align - 1):
                raise CloudXR64KCompatibilityError(
                    f"ELF load segment {index} has invalid alignment {align:#x}"
                )
            if align > 1 and (vaddr - offset) % align:
                raise CloudXR64KCompatibilityError(
                    f"ELF load segment {index} violates its declared alignment"
                )
        programs.append(
            _ProgramHeader(
                index=index,
                table_offset=table_offset,
                type=segment_type,
                flags=segment_flags,
                offset=offset,
                vaddr=vaddr,
                filesz=filesz,
                memsz=memsz,
                align=align,
            )
        )

    sections: list[_SectionHeader] = []
    for index in range(shnum):
        table_offset = shoff + index * shentsize
        (
            _name,
            section_type,
            _section_flags,
            _addr,
            offset,
            size,
            _link,
            _info,
            _addralign,
            _entsize,
        ) = _ELF64_SECTION_HEADER.unpack_from(data, table_offset)
        if size and section_type != _SHT_NOBITS:
            _require_span(data, offset, size, f"section {index}")
        sections.append(
            _SectionHeader(
                index=index,
                table_offset=table_offset,
                type=section_type,
                offset=offset,
                size=size,
            )
        )
    return _ElfLayout(
        phoff=phoff,
        shoff=shoff,
        phentsize=phentsize,
        shentsize=shentsize,
        programs=tuple(programs),
        sections=tuple(sections),
    )


def _misaligned_loads(layout: _ElfLayout, page_size: int) -> tuple[_ProgramHeader, ...]:
    """Return load segments invalid for *page_size*."""
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError("ELF page size must be a positive power of two")
    return tuple(
        segment
        for segment in layout.programs
        if segment.type == _PT_LOAD
        and segment.offset % page_size != segment.vaddr % page_size
    )


def _repair_elf_load_offsets(data: bytes, page_size: int) -> bytes:
    """Return a copy whose trailing load offsets are page-congruent.

    The affected CloudXR binaries need one padding insertion before a suffix of
    load segments.  Their virtual addresses and contents remain unchanged;
    only later file offsets and alignment declarations move.
    """
    layout = _read_elf_layout(data)
    bad = _misaligned_loads(layout, page_size)
    if not bad:
        return data

    insertion = min(segment.offset for segment in bad)
    affected_loads = tuple(
        segment
        for segment in layout.programs
        if segment.type == _PT_LOAD and segment.offset >= insertion
    )
    if set(affected_loads) != set(bad):
        raise CloudXR64KCompatibilityError(
            "CloudXR ELF does not have one repairable trailing load-segment suffix"
        )
    congruence_paddings = {
        (segment.vaddr - segment.offset) % page_size for segment in affected_loads
    }
    if len(congruence_paddings) != 1:
        raise CloudXR64KCompatibilityError(
            "CloudXR ELF load segments require incompatible padding"
        )
    congruence_padding = congruence_paddings.pop()
    if not 0 < congruence_padding < page_size:
        raise CloudXR64KCompatibilityError("CloudXR ELF padding is invalid")
    load_differences = {segment.vaddr - segment.offset for segment in affected_loads}
    if len(load_differences) != 1:
        raise CloudXR64KCompatibilityError(
            "CloudXR ELF trailing load segments do not share one file mapping"
        )

    # A 64 KiB loader maps from floor(p_offset, page_size), including bytes
    # before a segment's declared offset. Put the first shifted segment in a
    # later page so that this prefix comes only from newly inserted zeroes,
    # rather than from the preceding segment's file-backed data. This
    # preserves that preceding segment's BSS tail.
    first_affected = min(affected_loads, key=lambda segment: segment.offset)
    next_file_page = (insertion + page_size - 1) & -page_size
    new_first_offset = next_file_page + first_affected.vaddr % page_size
    padding = new_first_offset - insertion
    if padding % page_size != congruence_padding:
        raise CloudXR64KCompatibilityError(
            "CloudXR ELF safe padding does not preserve load congruence"
        )

    prefix_vaddr_start = first_affected.vaddr & -page_size
    prefix_vaddr_end = first_affected.vaddr
    for segment in layout.programs:
        if segment.type != _PT_LOAD or segment in affected_loads:
            continue
        file_vaddr_end = segment.vaddr + segment.filesz
        if segment.vaddr < prefix_vaddr_end and prefix_vaddr_start < file_vaddr_end:
            raise CloudXR64KCompatibilityError(
                "CloudXR ELF shifted page prefix overlaps file-backed data in "
                f"load segment {segment.index}"
            )
        memory_vaddr_end = segment.vaddr + segment.memsz
        if (
            segment.vaddr < prefix_vaddr_end
            and prefix_vaddr_start < memory_vaddr_end
            and segment.flags != first_affected.flags
        ):
            raise CloudXR64KCompatibilityError(
                "CloudXR ELF shifted page prefix changes permissions over BSS in "
                f"load segment {segment.index}"
            )

    program_table_end = layout.phoff + layout.phentsize * len(layout.programs)
    if insertion < program_table_end:
        raise CloudXR64KCompatibilityError(
            "CloudXR ELF requires moving its program-header table"
        )
    section_table_end = layout.shoff + layout.shentsize * len(layout.sections)
    if layout.shoff < insertion < section_table_end:
        raise CloudXR64KCompatibilityError(
            "repair boundary intersects the ELF section-header table"
        )
    for segment in layout.programs:
        if (
            segment.filesz
            and segment.offset < insertion < segment.offset + segment.filesz
        ):
            raise CloudXR64KCompatibilityError(
                f"repair boundary intersects program segment {segment.index}"
            )
    for section in layout.sections:
        if (
            section.type != _SHT_NOBITS
            and section.size
            and section.offset < insertion < section.offset + section.size
        ):
            raise CloudXR64KCompatibilityError(
                f"repair boundary intersects section {section.index}"
            )

    output = bytearray(data[:insertion])
    output.extend(b"\0" * padding)
    output.extend(data[insertion:])
    if layout.shoff >= insertion:
        struct.pack_into("<Q", output, 0x28, layout.shoff + padding)

    for segment in layout.programs:
        if segment.offset >= insertion:
            struct.pack_into(
                "<Q", output, segment.table_offset + 8, segment.offset + padding
            )
            if segment.type == _PT_LOAD:
                struct.pack_into(
                    "<Q",
                    output,
                    segment.table_offset + 48,
                    max(segment.align, page_size),
                )

    section_table_shift = padding if layout.shoff >= insertion else 0
    for section in layout.sections:
        if section.offset < insertion:
            continue
        new_table_offset = section.table_offset + section_table_shift
        struct.pack_into("<Q", output, new_table_offset + 24, section.offset + padding)

    repaired = bytes(output)
    repaired_layout = _read_elf_layout(repaired)
    remaining = _misaligned_loads(repaired_layout, page_size)
    if remaining:
        detail = ", ".join(str(segment.index) for segment in remaining)
        raise CloudXR64KCompatibilityError(
            f"CloudXR ELF repair left misaligned load segments: {detail}"
        )
    for segment in layout.programs:
        if not segment.filesz:
            continue
        segment_end = segment.offset + segment.filesz
        program_table_end = layout.phoff + layout.phentsize * len(layout.programs)
        if segment.offset < _ELF64_HEADER.size and segment_end > 0:
            # The ELF header contains the intentionally updated section-table
            # offset.
            continue
        if segment.offset < program_table_end and segment_end > layout.phoff:
            # Program headers contain the intentionally updated offsets and
            # alignment values.
            continue
        new_offset = segment.offset + (padding if segment.offset >= insertion else 0)
        if (
            repaired[new_offset : new_offset + segment.filesz]
            != data[segment.offset : segment.offset + segment.filesz]
        ):
            raise CloudXR64KCompatibilityError(
                f"CloudXR ELF repair changed program segment {segment.index}"
            )
    return repaired


def _atomic_write(path: Path, data: bytes, mode: int) -> None:
    """Atomically write one generated compatibility library."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cloudxr_64k_workaround_required(
    *, page_size: int | None = None, machine: str | None = None
) -> bool:
    """Report whether this host needs the guarded CloudXR workaround."""
    host_machine = platform.machine() if machine is None else machine
    if host_machine.lower() not in {"aarch64", "arm64"}:
        return False
    host_page_size = (
        int(os.sysconf("SC_PAGE_SIZE")) if page_size is None else int(page_size)
    )
    return host_page_size > 4096


def prepare_cloudxr_64k_preload(
    sdk_path: str | Path,
    *,
    page_size: int | None = None,
    machine: str | None = None,
) -> Path | None:
    """Create and return a guarded 64 KiB CloudXR preload when required.

    Returns ``None`` on platforms whose loader accepts the installed binary.
    The original ``libNvStreamServer.so`` is always left byte-for-byte intact.
    """
    if not cloudxr_64k_workaround_required(page_size=page_size, machine=machine):
        return None
    host_page_size = (
        int(os.sysconf("SC_PAGE_SIZE")) if page_size is None else int(page_size)
    )
    if host_page_size <= 4096:
        return None
    if host_page_size != _SUPPORTED_PAGE_SIZE:
        raise CloudXR64KCompatibilityError(
            f"unsupported ARM64 host page size {host_page_size}"
        )

    source = Path(sdk_path) / "libNvStreamServer.so"
    try:
        source_data = source.read_bytes()
    except OSError as error:
        raise CloudXR64KCompatibilityError(
            f"cannot read CloudXR server library at {source}: {error}"
        ) from error
    layout = _read_elf_layout(source_data)
    if not _misaligned_loads(layout, host_page_size):
        return None

    digest = hashlib.sha256(source_data).hexdigest()
    expected_repaired_digest = _SUPPORTED_SERVER_REPAIRED_SHA256.get(digest)
    if expected_repaired_digest is None:
        raise CloudXR64KCompatibilityError(
            "refusing to repair an unknown CloudXR ARM64 server binary "
            f"(sha256={digest})"
        )
    repaired = _repair_elf_load_offsets(source_data, host_page_size)
    repaired_digest = hashlib.sha256(repaired).hexdigest()
    if repaired_digest != expected_repaired_digest:
        raise CloudXR64KCompatibilityError(
            "CloudXR compatibility output failed its deterministic hash check "
            f"(sha256={repaired_digest})"
        )
    output = source.with_name(f"libNvStreamServer.flashdreams-64k-{digest[:12]}.so")
    if output.is_symlink():
        raise CloudXR64KCompatibilityError(
            f"refusing to replace compatibility-library symlink {output}"
        )
    try:
        if output.is_file() and output.read_bytes() == repaired:
            return output.absolute()
        _atomic_write(output, repaired, stat.S_IMODE(source.stat().st_mode))
    except OSError as error:
        raise CloudXR64KCompatibilityError(
            f"cannot create CloudXR compatibility library at {output}: {error}"
        ) from error
    return output.absolute()
