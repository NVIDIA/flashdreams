# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled sample drives, fetched from Hugging Face when a run asks for one."""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from omnidreams.runner import (
    DEFAULT_EXAMPLE_DATA_UUID_1V,
    EXAMPLE_DATA_HF_BROWSER_URL,
    EXAMPLE_DATA_HF_REPO,
)

DEFAULT_HDMAP_SAMPLE = DEFAULT_EXAMPLE_DATA_UUID_1V
"""Recording a run gets when it asks to replay one without naming it."""

_RECORDING_SUFFIX = "_hdmap.mp4"
"""What the HDMap recording in a sample directory ends with."""

_FIRST_FRAME_NAME = "first_frame.png"
"""What the frame to continue from is called in a sample directory."""


def fetch_hdmap_sample(sample_id: str) -> tuple[Path, Path]:
    """Download one bundled single-camera HDMap recording.

    Args:
        sample_id: Directory under ``data/single_view`` in the samples dataset,
            which is a clip UUID.

    Returns:
        The HDMap recording, then the frame to continue from. A sample carries
        both, which is why replaying one takes no other arguments. They land in
        the Hugging Face cache, so asking again costs nothing.

    Raises:
        FileNotFoundError: The sample has no recording in it, or more than one,
            so which to drive through is not clear.
    """
    directory = f"data/single_view/{sample_id}"
    entries = HfApi().list_repo_tree(
        repo_id=EXAMPLE_DATA_HF_REPO,
        repo_type="dataset",
        path_in_repo=directory,
        recursive=False,
    )
    # The recording is named after the clip rather than predictably, so the
    # directory is listed rather than the filename built from the drive id.
    recordings = [
        entry.path
        for entry in entries
        if isinstance(entry, RepoFile) and entry.path.endswith(_RECORDING_SUFFIX)
    ]
    if len(recordings) != 1:
        found = ", ".join(recordings) if recordings else "none"
        raise FileNotFoundError(
            f"Expected one '*{_RECORDING_SUFFIX}' in {directory} of "
            f"{EXAMPLE_DATA_HF_REPO}, found {found}. Samples are listed at "
            f"{EXAMPLE_DATA_HF_BROWSER_URL}."
        )
    return (
        _download(recordings[0]),
        _download(f"{directory}/{_FIRST_FRAME_NAME}"),
    )


def _download(path_in_repo: str) -> Path:
    """Return one file of the samples dataset, from the cache or from the hub."""
    return Path(
        hf_hub_download(
            repo_id=EXAMPLE_DATA_HF_REPO,
            repo_type="dataset",
            filename=path_in_repo,
        )
    )
