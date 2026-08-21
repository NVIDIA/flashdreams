# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One Omnidreams rollout: road layout in, driving video out."""

from typing import Any

from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents

from .conditioning import HDMapSource


class OmnidreamsSession(ISession):
    """One HDMap-conditioned rollout, continuing from a first frame.

    A step is one autoregressive block, conditioned on the chunk of HDMap
    pixels covering the frames it is about to generate. The source supplies
    that chunk, and how it came by it is its own business: a run replaying a
    recording and a run rendering what a driver steered differ only there. It
    is also what says how long the run is, since generating past the end of
    the road would have nothing to condition on.

    What belongs to a run is the cache this initializes, holding the encoded
    prompt and first frame alongside the attention state, and the source, which
    is positioned partway through a drive. The pipeline belongs to the
    application.
    """

    def __init__(
        self,
        pipeline: Any,
        prompt: str,
        source: HDMapSource,
        session_desc: SessionDesc,
        max_blocks: int | None = None,
    ) -> None:
        """
        Args:
            pipeline: Loaded pipeline, owned by the application.
            prompt: Text describing the drive, applied to every camera.
            source: Conditioning for this run, owned by this session.
            session_desc: Session the application accepted, already checked
                against what the model can produce.
            max_blocks: Blocks to stop after, for a run shorter than the
                conditioning. The default drives until the source runs out,
                which for a recording is its end and for a renderer is never.
        """
        self._pipeline = pipeline
        self._prompt = prompt
        self._source = source
        self._session_desc = session_desc
        self._max_blocks = max_blocks
        self._blocks_generated = 0
        self._cache: Any = None

    def init(self) -> None:
        """Open the conditioning, then encode the prompt and the first frame.

        Where the text encoder runs and the recording is decoded, so it is far
        slower than a step.
        """
        self._source.open()
        self._cache = self._new_cache()

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Generate the next block of frames from the next chunk of HDMap.

        Args:
            step_index: Also the autoregressive index the rollout is up to. The
                pipeline rejects a step out of order.
            events: Passed to the source, which is what turns them into
                geometry. A recording ignores them; a renderer steers by them.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._cache is None:
            raise RuntimeError(f"{type(self).__name__}.init() must run before step().")
        # The model decides the chunk length, and the conditioning has to cover
        # exactly it. A causal decoder's first block is usually shorter.
        frame_count = self._pipeline.get_num_frames(step_index)
        hdmap = self._source.next_chunk(frame_count, events)
        self._blocks_generated += 1
        frames = self._pipeline.generate(
            autoregressive_index=step_index, cache=self._cache, hdmap=hdmap
        )
        # Advancing the attention state is what makes the next step continue
        # this one, and it reports what the step cost.
        metrics = self._pipeline.finalize(
            autoregressive_index=step_index, cache=self._cache
        )
        return StepResult(
            step_index=step_index,
            output=frames.detach(),
            frame_count=int(frames.shape[2]),
            output_layout=self._session_desc.output_layout,
            metrics=dict(metrics or {}),
        )

    def is_finished(self) -> bool:
        """Report whether the rollout has run out of road, or of blocks.

        The road is the length of a run: a recording ends when it ends, and a
        renderer never does, so a run against one lasts until its client goes
        away. ``max_blocks`` stops a run before either, and is how a smoke run
        asks for a few seconds of a long drive.
        """
        if self._max_blocks is not None and self._blocks_generated >= self._max_blocks:
            return True
        return not self._source.has_frames(
            self._pipeline.get_num_frames(self._blocks_generated)
        )

    def reset(self) -> None:
        """Drive the same route again from the first frame.

        The cache is replaced rather than cleared, so nothing of the abandoned
        run reaches the new one, and the source rewinds with it.
        """
        self._blocks_generated = 0
        self._source.reset()
        self._cache = self._new_cache()

    def close(self) -> None:
        """Release the rollout's cache and its conditioning, leaving the model."""
        self._cache = None
        self._source.close()

    def _new_cache(self) -> Any:
        """Encode the prompt and first frame into a cache for one rollout.

        The one-shot encoders are left loaded, unlike a batch run that releases
        them to reclaim their memory: a reset comes back here, and an encoder
        released is a session that cannot start over.
        """
        return self._pipeline.initialize_cache(
            text=[[self._prompt] * len(self._source.view_names)],
            image=self._source.first_frame(),
            view_names=list(self._source.view_names),
        )
