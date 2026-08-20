# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One text-to-video rollout: a prompt in, a chunk of frames per step out."""

from typing import Any

from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


class T2VSession(ISession):
    """One rollout of a streaming text-to-video model.

    A step is one autoregressive block, and each one continues the block before
    it rather than starting again. A causal decoder's first block usually decodes
    fewer frames than the rest, because its first latent frame covers one frame
    rather than a chunk of them.

    A rollout is ``total_blocks`` long, which is why the run ends without anyone
    counting for it: how much video a prompt turns into is the model's business.

    The pipeline belongs to the application and is shared with every other
    session. What belongs to a run is the cache this initializes, which holds
    the encoded prompt and the attention state the rollout builds up.
    """

    def __init__(
        self,
        pipeline: Any,
        prompt: str,
        session_desc: SessionDesc,
        total_blocks: int,
    ) -> None:
        """
        Args:
            pipeline: Loaded pipeline, owned by the application.
            prompt: Text to generate from.
            session_desc: Session the application accepted. Already checked
                against what the model can produce.
            total_blocks: Blocks this rollout generates before it is finished.
        """
        self._pipeline = pipeline
        self._prompt = prompt
        self._session_desc = session_desc
        self._total_blocks = total_blocks
        self._blocks_generated = 0
        self._cache: Any = None

    def init(self) -> None:
        """Encode the prompt and prepare the rollout's cache.

        This is where the text encoder runs, so it is slower than a step and
        happens once.
        """
        self._cache = self._new_cache()

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Generate the next block of frames.

        Args:
            step_index: Zero-based index of this step, which is also the
                autoregressive index the rollout is up to. The pipeline rejects
                a step out of order.
            events: Ignored. This kind of model takes its prompt at the start of
                a run and nothing after it.

        Returns:
            Result carrying the frames the block decoded, in the layout the
            session was created for, and whatever the pipeline measured while
            generating them.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._cache is None:
            raise RuntimeError(f"{type(self).__name__}.init() must run before step().")
        self._blocks_generated += 1
        frames = self._pipeline.generate(
            autoregressive_index=step_index, cache=self._cache
        )
        # Advancing the attention state is what makes the next step continue
        # this one, and it reports what the step cost when the pipeline is
        # configured to measure it.
        metrics = self._pipeline.finalize(
            autoregressive_index=step_index, cache=self._cache
        )
        return StepResult(
            step_index=step_index,
            output=frames.detach(),
            frame_count=int(frames.shape[0]),
            output_layout=self._session_desc.output_layout,
            metrics=dict(metrics or {}),
        )

    def is_finished(self) -> bool:
        """Report whether the whole rollout has been generated.

        Returns:
            Whether ``total_blocks`` blocks have been generated.
        """
        return self._blocks_generated >= self._total_blocks

    def reset(self) -> None:
        """Start the rollout again from the same prompt.

        The cache is replaced rather than cleared, so nothing of the abandoned
        run reaches the new one, and the rollout has its whole length again.
        """
        self._blocks_generated = 0
        self._cache = self._new_cache()

    def close(self) -> None:
        """Release the rollout's cache, leaving the loaded model alone."""
        self._cache = None

    def _new_cache(self) -> Any:
        """Encode the prompt into a cache for one rollout."""
        ratio = self._pipeline.decoder.spatial_compression_ratio
        return self._pipeline.initialize_cache(
            text=[self._prompt],
            image=None,
            height=self._session_desc.video_height // ratio,
            width=self._session_desc.video_width // ratio,
        )
