# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How closely one drawn control frame agrees with the reference's own.

A control frame is line art on black: a few percent of it is ink and the rest
is nothing. That shapes every number here.

Subtracting one frame from the other is the obvious measure and the least
useful one. The two renderers put a line in slightly different places, so a
difference image is dominated by which pixel a line fell in and reads as
disagreement however well they agree about the road. Worse, it is dominated by
the parts of the frame with the most ink, which is the near foreground, while
what matters for fidelity is usually further off.

So ink is counted instead: which pixels either renderer drew anything at, and
whether the other drew the same paint there too. Allowing a one-pixel shift
then separates the two failures that matter -- drawing the wrong thing, and
drawing the right thing a pixel over -- and only the first is a fidelity
problem.

The shift is allowed on where a line is and never on what colour it is, or the
two excuses cover for each other and a line that moves a pixel while changing
paint reads as no error at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DARK = 24
"""Below this the brightest channel is background, however non-zero it reads.

The reference is four fifths black, and h264 leaves that black as a drift of
ones and twos across twenty thousand distinct values. Counting those as ink put
most of a frame in the unrecognised column and read as a whole missing layer.
"""

NEAR = 48
"""How far a pixel may sit from a palette colour and still be counted as it.

The reference arrives as h264, which smears an edge across several shades and
pulls flat regions off their nominal value, and ludus shades a line toward
black along its length, so an exact match counts almost nothing. Only the
question of whether a pixel is ink at all rests on this, since each one goes to
its nearest colour regardless. Colours nearer each other than this cannot be
told apart, and callers fold them together rather than guessing.
"""

TINT = 0.25
"""How far two inked pixels' colours may sit apart and still be the same paint.

Compared as a tint, each channel against the brightest, so that a line dimmed
along its length or by compression still counts as its own colour.
"""


Palette = list[tuple[str, np.ndarray]]
"""Colours to account a frame's ink to, each named. Used by offline map comparisons."""


@dataclass(frozen=True)
class Agreement:
    """What two drawings of one frame do and do not share.

    Every field is a fraction of the whole frame, so they can be averaged over
    a clip and compared across cameras of different sizes.
    """

    ink: float
    """How much of the frame we inked."""

    their_ink: float
    """How much of it the reference inked."""

    unmatched: float
    """Inked by one and not the other, wherever it fell."""

    misplaced: float
    """Inked by one and not the other, allowing a one-pixel shift.

    This is the number to watch. What it leaves out is a line drawn a pixel
    from where the reference drew it, which no downstream model will notice;
    what it keeps is a shape one of them draws and the other has never heard
    of, which is a missing layer or the wrong rule.
    """

    ours_misplaced: float
    """The part of :attr:`misplaced` that we drew and the reference did not."""

    theirs_misplaced: float
    """The part of :attr:`misplaced` that the reference drew and we did not.

    Worth more than our own surplus: it is the reference asking for something
    we never draw, which is the direction a missing layer shows up in.
    """

    recoloured: float
    """Inked by both within a pixel, in colours that are not each other's.

    As a fraction of the whole frame, not of the shared ink, so that it can sit
    beside the rest. A layer drawn in the wrong paint lands here rather than in
    :attr:`misplaced`, which is why both are reported. The same one-pixel
    licence applies as there, and for the same reason -- a line a pixel over is
    the right line -- but it is spent only on where the paint is, never on
    which paint it is, so a layer both moved and repainted still lands here.
    """

    @property
    def wrong(self) -> float:
        """One number for a frame: ink in the wrong place or the wrong colour."""
        return self.misplaced + self.recoloured


def inked(frame: np.ndarray) -> np.ndarray:
    """Which pixels of a frame have been drawn on at all."""
    return frame.max(axis=2) > DARK


def agreement(ours: np.ndarray, theirs: np.ndarray) -> Agreement:
    """Hold two ``H x W x 3`` uint8 frames of the same view up against each other."""
    if ours.shape != theirs.shape:
        raise ValueError(f"{ours.shape} and {theirs.shape} are not the same view")

    mine, yours = inked(ours), inked(theirs)
    apart = _disagreement(ours, theirs, mine, yours)
    return Agreement(
        ink=float(mine.mean()),
        their_ink=float(yours.mean()),
        unmatched=float((mine ^ yours).mean()),
        misplaced=float((apart.ours_only | apart.theirs_only).mean()),
        ours_misplaced=float(apart.ours_only.mean()),
        theirs_misplaced=float(apart.theirs_only.mean()),
        recoloured=float(apart.repainted.mean()),
    )


def coverage(frame: np.ndarray, palette: Palette) -> tuple[list[float], float]:
    """What fraction of a frame each colour covers, and what fraction no colour does.

    Every inked pixel goes to its nearest colour and is counted once, so the
    shares can be read against the reference's own colour by colour. Which is
    the point of doing it: told that two frames disagree over half a percent
    of their ink, there is nothing to do about it, whereas told the surplus is
    all in one paint, there is a layer to go and look at.

    The leftover is ink no colour claims, which on the reference's side means
    a paint we do not know about and on ours means one we should not have.

    Only the ink is measured against the palette, the background being most of
    a frame and none of the question, which is what keeps a sweep of a hundred
    frames against thirty colours to seconds rather than minutes.
    """
    pixels = frame.reshape(-1, 3)
    whole = pixels.shape[0]
    standing = pixels[pixels.max(axis=1) > DARK].astype(np.int16)
    if not standing.size:
        return [0.0] * len(palette), 0.0

    apart = np.stack([np.abs(standing - colour).max(axis=1) for _, colour in palette])
    nearest = apart.argmin(axis=0)
    claimed = apart.min(axis=0) <= NEAR

    share = [
        float(((nearest == index) & claimed).sum()) / whole
        for index in range(len(palette))
    ]
    return share, float((~claimed).sum()) / whole


def repainting(
    ours: np.ndarray, theirs: np.ndarray, palette: Palette
) -> dict[tuple[str, str], float]:
    """Which paints stand where the two inked the same place in different colours.

    :attr:`Agreement.recoloured` counts those pixels; this says what they are,
    as fractions of the whole frame keyed by the pair of colours. A layer in
    the wrong paint reads as one pair carrying nearly all of it. Compression
    along an edge reads as a scatter of pairs of neighbouring colours, each
    too small to be a layer.

    Where only one of them inked the pixel itself, the other is named by the
    paint it put nearest, that being the ink which stood in for this one.
    """
    mine, yours = inked(ours), inked(theirs)
    apart = _disagreement(ours, theirs, mine, yours)
    wrong = apart.repainted
    if not wrong.any():
        return {}

    ours_at = np.where(mine[..., None], ours, _fetched(ours, apart.their_nearest))
    theirs_at = np.where(yours[..., None], theirs, _fetched(theirs, apart.my_nearest))
    pairs, counts = np.unique(
        np.stack(
            [
                _painted_as(ours_at[wrong], palette),
                _painted_as(theirs_at[wrong], palette),
            ],
            axis=1,
        ),
        axis=0,
        return_counts=True,
    )
    whole = ours.shape[0] * ours.shape[1]
    names = [name for name, _ in palette]
    return {
        (names[mine], names[yours]): float(count) / whole
        for (mine, yours), count in zip(pairs.tolist(), counts.tolist())
    }


def _painted_as(pixels: np.ndarray, palette: Palette) -> np.ndarray:
    """Which palette colour each of an ``N x 3`` run of pixels reads as.

    By tint rather than by value, so that a line shaded towards black along
    its length is still counted as the paint it was drawn in.
    """
    tint = _tint(pixels)
    against = np.stack(
        [np.asarray(colour, np.float32) / max(max(colour), 1) for _, colour in palette]
    )
    return np.stack([np.abs(tint - one).max(axis=1) for one in against]).argmin(axis=0)


def _tint(frame: np.ndarray) -> np.ndarray:
    """Colours with their brightness divided out, a frame's or a run of pixels'."""
    lit = frame.astype(np.float32)
    return lit / np.maximum(lit.max(axis=-1, keepdims=True), 1.0)


_AROUND = [(down, across) for down in range(3) for across in range(3)]
"""The nine places a pixel may have moved to and still count as where it was."""


@dataclass(frozen=True, eq=False)
class _Disagreement:
    """Masks of the ink that went unanswered, and where each pixel looked."""

    ours_only: np.ndarray
    theirs_only: np.ndarray
    repainted: np.ndarray
    my_nearest: np.ndarray
    their_nearest: np.ndarray


def _disagreement(
    ours: np.ndarray, theirs: np.ndarray, mine: np.ndarray, yours: np.ndarray
) -> _Disagreement:
    """Ink one of them drew that the other did not answer, and how it went unanswered.

    Position and colour are settled together, in one look at the nine places a
    pixel may have moved to. Settling them apart is what lets a line that moved
    a pixel *and* changed paint escape both: it is near enough to their ink to
    pass the position test, and their pixel of that colour is not the one under
    it, so the colour test never sees it.

    So each inked pixel asks for the nearest thing the other drew within a
    pixel of it. Nothing at all there makes it misplaced. Something there in a
    paint that is not its own makes it repainted. Both are counted from either
    side, as a shape drawn by one and not the other has to be.
    """
    my_gap, my_where = _nearest(ours, theirs, mine, yours)
    their_gap, their_where = _nearest(theirs, ours, yours, mine)
    answered, they_answered = my_where >= 0, their_where >= 0
    return _Disagreement(
        ours_only=mine & ~answered,
        theirs_only=yours & ~they_answered,
        repainted=(mine & answered & (my_gap > TINT))
        | (yours & they_answered & (their_gap > TINT)),
        my_nearest=my_where,
        their_nearest=their_where,
    )


def _nearest(
    ours: np.ndarray, theirs: np.ndarray, mine: np.ndarray, yours: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each pixel we inked, the nearest in tint of their ink within a pixel of it.

    How far away in tint that is, infinitely where they inked none of those
    nine places, and which place it was found in so :func:`_fetched` can go and
    get it.

    Only our own ink is asked, and only the nine pixels around each of those
    are read, which is what keeps this to a few percent of a frame's work: the
    rest is black on both sides and has no question to answer.
    """
    shape = yours.shape
    apart = np.full(shape[0] * shape[1], np.inf, np.float32)
    where = np.full(shape[0] * shape[1], -1, np.int8)

    at = np.flatnonzero(mine)
    if at.size:
        tinted = _tint(ours.reshape(-1, 3)[at])
        theirs_pixels = np.pad(theirs, ((1, 1), (1, 1), (0, 0))).reshape(-1, 3)
        theirs_inked = np.pad(yours, 1).reshape(-1)
        for index, (down, across) in enumerate(_AROUND):
            nearby = _around(at, shape, down, across)
            gap = np.abs(tinted - _tint(theirs_pixels[nearby])).max(axis=1)
            closer = theirs_inked[nearby] & (gap < apart[at])
            apart[at[closer]] = gap[closer]
            where[at[closer]] = index
    return apart.reshape(shape), where.reshape(shape)


def _fetched(frame: np.ndarray, where: np.ndarray) -> np.ndarray:
    """The pixels :func:`_nearest` picked out, black wherever it picked none."""
    shape = where.shape
    padded = np.pad(frame, ((1, 1), (1, 1), (0, 0))).reshape(-1, 3)
    picked = np.zeros((shape[0] * shape[1], 3), frame.dtype)
    found = where.reshape(-1)
    for index, (down, across) in enumerate(_AROUND):
        at = np.flatnonzero(found == index)
        picked[at] = padded[_around(at, shape, down, across)]
    return picked.reshape(frame.shape)


def _around(
    at: np.ndarray, shape: tuple[int, int], down: int, across: int
) -> np.ndarray:
    """Where a run of a frame's positions lands in the same frame padded by a pixel."""
    width = shape[1]
    return (at // width + down) * (width + 2) + (at % width + across)
