"""Which contacts are neighbours on their device.

A montage drawn as bare markers says nothing about which contacts share a shank.
Joining them with a line does, and the only question is which pairs to join.

That is a geometric question, so it is answered here rather than in the browser:
once, in numpy, from the *measured* coordinates. Deriving it in JavaScript from
the drawn positions instead would make the topology depend on the inflation
slider -- edges appearing and disappearing mid-morph as a deforming surface
stretched some spacings past a threshold -- when the fact being drawn is a
property of the physical device and does not change at all.

Two shapes of device, because there are two ways contacts are laid out:

- a **linear** probe (a depth or sEEG shank, a subdural strip) is a sequence, so
  its contacts are chained in order along it;
- a **grid** is a lattice, so each contact joins the neighbours above, below and
  beside it, but not the ones diagonally across a square from it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ._set import infer_index

LINEAR_TYPES = ("depth", "seeg", "strip")
"""Devices whose contacts are a sequence rather than a lattice. Anything else --
including a ``group_type`` nobody recognises, since the field is free-form -- is
treated as a grid, whose neighbour test degrades to exactly this chain when the
contacts happen to be collinear."""

GRID_SLACK = 1.05
"""Slack in :func:`_neighbourhood`'s test, for exact ties and small jitter.

That test is an equality on a perfect lattice, so it has to be tolerant to
survive floating point and a fraction of a millimetre of measurement noise. The
number it has to stay well under is 1.41, the ratio between a lattice diagonal
and a lattice edge, so 5% is generous in one direction and nowhere near the
other."""


def group_edges(
    coords: npt.ArrayLike,
    group: Sequence[Any],
    group_type: Sequence[Any],
    names: Optional[Sequence[Any]] = None,
) -> list[tuple[int, int]]:
    """Pairs of contacts that are adjacent on the same device.

    Parameters
    ----------
    coords : (n, 3) array
        Contact positions. Non-finite rows take part in no edge -- a contact
        with no coordinate has nowhere to draw a line to.
    group : (n,) sequence
        The device each contact belongs to. Edges never cross a group.
    group_type : (n,) sequence
        The kind of device, read from the first member of each group. See
        :data:`LINEAR_TYPES`.
    names : (n,) sequence, optional
        Channel names. A linear device is ordered by the contact number in its
        names (:func:`~cortex.electrodes.infer_index`) so that the chain follows
        the shank rather than the array; without names, or where any member's
        name carries no number, array order is used instead.

    Returns
    -------
    list of (int, int)
        Index pairs into the arrays passed in, each with the lower index first.
    """
    coords = np.asarray(coords, dtype=float)
    group = np.asarray(group)
    group_type = np.asarray(group_type)
    names = None if names is None else np.asarray(names)

    usable = np.isfinite(coords).all(axis=1)
    edges: list[tuple[int, int]] = []

    for device in dict.fromkeys(group.tolist()):
        members = np.flatnonzero((group == device) & usable)
        if len(members) < 2:
            continue
        kind = str(group_type[members[0]]).lower()
        numbers = _contact_numbers(members, names)
        if kind in LINEAR_TYPES:
            edges.extend(_chain(members, numbers))
        else:
            edges.extend(_lattice(members, coords[members], numbers))

    return sorted(edges)


def _contact_numbers(
    members: npt.NDArray[np.integer], names: Optional[npt.NDArray[Any]]
) -> Optional[list[int]]:
    """The number each of a group's contacts carries in its name, or None.

    None as soon as any member's name has no number in it, or two share one:
    a numbering with a hole in it still describes the device, but one that is
    partly missing or ambiguous describes nothing, and half a numbering is worse
    than none because it looks usable.
    """
    if names is None:
        return None
    numbers = [infer_index(names[i]) for i in members]
    if any(number is None for number in numbers):
        return None
    if len(set(numbers)) != len(numbers):
        return None
    return [int(number) for number in numbers]


def _chain(
    members: npt.NDArray[np.integer], numbers: Optional[list[int]]
) -> list[tuple[int, int]]:
    """A linear device: each contact joined to the next one along the shank.

    Contact order, not array order: a shank's contacts are numbered from its
    tip, and a montage written out in some other order -- sorted by name as
    text, say, where ``D10`` lands between ``D1`` and ``D2`` -- would otherwise
    be chained into a zigzag up and down the probe.
    """
    order = ([int(i) for i in members] if numbers is None
             else [int(i) for _, i in sorted(zip(numbers, members))])
    return [_pair(order[i], order[i + 1]) for i in range(len(order) - 1)]


def _lattice(
    members: npt.NDArray[np.integer],
    points: npt.NDArray[np.floating],
    numbers: Optional[list[int]],
) -> list[tuple[int, int]]:
    """A grid: each contact joined to the ones beside and above it.

    A grid is a rigid rectangular sheet whose contacts are numbered along its
    rows, so its lattice is a fact about the numbering and only the row width is
    unknown -- no montage format the readers support records one. Recovering
    that width from the positions (:func:`_row_major`) is far steadier than
    recovering the whole lattice from them, because it asks the geometry a
    question with a handful of possible answers that are far apart, rather than
    asking it to adjudicate every pair.

    Falling back on the geometry alone (:func:`_neighbourhood`) when the
    numbering cannot be read at all -- a montage with an explicit group column
    and names that carry no contact number.
    """
    if numbers is not None:
        return _row_major(members, points, numbers)
    return _neighbourhood(members, points)


def _row_major(
    members: npt.NDArray[np.integer],
    points: npt.NDArray[np.floating],
    numbers: list[int],
) -> list[tuple[int, int]]:
    """The lattice implied by the contact numbers, at its likeliest row width.

    Two questions, asked separately because only the first needs the geometry
    to be well behaved:

    The **width** is read off the contacts numbered ``width`` apart, which are
    the ones directly above and below each other. Every candidate width is
    tried and the one whose pairs come out shortest wins, which is decisive
    because a wrong width steps sideways along a row instead of down a column
    and lands multiples of the pitch away. Only these pairs are scored, and
    they do not depend on where the rows are cut, which is what lets the width
    be settled before the phase.

    The **phase** -- where the row boundaries fall -- then follows from the
    consecutive pairs: at every boundary, consecutive numbers are at opposite
    ends of the grid rather than side by side, so the cut that leaves the
    remaining consecutive pairs shortest is the real one. It is a separate
    question from the width because a montage missing its first contact starts
    counting from the second, putting the boundaries one column out.
    """
    where = {number: int(i) for number, i in zip(numbers, members)}
    index = {number: row for row, number in enumerate(numbers)}
    span = max(numbers) - min(numbers) + 1

    def length(a: int, b: int) -> float:
        return float(np.linalg.norm(points[index[a]] - points[index[b]]))

    width, best = 1, np.inf
    for candidate in range(1, span):
        stacked = [length(n, n + candidate) for n in where if n + candidate in where]
        # One pair is a coincidence rather than evidence of a column.
        if len(stacked) < 2:
            continue
        score = float(np.mean(stacked))
        if score < best:
            width, best = candidate, score

    side_by_side = [(n, length(n, n + 1)) for n in where if n + 1 in where]
    origin = min(numbers)
    phase, best = 0, np.inf
    for candidate in range(width):
        kept = [d for n, d in side_by_side
                if (n - origin + candidate) % width != width - 1]
        if not kept:
            continue
        score = float(np.mean(kept))
        if score < best:
            phase, best = candidate, score

    edges = []
    for number, i in where.items():
        if number + width in where:
            edges.append(_pair(i, where[number + width]))
        if (number + 1 in where
                and (number - origin + phase) % width != width - 1):
            edges.append(_pair(i, where[number + 1]))
    return edges


def _neighbourhood(
    members: npt.NDArray[np.integer], points: npt.NDArray[np.floating]
) -> list[tuple[int, int]]:
    """A grid with no readable numbering: the relative neighbourhood graph.

    Two contacts are joined when no third contact is closer to both of them than
    they are to each other. On a lattice that is exactly the four-neighbour
    result wanted here: a contact beside another has no third contact nearer
    than the pitch, while the two ends of a diagonal both have the corner
    between them closer than they are to each other, so the diagonal goes.

    Stated as a comparison between distances rather than as a distance
    threshold, it carries no notion of scale, which matters because a grid
    conformed to a folded surface has no single pitch to threshold against: it
    is compressed inside a sulcus and stretched over a gyral crown. A threshold
    tuned to the median pitch tears holes in exactly the places the cortex bends
    most.

    Costs one pass over an (n, n) array per contact. A grid is a few hundred
    contacts at the very most, and the alternative -- the (n, n, n) array this
    is a loop over -- is what makes 256 contacts a problem rather than 130 MB.
    """
    count = len(points)
    separation = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    keep = np.zeros((count, count), dtype=bool)
    every = np.arange(count)

    for i in range(count):
        # lune[j, k] is how far the further of contacts i and j is from k; the
        # smallest of those, over the contacts k that are neither i nor j, is
        # the closest any third contact comes to both.
        lune = np.maximum(separation[i][None, :], separation)
        lune[:, i] = np.inf
        lune[every, every] = np.inf
        keep[i] = separation[i] <= GRID_SLACK * lune.min(axis=1)

    np.fill_diagonal(keep, False)
    return [_pair(members[a], members[b]) for a, b in zip(*np.nonzero(np.triu(keep, 1)))]


def _pair(a: Any, b: Any) -> tuple[int, int]:
    """One edge, lower index first, so pairs compare and deduplicate."""
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)
