"""Choosing a device's anchors together instead of one contact at a time.

Anchoring picks, for each contact independently, the cortical column nearest it.
For a grid that is right. For a depth electrode it is what tears the montage
apart: a shaft threads folds, consecutive contacts sit near *opposite banks* of
the same sulcus, and each one independently picks whichever bank happens to be
half a millimetre closer. The choice looks harmless in native space -- both banks
really are adjacent -- and inflation then pulls them centimetres apart.

Measured on ``S0033_complete``, over its thirteen depth electrodes: the gap
between consecutive contacts' anchors on the inflated surface, divided by their
native spacing, is 0.89 at the median -- and **14.7% of consecutive pairs exceed
3x**, reaching 13x. That is the tearing, and it is entirely a consequence of
*which column was chosen*, not of where the contact was drawn.

What makes it fixable is that the choice is a near-tie. Among the fifty nearest
candidate columns for a real contact:

===================================  =========================
distance to the best column          p50 0.40 mm, p90 2.45 mm
extra cost of the fiftieth-best      p50 1.34 mm, p90 1.79 mm
===================================  =========================

So a contact can be reassigned among fifty columns for about a millimetre of
native-space fidelity, while that reassignment moves it by up to 77 mm on the
inflated surface. Tiny cost, enormous effect -- exactly the structure in which
choosing jointly beats choosing greedily.

The joint choice is a chain, so it is solved exactly by dynamic programming:

    total = sum_i unary(i, a_i)  +  lambda * sum_i pairwise(a_i, a_{i+1})

with ``unary`` the distance from contact *i* to column ``a_i`` -- the fidelity
term, in millimetres -- and ``pairwise`` the mismatch between the inflated gap
and the native spacing, also in millimetres. Both terms carry the same units, so
``lambda`` reads as "millimetres of fidelity I will pay to remove a millimetre of
spacing error".

Two properties are worth stating because they are what a continuous optimiser
cannot offer:

- **The fidelity cap is hard.** A candidate more than
  ``max_fidelity_loss_mm`` worse than the best available is not in the state
  space at all, so no weighting, no pathological device and no choice of lambda
  can move a contact off cortex it genuinely touches. Anatomical labelling is
  preserved by construction rather than optimised for.
- **The solution is exact.** A chain DP has no local minima, needs no
  initialisation and no gradient, and returns the same answer every time.

What it does not do -- and cannot -- is remove the tearing where the anatomy
genuinely separates. Where two consecutive contacts really do belong to gyri that
inflation moves apart, the cap forbids the compromise and the spacing breaks. The
gain is that the break lands at the one real crossing instead of being scattered
along the shaft.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Optional, Sequence

import numpy as np
import numpy.typing as npt

from ._anchor import (
    HEMIS,
    SurfacePair,
    _closest_point_weights,
    _triangle_basis,
    _vertex_faces,
    column_costs,
    frame_components,
)

MAX_FIDELITY_LOSS_MM = 2.0
"""How much further than the best available column a contact may be anchored.

A hard bound, not a penalty weight: candidates outside it are removed from the
state space, so the coherence term can never trade away anatomy however strongly
it is weighted. Two millimetres because that is where the measured candidate set
already lives -- the fiftieth-best column costs 1.34 mm at the median and 1.79 mm
at p90 -- so the cap admits essentially all the freedom that exists while making
"the contact left its own gyrus" unrepresentable.
"""

COHERENCE_WEIGHT = 4.0
"""Millimetres of fidelity paid per millimetre of spacing error removed.

Both terms are distances in millimetres, so this is a ratio rather than a scale
factor. Above one because ``max_fidelity_loss_mm`` already guarantees the thing
that must not be traded away: no candidate outside the cap is in the state space,
so raising this cannot move a contact off its own cortex however hard it pulls.
Within that bound the remaining freedom should go to spacing, which is the part
the caller has no other way to control.

Measured over S0033's thirteen devices at a 2 mm cap, median spacing error:
1.0 gives 0.28 mm, 4.0 gives 0.13 mm, and the worst fidelity loss is unchanged
at 1.9 mm. Lowering it towards zero recovers independent per-contact anchoring.
"""

MAX_ANCHOR_SHIFT_MM = 3.0
"""How far the anchor point itself may move, on the anatomical surface, from the
column ordinary anchoring would have chosen.

The second and stricter of the two caps, and the one that actually implements
"do not move ROIs". :data:`MAX_FIDELITY_LOSS_MM` bounds how much *further from
the contact* a column may be, which turns out not to bound how far away it is
**on the sheet**: two columns can each sit 2 mm from a contact and 9 mm from each
other, on opposite banks or at different depths of the same sulcus. Measured on
S0033 with only the fidelity cap in force, anchors moved up to 9.5 mm and 4.7% of
them moved more than 5 mm -- far enough to cross a gyral crown, which is the one
outcome ruled out.

Three millimetres because a gyral crown is a few millimetres across, so this
keeps a contact on the fold it started on while still allowing the choice that
matters: which of two facing banks a contact between them is assigned to.
"""

CANDIDATES = 400
"""Ceiling on columns considered per contact. Rarely binding -- the fidelity cap
removes far more than this does -- and present so a pathological neighbourhood
cannot blow up the transition table."""

NEIGHBOURS = 48
"""Mid-surface vertices seeding the candidate faces.

Much larger than anchoring's default of eight, because this is choosing *among*
candidates rather than taking the nearest, and a set that reaches only one bank
of a sulcus cannot express the choice at all. Measured over S0033: 24 gives a
median spacing error of 0.39 mm, 48 gives 0.28, and 96 gives 0.26 while making
the tear count slightly worse. The whole montage solves in under two seconds at
48, so the cost is not what decides it -- 48 is simply where the benefit stops.
"""


class DeviceAnchors(NamedTuple):
    """What the DP decided for one device."""

    index: npt.NDArray[np.intp]        #: rows of the electrode set, in order
    verts: npt.NDArray[np.intp]        #: (m, 3) chosen triangle per contact
    weights: npt.NDArray[np.floating]  #: (m, 3) barycentric weights
    fidelity_mm: npt.NDArray[np.floating]   #: distance to the chosen column
    fidelity_loss_mm: npt.NDArray[np.floating]  #: how much worse than the best
    switched: npt.NDArray[np.bool_]    #: where the track breaks anyway
    # Recomputed against the chosen column, not carried over. Depth, thickness
    # and the perpendicular offset are all statements *about a column*, so
    # changing which column a contact is anchored to invalidates every one of
    # them -- and a stale depth would be read by the placement policy and the
    # viewer's depth window without anything saying it no longer matched.
    depth: npt.NDArray[np.floating]
    thickness_mm: npt.NDArray[np.floating]
    offset_mm: npt.NDArray[np.floating]


def _candidates(
    point: npt.NDArray[np.floating],
    pair: SurfacePair,
    tree,
    indptr: npt.NDArray[np.intp],
    indices: npt.NDArray[np.intp],
    mid: npt.NDArray[np.floating],
    neighbours: int,
    keep: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """The ``keep`` best candidate columns for one contact.

    Ranked by :func:`~cortex.electrodes._anchor.column_costs`, the same measure
    ordinary anchoring selects its single winner by -- so the best candidate here
    is exactly the anchor that would have been chosen independently, and the DP
    is choosing among genuine alternatives to it rather than against a different
    criterion.
    """
    _, near = tree.query(point, k=min(neighbours, len(mid)))
    near = np.atleast_1d(near)
    cand = np.unique(
        np.concatenate([indices[indptr[v]:indptr[v + 1]] for v in near])
    )
    tri = np.asarray(pair.polys)[cand]
    w = _closest_point_weights(mid[tri], point)
    if pair.wm is None:
        foot = np.einsum("fij,fi->fj", mid[tri], w)
        cost = np.linalg.norm(foot - point, axis=1)
    else:
        cost = column_costs(point, tri, pair.pia, pair.wm, w)[0]

    order = np.argsort(cost, kind="stable")[:keep]
    return tri[order], w[order], cost[order]


def _local_scale(
    verts: npt.NDArray[np.integer],
    pia: npt.NDArray[np.floating],
    inflated: npt.NDArray[np.floating],
) -> float:
    """Median linear scale from pia to inflated over these triangles.

    A device-level aggregate: individual triangles disagree wildly -- the ratio
    runs 0.44 to 1.92 across a hemisphere -- but their median over a device is
    steady, and it is what makes a native spacing comparable to an inflated one.
    """
    src = _triangle_basis(pia[verts])[3]
    dst = _triangle_basis(inflated[verts])[3]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = dst / src
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    return float(np.median(ratio)) if len(ratio) else 1.0


def solve_device(
    coords: npt.NDArray[np.floating],
    pair: SurfacePair,
    inflated: npt.NDArray[np.floating],
    incumbent: Optional[tuple] = None,
    max_fidelity_loss_mm: float = MAX_FIDELITY_LOSS_MM,
    max_anchor_shift_mm: float = MAX_ANCHOR_SHIFT_MM,
    coherence_weight: float = COHERENCE_WEIGHT,
    candidates: int = CANDIDATES,
    neighbours: int = NEIGHBOURS,
) -> DeviceAnchors:
    """Choose anchors for one device's contacts, in order, jointly.

    Parameters
    ----------
    coords : (m, 3) array
        The device's contacts **in order along the shank**, in the same frame as
        the surfaces. Order matters: the chain is what makes this a DP, and a
        shuffled device would have its spacing term compare unrelated pairs.
    pair : SurfacePair
        The hemisphere the device is in.
    inflated : (nverts, 3) array
        The inflated surface, un-nudged. Needed because the coherence term is a
        statement about what inflation does to a pair of anchors, which cannot be
        seen on the anatomical surface -- two anchors a millimetre apart there
        may be forty apart here.

    Returns
    -------
    DeviceAnchors
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    m = len(coords)
    pia = np.asarray(pair.pia, dtype=np.float64)
    polys = np.asarray(pair.polys)
    wm_or_none = None if pair.wm is None else np.asarray(pair.wm, np.float64)
    mid = pia if wm_or_none is None else (pia + wm_or_none) / 2.0

    from scipy.spatial import cKDTree

    tree = cKDTree(mid)
    indptr, indices = _vertex_faces(polys, len(mid))

    tri, weights, cost = [], [], []
    for i in range(m):
        t, w, c = _candidates(
            coords[i], pair, tree, indptr, indices, mid, neighbours, candidates
        )
        if incumbent is not None:
            # The anchor ordinary anchoring chose, injected as a candidate in
            # its own right. Prepending it rather than trusting it to appear is
            # what makes the shift cap a guarantee: every other candidate is
            # bounded relative to this one, and this one is always reachable, so
            # the DP can never be forced onto something outside the bound. The
            # wider neighbourhood used here does not always contain it -- it
            # searches from the contact, not from the previous answer.
            itri, iw = incumbent[0][i][None, :], incumbent[1][i][None, :]
            icost = column_costs(coords[i], itri, pia, wm_or_none, iw)[0] \
                if pair.wm is not None else np.linalg.norm(
                    np.einsum("fij,fi->fj", mid[itri], iw) - coords[i], axis=1)
            same = (t == itri).all(axis=1)
            t = np.vstack([itri, t[~same]])
            w = np.vstack([iw, w[~same]])
            c = np.concatenate([icost, c[~same]])
        # Both caps, applied here so the excluded columns never enter the state
        # space and no weighting can reach them. The first bounds how much
        # further from the contact a column may be; the second bounds how far
        # the anchor point may travel *across the sheet*, which is what keeps a
        # contact on its own gyrus and which the first does not imply.
        here = np.einsum("fij,fi->fj", pia[t], w)
        # Measured from the anchor ordinary anchoring actually chose, not from
        # this function's own best candidate. They are not the same: the wider
        # neighbourhood used here sometimes finds a genuinely closer column, and
        # a cap measured from *that* would permit a contact to drift several
        # millimetres from where it would otherwise have been drawn while
        # reporting that it had not moved at all. The guarantee has to be
        # relative to the answer it is replacing.
        # Candidate 0 *is* the incumbent when one was given, so this measures
        # displacement from the answer being replaced.
        shift = np.linalg.norm(here - here[0], axis=1)
        allowed = (c <= c[0] + max_fidelity_loss_mm) & (shift <= max_anchor_shift_mm)
        # Always keep the independent choice, so a contact with no admissible
        # alternative still anchors exactly where it would have anyway.
        allowed[0] = True
        tri.append(t[allowed])
        weights.append(w[allowed])
        cost.append(c[allowed])

    scale = _local_scale(
        np.concatenate([t[:1] for t in tri]), pia, np.asarray(inflated, np.float64)
    )
    pitch = np.linalg.norm(np.diff(coords, axis=0), axis=1) * scale

    # Where each candidate would land on the inflated surface. Precomputed
    # because the transition table needs every pair of consecutive candidates.
    place = [
        np.einsum("fij,fi->fj", np.asarray(inflated, np.float64)[t], w)
        for t, w in zip(tri, weights)
    ]

    # Viterbi. `total[k]` is the least cost of any path ending in candidate k of
    # the current contact; `back` remembers how each was reached.
    total = np.asarray(cost[0], dtype=np.float64)
    back: list[npt.NDArray[np.intp]] = []
    for i in range(1, m):
        gap = np.linalg.norm(place[i - 1][:, None, :] - place[i][None, :, :], axis=2)
        step = total[:, None] + coherence_weight * np.abs(gap - pitch[i - 1])
        choice = np.argmin(step, axis=0)
        back.append(choice.astype(np.intp))
        total = step[choice, np.arange(step.shape[1])] + cost[i]

    chosen = np.empty(m, dtype=np.intp)
    chosen[-1] = int(np.argmin(total))
    for i in range(m - 1, 0, -1):
        chosen[i - 1] = back[i - 1][chosen[i]]

    picked_tri = np.array([tri[i][chosen[i]] for i in range(m)], dtype=np.intp)
    picked_w = np.array([weights[i][chosen[i]] for i in range(m)])
    fidelity = np.array([cost[i][chosen[i]] for i in range(m)])
    loss = np.array([cost[i][chosen[i]] - cost[i][0] for i in range(m)])

    depth = np.full(m, np.nan)
    thickness = np.full(m, np.nan)
    offset = np.array(fidelity, dtype=np.float64)
    if pair.wm is not None:
        wm = np.asarray(pair.wm, dtype=np.float64)
        for i in range(m):
            one = picked_tri[i][None, :]
            w1 = picked_w[i][None, :]
            _, t, length2, rel, column = column_costs(coords[i], one, pia, wm, w1)
            depth[i] = t[0] if length2[0] > 0 else np.nan
            thickness[i] = np.sqrt(length2[0])
            offset[i] = np.linalg.norm(rel[0] - t[0] * column[0])

    final = np.array([place[i][chosen[i]] for i in range(m)])
    realised = np.linalg.norm(np.diff(final, axis=0), axis=1)
    # A break the cap forced: the spacing is still badly wrong after the DP has
    # done what it can, which means the anatomy genuinely separates here.
    switched = np.zeros(m, dtype=bool)
    if m > 1:
        switched[1:] = realised > 3.0 * np.maximum(pitch, 1e-6)

    return DeviceAnchors(
        index=np.arange(m, dtype=np.intp),
        verts=picked_tri,
        weights=picked_w,
        fidelity_mm=fidelity,
        fidelity_loss_mm=loss,
        switched=switched,
        depth=depth,
        thickness_mm=thickness,
        offset_mm=offset,
    )


def coherent_anchors(
    anchors,
    coords: npt.NDArray[np.floating],
    hemis: Mapping[str, SurfacePair],
    inflated: Mapping[str, npt.NDArray[np.floating]],
    groups: Optional[Sequence[str]] = None,
    group_types: Optional[Sequence[str]] = None,
    order: Optional[Sequence[Sequence[int]]] = None,
    max_fidelity_loss_mm: float = MAX_FIDELITY_LOSS_MM,
    max_anchor_shift_mm: float = MAX_ANCHOR_SHIFT_MM,
    coherence_weight: float = COHERENCE_WEIGHT,
    neighbours: int = NEIGHBOURS,
    shared_frame_types: Sequence[str] = ("seeg", "depth"),
):
    """Re-anchor a montage's linear devices jointly, leaving everything else alone.

    Only devices that are rigid shanks are touched: a grid's contacts genuinely
    do each belong to the column nearest them, and re-choosing their anchors
    would be solving a problem they do not have. Which devices those are is
    decided the same way :func:`~cortex.electrodes.regroup_anchors` decides it --
    an explicit ``seeg``/``depth`` ``group_type``, or, where the montage records
    none, :func:`~cortex.electrodes.is_straight`.

    Parameters
    ----------
    anchors : ElectrodeAnchors
        Independently-computed anchors. Contacts this function does not touch
        keep theirs exactly.
    coords : (n, 3) array
        The coordinates the anchors were computed from, in the surfaces' frame.
    hemis, inflated : mappings
        ``{"lh": ..., "rh": ...}``. The inflated surface is what the coherence
        term is measured on; two anchors a millimetre apart anatomically can be
        forty apart there, and that difference is the entire problem.
    order : sequence of index sequences, optional
        Each device's rows in order along its shank. Without it, contacts are
        taken in montage order within each ``(group, hemi)``, which is how every
        format in use records a shank.

    Returns
    -------
    (ElectrodeAnchors, CoherenceReport)
    """
    from dataclasses import replace

    from ._anchor import NO_COORDINATE, is_straight

    n = len(anchors)
    if groups is None:
        return anchors, CoherenceReport(0, 0, 0.0, 0, n)

    groups = np.asarray(groups, dtype=object)
    types = (np.array([""] * n, dtype=object) if group_types is None
             else np.asarray(group_types, dtype=object))
    shared = {str(t).lower() for t in shared_frame_types}
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))

    verts = np.array(anchors.verts, copy=True)
    weights = np.array(anchors.weights, copy=True)
    depth = np.array(anchors.depth, copy=True)
    depth_mm = np.array(anchors.depth_mm, copy=True)
    thickness_mm = np.array(anchors.thickness_mm, copy=True)
    offset_mm = np.array(anchors.offset_mm, copy=True)
    usable = (anchors.placement != NO_COORDINATE) & np.isfinite(coords).all(axis=1)

    devices = order
    if devices is None:
        devices = []
        for key in sorted({(str(g), str(h)) for g, h in zip(groups, anchors.hemi)}):
            member = np.nonzero(
                (groups == key[0]) & (anchors.hemi == key[1]) & usable
            )[0]
            if len(member) >= 3:
                devices.append(member)

    touched = 0
    worst = 0.0
    tears = 0
    ndev = 0
    for member in devices:
        member = np.asarray(member, dtype=np.intp)
        hemi = str(anchors.hemi[member[0]])
        if hemi not in HEMIS or hemi not in hemis or hemi not in inflated:
            continue
        # The label says what the device is *meant* to be; the geometry says
        # whether this model applies to it. Both must agree.
        #
        # A ``grid`` or ``strip`` label excludes outright: those contacts each
        # genuinely belong to the column nearest them, and re-choosing their
        # anchors would be solving a problem they do not have.
        kinds = {str(t).lower() for t in types[member] if str(t).strip()}
        if kinds and not kinds & shared:
            continue
        # Straightness is then required whatever the label -- including an
        # explicit ``seeg``. The coherence term compares an inflated gap against
        # a native spacing, which is a statement about a rigid shank and means
        # nothing for a set of contacts that merely share a name. Real shanks are
        # rigid needles and pass easily: all 21 of TCH06's groups and all 13 of
        # S0033's clear the threshold by an order of magnitude. What this rejects
        # is a mislabelled or scattered group, where the DP would otherwise
        # shuffle anchors within the cap for no benefit at all.
        if not is_straight(coords[member]):
            continue

        solved = solve_device(
            coords[member], hemis[hemi], inflated[hemi],
            incumbent=(anchors.verts[member], anchors.weights[member]),
            max_fidelity_loss_mm=max_fidelity_loss_mm,
            max_anchor_shift_mm=max_anchor_shift_mm,
            coherence_weight=coherence_weight,
            neighbours=neighbours,
        )
        verts[member] = solved.verts
        weights[member] = solved.weights
        depth[member] = solved.depth
        depth_mm[member] = solved.depth * solved.thickness_mm
        thickness_mm[member] = solved.thickness_mm
        offset_mm[member] = solved.offset_mm
        touched += len(member)
        ndev += 1
        worst = max(worst, float(np.nanmax(solved.fidelity_loss_mm)))
        tears += int(solved.switched.sum())

    # `dist_pia_mm` and `dist_wm_mm` are deliberately *not* recomputed: they ask
    # how far this contact is from any cortex at all, over the neighbourhood the
    # search gathered, which is a fact about where the contact is rather than
    # about which column it was assigned. The placement policy bounds them, and
    # re-anchoring must not be able to change whether a contact is `too_far`.
    out = replace(
        anchors, verts=verts, weights=weights, depth=depth, depth_mm=depth_mm,
        thickness_mm=thickness_mm, offset_mm=offset_mm,
    )

    # The frame is measured *against the anchor triangle*, so re-anchoring
    # invalidates it too. Recomputed here rather than left to a caller, because
    # the invariant belongs with whoever broke it: `regroup_anchors` happens to
    # rebuild frames on most paths and returns early on `per_contact`, which
    # left a stale frame reconstructing positions 14 mm out -- silently, since a
    # stale frame is a perfectly well-formed one.
    if touched and out.frame is not None:
        frame, scale = frame_components(
            coords, out, {h: pair.pia for h, pair in hemis.items()},
            anchor_index=out.anchor_index,
        )
        out = replace(out, frame=frame, frame_scale_mm=scale)
    return out, CoherenceReport(ndev, touched, worst, tears, n)


class CoherenceReport(NamedTuple):
    """What :func:`coherent_anchors` did, and what it could not do.

    ``tears`` is the honest half. Where two consecutive contacts belong to gyri
    that inflation genuinely separates, no anchor within the fidelity cap closes
    the gap -- measured on one S0033 device, a 45 mm jump reduces to 42.5 mm even
    with an eighty-neighbour candidate set and *no* cap at all. Those breaks are
    anatomy, not error, and are reported rather than smoothed away.
    """

    devices: int
    contacts: int
    worst_fidelity_loss_mm: float
    tears: int
    total: int

    def summary(self) -> str:
        return (
            "coherent anchoring: %d devices, %d of %d contacts re-anchored; "
            "worst fidelity loss %.2f mm; %d pairs still torn by anatomy"
            % (self.devices, self.contacts, self.total,
               self.worst_fidelity_loss_mm, self.tears)
        )
