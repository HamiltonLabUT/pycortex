"""Keeping an electrode off the surface it is anchored to.

The arithmetic, on synthetic sheets, so every expected value is checkable by
hand. :mod:`cortex.tests.test_electrodes_subject` runs the same ideas against
real folded cortex, where the interesting failures live.

The property under test throughout is that a frame is a *complete* description:
three barycentric weights say where in the sheet, three signed millimetres say
how far off it, and together they reconstruct the input coordinate exactly. Not
approximately -- exactly, to floating point -- because the frame is an
orthonormal decomposition and nothing has been discarded. Anywhere a test here
uses ``allclose`` with a loose tolerance, that is a statement about the surface,
not about the frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex.electrodes import (
    ANCHOR_PER_CONTACT,
    ANCHOR_PER_DEVICE,
    NO_COORDINATE,
    OFFSET_FRAME,
    OFFSET_NONE,
    SCALE_ANISOTROPIC,
    SCALE_RIGID,
    SCALE_SIMILARITY,
    anchor_to_surfaces,
    frame_components,
    is_straight,
    regroup_anchors,
)
from cortex.electrodes._anchor import _triangle_basis

from .test_electrode_anchor import plane, plane_pair

EXACT = 1e-9
"""Millimetres. A frame round-trip is an orthonormal change of basis, so its
error is floating-point noise -- around 1e-15 mm on real surfaces -- rather than
anything the geometry contributes. Asserting a *tight* bound is the point: a
loose one would pass for a decomposition that had quietly lost the tangential
components, which is exactly the bug that renders plausibly."""


@pytest.fixture
def hemis():
    return {"lh": plane_pair(-30.0, -10.0), "rh": plane_pair(10.0, 30.0)}


def pia_of(hemis):
    return {h: pair.pia for h, pair in hemis.items()}


# -- the triangle basis ----------------------------------------------------

def test_the_basis_is_orthonormal_and_right_handed():
    tri = np.array([
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        [[1.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 5.0]],
    ])
    t1, t2, normal, scale = _triangle_basis(tri)

    for v in (t1, t2, normal):
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
    assert np.allclose(np.einsum("mi,mi->m", t1, t2), 0.0)
    assert np.allclose(np.einsum("mi,mi->m", t1, normal), 0.0)
    assert np.allclose(np.einsum("mi,mi->m", t2, normal), 0.0)
    # Right-handed: t1 x t2 == normal, not -normal.
    assert np.allclose(np.cross(t1, t2), normal)

    # t1 runs along the v0 -> v1 edge.
    assert np.allclose(t1[0], [1.0, 0.0, 0.0])
    # sqrt(2 * area): the first triangle has area 3, the second 6.
    assert np.allclose(scale, np.sqrt([6.0, 12.0]))


def test_the_scale_is_isotropic_not_the_first_edge():
    """A triangle scaled by 2 reports a scale factor of 2 whatever its shape.

    Worth pinning because the obvious implementation -- the length of the
    ``v0 -> v1`` edge -- gives the same answer for an equilateral triangle and a
    badly wrong one for a sliver, and *which* edge is ``v0 -> v1`` is an artifact
    of how the triangle was written down rather than a fact about the surface.
    """
    sliver = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.1, 0.0]]])
    assert _triangle_basis(sliver * 2)[3] / _triangle_basis(sliver)[3] == pytest.approx(2.0)

    # And it does not depend on which corner is written first.
    rolled = np.roll(sliver, 1, axis=1)
    assert _triangle_basis(rolled)[3] == pytest.approx(_triangle_basis(sliver)[3])


def test_a_degenerate_triangle_gives_nan_rather_than_a_fabricated_frame():
    """The flat surface has real degenerate triangles.

    ``freesurfer._move_disconnect_points_to_zero`` collapses vertices that no
    surviving polygon references onto the origin, so a triangle that references
    one has zero area. A frame there is not merely inaccurate, it does not
    exist, and saying so is better than returning an arbitrary basis that puts
    an electrode somewhere specific and wrong.
    """
    degenerate = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],   # collinear
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],   # collapsed
    ])
    t1, t2, normal, scale = _triangle_basis(degenerate)
    assert np.isnan(t1).all() and np.isnan(t2).all() and np.isnan(normal).all()
    assert np.isnan(scale).all()


# -- the round trip --------------------------------------------------------

def test_a_frame_reconstructs_the_coordinate_exactly(hemis):
    """The property everything else rests on.

    Evaluated against the surface it was measured on, a frame gives back the
    electrode's own coordinate. Today's purely barycentric ``evaluate`` cannot:
    it returns the anchor point, which is a different thing.
    """
    rng = np.random.default_rng(0)
    coords = np.column_stack([
        rng.uniform(-28.0, -12.0, 50),
        rng.uniform(-8.0, 8.0, 50),
        rng.uniform(-6.0, 4.0, 50),          # inside, above and below the slab
    ])
    anchors = anchor_to_surfaces(coords, hemis)

    back = anchors.evaluate(pia_of(hemis), offset=OFFSET_FRAME)
    assert np.abs(back - coords).max() < EXACT

    # And the projection is genuinely a different answer, or this proves nothing.
    on = anchors.evaluate(pia_of(hemis), offset=OFFSET_NONE)
    assert np.linalg.norm(on - coords, axis=1).max() > 1.0


def test_offset_none_is_unchanged(hemis):
    """The default must be the old behaviour, byte for byte."""
    coords = np.array([[-20.0, 0.0, 2.0], [-15.0, 3.0, -5.0], [20.0, -4.0, 1.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    surfaces = pia_of(hemis)
    assert np.array_equal(
        anchors.evaluate(surfaces), anchors.evaluate(surfaces, offset=OFFSET_NONE)
    )


def test_a_frame_on_a_flat_sheet_reads_as_height_and_slide(hemis):
    """On a sheet in the z = 0 plane the components have obvious meanings.

    The normal component is height above the sheet and the tangential ones are
    the in-plane displacement from the anchor point. On a flat test surface the
    anchor point is directly below the electrode, so the tangential components
    are zero and the normal one is the height -- which is the simplest possible
    statement of what the decomposition means.
    """
    coords = np.array([[-20.0, 0.0, 2.5], [-20.0, 0.0, -8.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    frame = anchors.frame

    assert np.allclose(frame[:, :2], 0.0, atol=EXACT)
    # The sheet's normal is +z (its polys are wound counter-clockwise from
    # above), so above the sheet is positive and below it negative.
    assert frame[0, 2] == pytest.approx(2.5, abs=EXACT)
    assert frame[1, 2] == pytest.approx(-8.0, abs=EXACT)


def test_evaluating_on_a_scaled_sheet_scales_the_offset(hemis):
    """The three scale modes, on a target that is uniformly twice the size.

    A doubled sheet is the one case where the right answer for each mode is
    obvious, which is what makes it worth testing here rather than on cortex.
    """
    coords = np.array([[-20.0, 0.0, 3.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    doubled = {h: pair.pia * 2.0 for h, pair in hemis.items()}

    on = anchors.evaluate(doubled)
    height = lambda mode: float(  # noqa: E731
        (anchors.evaluate(doubled, offset=OFFSET_FRAME, scale_mode=mode) - on)[0, 2]
    )
    assert height(SCALE_RIGID) == pytest.approx(3.0, abs=EXACT)
    assert height(SCALE_SIMILARITY) == pytest.approx(6.0, abs=EXACT)
    # Anisotropic scales the two tangential axes and leaves the normal alone.
    assert height(SCALE_ANISOTROPIC) == pytest.approx(3.0, abs=EXACT)


# -- sharing a frame across a device ---------------------------------------

def shaft(start, direction, n=8, pitch=4.0):
    return np.asarray(start) + np.outer(np.arange(n) * pitch, direction)


def test_a_shared_frame_keeps_a_device_rigid_on_a_stretched_surface(hemis):
    """The point of ``per_device``, on a target that stretches non-uniformly.

    A sheet stretched more at one end than the other is what a folded cortex
    does to a shaft under inflation. With one frame per contact the device
    smears; with one frame for the device it stays a rigid ruler and only its
    overall size changes.
    """
    coords = shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8])
    groups = ["LTD"] * len(coords)
    types = ["seeg"] * len(coords)
    anchors = anchor_to_surfaces(coords, hemis)
    surfaces = pia_of(hemis)

    shared = regroup_anchors(
        anchors, coords, surfaces, groups, types, mode=ANCHOR_PER_DEVICE
    )
    assert len(np.unique(shared.hosts)) == 1

    # A sheet that stretches along y, so the anchor triangles are not all
    # scaled alike and a per-contact answer cannot stay uniform.
    stretched = {}
    for hemi, pair in hemis.items():
        pts = pair.pia.copy()
        pts[:, 1] *= 1.0 + 0.05 * (pts[:, 1] + 10.0)
        stretched[hemi] = pts

    positions = shared.evaluate(
        stretched, offset=OFFSET_FRAME, scale_mode=SCALE_SIMILARITY
    )
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert steps.max() - steps.min() < EXACT           # uniform pitch

    centred = positions - positions.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    off_line = np.linalg.norm(centred - np.outer(centred @ axis, axis), axis=1)
    assert off_line.max() < EXACT                      # and still a straight line

    # The round trip is untouched by sharing: same surface, same coordinates.
    assert np.abs(shared.evaluate(surfaces, offset=OFFSET_FRAME) - coords).max() < EXACT


def test_auto_shares_a_shaft_and_leaves_a_grid_alone(hemis):
    coords = np.vstack([
        shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8]),
        np.column_stack([
            np.full(9, -16.0) + np.tile([0.0, 2.0, 4.0], 3),
            np.repeat([-2.0, 0.0, 2.0], 3),
            np.full(9, 1.5),
        ]),
    ])
    groups = ["LTD"] * 8 + ["LG"] * 9
    types = ["seeg"] * 8 + ["grid"] * 9
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis), groups, types
    )
    assert len(np.unique(anchors.hosts[:8])) == 1       # the shaft is one body
    assert len(np.unique(anchors.hosts[8:])) == 9       # every grid contact its own


def test_straightness_separates_a_shaft_from_a_draped_device():
    """The geometric fallback, on the shapes it has to tell apart."""
    straight = shaft([0.0, 0.0, 0.0], [0.0, 1.0, -0.8], n=10)
    assert is_straight(straight)
    # Real localisation is not exact; a shaft jittered by half a millimetre is
    # still a shaft. The measured worst case over three clinical montages was
    # 0.95 mm.
    jittered = straight + np.array([
        [0.4, -0.3, 0.5], [-0.5, 0.2, -0.4], [0.3, 0.5, 0.2], [-0.2, -0.5, 0.3],
        [0.5, 0.3, -0.5], [-0.4, -0.2, 0.4], [0.2, 0.4, -0.3], [-0.3, 0.5, 0.2],
        [0.5, -0.4, -0.2], [-0.5, 0.3, 0.5],
    ])
    assert is_straight(jittered)

    # A grid is not a line at all.
    grid = np.array([[x * 5.0, y * 5.0, 0.0] for x in range(4) for y in range(4)])
    assert not is_straight(grid)

    # Nor is a strip that follows the convexity it lies on: linear, but bent.
    angle = np.linspace(-0.6, 0.6, 8)
    strip = np.column_stack([40 * np.sin(angle), 40 * np.cos(angle), np.zeros(8)])
    assert not is_straight(strip)


def test_two_contacts_are_not_evidence_of_a_device():
    """Any two points are collinear, so a pair must not claim to be a shaft."""
    assert not is_straight(np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
    assert not is_straight(np.zeros((0, 3)))
    # Non-finite rows are excluded before the fit rather than poisoning it.
    with_nan = np.array([[0.0, 0.0, 0.0], [np.nan] * 3, [0.0, 4.0, 0.0]])
    assert not is_straight(with_nan)              # two usable points is still two


def test_an_unlabelled_shaft_is_grouped_from_its_geometry(hemis):
    """Two of the three real montages in this filestore carry no group type.

    Every one of TCH06's twenty-one groups is a depth electrode, and without
    this fallback all 186 of its contacts got per-contact anchoring -- silently,
    and for no better reason than a missing column.
    """
    coords = shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=8)
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LTD"] * 8, group_types=[""] * 8,
    )
    assert len(np.unique(anchors.hosts)) == 1

    # Passing no group_types at all takes the same path.
    again = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis), ["LTD"] * 8
    )
    assert len(np.unique(again.hosts)) == 1


def test_an_unlabelled_grid_is_left_alone(hemis):
    coords = np.array([
        [-16.0 + i * 3.0, -3.0 + j * 3.0, 1.5] for i in range(4) for j in range(4)
    ])
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LG"] * 16, group_types=[""] * 16,
    )
    assert len(np.unique(anchors.hosts)) == 16


def test_an_explicit_label_beats_the_geometry(hemis):
    """A montage that says "grid" is believed even if it happens to be straight.

    A 1xN grid is a legal thing to record, and guessing against a column the
    montage actually filled in would be overreach -- the fallback exists for
    the missing case, not to second-guess the stated one.
    """
    coords = shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=8)
    assert is_straight(coords)
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LG"] * 8, group_types=["grid"] * 8,
    )
    assert len(np.unique(anchors.hosts)) == 8

    # And per_device still overrides, for a caller who knows better.
    forced = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LG"] * 8, group_types=["grid"] * 8, mode=ANCHOR_PER_DEVICE,
    )
    assert len(np.unique(forced.hosts)) == 1


def test_per_contact_leaves_the_anchors_untouched(hemis):
    coords = shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8])
    anchors = anchor_to_surfaces(coords, hemis)
    same = regroup_anchors(
        anchors, coords, pia_of(hemis), ["LTD"] * 8, ["seeg"] * 8,
        mode=ANCHOR_PER_CONTACT,
    )
    assert same is anchors


def test_a_device_never_spans_hemispheres(hemis):
    """One group name on both sides splits rather than anchoring across.

    A montage that names its left and right shafts alike is a real thing, and
    anchoring a right-hemisphere contact to a left-hemisphere triangle would put
    it centimetres away with nothing to say so.
    """
    coords = np.vstack([
        shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=4),
        shaft([20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=4),
    ])
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["TD"] * 8, ["seeg"] * 8,
    )
    hosts = anchors.hosts
    assert len(np.unique(hosts)) == 2
    assert len(np.unique(hosts[:4])) == 1 and len(np.unique(hosts[4:])) == 1
    assert hosts[0] != hosts[4]


def test_a_missing_coordinate_is_not_placed_by_its_neighbours(hemis):
    """A NaN row keeps its own anchor and stays NaN.

    Unconnected amplifier channels arrive inside a real device's group. Letting
    the device's frame place them would turn a row that means "no electrode
    here" into a marker somewhere plausible.
    """
    coords = shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=6)
    coords[3] = np.nan
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LTD"] * 6, ["seeg"] * 6,
    )
    assert anchors.placement[3] == NO_COORDINATE
    assert anchors.hosts[3] == 3
    assert len(np.unique(np.delete(anchors.hosts, 3))) == 1

    positions = anchors.evaluate(pia_of(hemis), offset=OFFSET_FRAME)
    assert np.isnan(positions[3]).all()
    assert np.isfinite(np.delete(positions, 3, axis=0)).all()


# -- bookkeeping -----------------------------------------------------------

def test_selecting_a_whole_device_keeps_it_rigid(hemis):
    coords = np.vstack([
        shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=4),
        np.array([[-16.0, 0.0, 1.5], [-14.0, 0.0, 1.5]]),
    ])
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LTD"] * 4 + ["LG"] * 2, ["seeg"] * 4 + ["grid"] * 2,
    )
    sub = anchors[np.array([0, 1, 2, 3])]
    assert len(np.unique(sub.hosts)) == 1
    assert np.abs(
        sub.evaluate(pia_of(hemis), offset=OFFSET_FRAME) - coords[:4]
    ).max() < EXACT


def test_selecting_away_the_frames_host_falls_back_to_self_anchoring(hemis):
    """Losing the anchor contact means the rest are no longer one device.

    ``anchor_index`` names rows of the set it was computed for, so a subset has
    to renumber it. Where the frame-defining contact did not survive the
    selection there is nothing honest to renumber it to -- and silently keeping
    the stale index would point at whichever contact now occupies that row.
    """
    coords = shaft([-20.0, -6.0, 1.0], [0.0, 1.0, -0.8], n=5)
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, hemis), coords, pia_of(hemis),
        ["LTD"] * 5, ["seeg"] * 5,
    )
    host = int(anchors.hosts[0])
    keep = np.array([i for i in range(5) if i != host])
    sub = anchors[keep]

    assert np.array_equal(sub.hosts, np.arange(len(keep)))
    assert (sub.hosts < len(sub)).all()
    # The frames themselves were measured against the old host, so they are
    # stale; what matters is that nothing indexes out of range or silently
    # points at a different contact.
    assert len(sub) == 4


def test_frame_components_rejects_a_length_mismatch(hemis):
    coords = np.array([[-20.0, 0.0, 1.0], [-18.0, 0.0, 1.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    with pytest.raises(ValueError, match="rows but there are"):
        frame_components(coords[:1], anchors, pia_of(hemis))


def test_evaluate_rejects_unknown_modes(hemis):
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, 1.0]]), hemis)
    with pytest.raises(ValueError, match="offset must be one of"):
        anchors.evaluate(pia_of(hemis), offset="sideways")
    with pytest.raises(ValueError, match="scale_mode must be one of"):
        anchors.evaluate(pia_of(hemis), scale_mode="squish")


def test_anchors_without_a_frame_refuse_to_offset(hemis):
    """Anchors from before frames existed load, and say so rather than guessing."""
    from dataclasses import replace

    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, 1.0]]), hemis)
    old = replace(anchors, frame=None, frame_scale_mm=None)
    assert np.isfinite(old.evaluate(pia_of(hemis))).all()      # still works
    with pytest.raises(ValueError, match="needs a frame"):
        old.evaluate(pia_of(hemis), offset=OFFSET_FRAME)
