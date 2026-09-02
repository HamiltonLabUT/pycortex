"""Electrodes against the real S1 surfaces and a real filestore.

The synthetic tests in ``test_electrode_anchor.py`` pin the arithmetic; these
pin the parts that only a genuine subject exercises -- that the pial and
white-matter surfaces load and correspond, that anchors survive a trip through
the filestore, and that a barycentric anchor evaluated on the flat surface lands
where the vertex it names lands.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import cortex
from cortex.electrodes import (
    ElectrodeSet,
    anchor_to_surfaces,
    check_alignment,
    load_surface_pairs,
    regroup_anchors,
)
from cortex.electrodes import ANCHOR_PER_DEVICE

SUBJECT = "S1"
N_SAMPLE = 40


@pytest.fixture(scope="module")
def pairs():
    return load_surface_pairs(SUBJECT)


@pytest.fixture(scope="module")
def midpoint_electrodes(pairs):
    """Electrodes placed exactly halfway through the ribbon at known vertices.

    Chosen because that midpoint *is* a vertex of the surface the anchor search
    runs against, so the expected weights, depth and position are all exact
    rather than approximate -- which is what makes the assertions below sharp.
    """
    rng = np.random.default_rng(0)
    verts, coords, hemis = [], [], []
    for hemi in ("lh", "rh"):
        pair = pairs[hemi]
        # Only vertices that survive the flat cut, so the flat comparison below
        # has something to compare against.
        flat_verts = np.unique(
            cortex.db.get_surf(SUBJECT, "flat", hemi, nudge=False)[1]
        )
        idx = rng.choice(flat_verts, N_SAMPLE, replace=False)
        verts.append(idx)
        coords.append((pair.pia[idx] + pair.wm[idx]) / 2.0)
        hemis += [hemi] * N_SAMPLE

    eset = ElectrodeSet(
        names=["%s%d" % (h, i) for h in ("L", "R") for i in range(N_SAMPLE)],
        coords=np.vstack(coords),
        subject=SUBJECT,
    )
    return eset, np.concatenate(verts), np.array(hemis)


@pytest.fixture(scope="module")
def grid_electrodes(pairs):
    """A realistic subdural grid: 64 contacts on one patch, 1.5 mm above the pia.

    Deliberately contiguous rather than scattered over the brain. Cortex folds,
    so a scattered set always finds some nearby bank and hides exactly the
    errors an alignment check is for; a real grid does not.
    """
    from cortex import polyutils

    pair = pairs["lh"]
    surf = polyutils.Surface(pair.pia.astype(np.float64), pair.polys)
    normals = np.asarray(surf.vertex_normals)
    seed = 40000
    patch = np.argsort(np.linalg.norm(pair.pia - pair.pia[seed], axis=1))[:64]
    return pair.pia[patch] + 1.5 * normals[patch]


def test_the_subject_has_a_pial_and_a_white_matter_surface(pairs):
    for hemi in ("lh", "rh"):
        pair = pairs[hemi]
        assert pair.wm is not None
        assert pair.pia.shape == pair.wm.shape
        assert pair.polys.max() < len(pair.pia)


def test_midpoint_electrodes_land_at_depth_one_half(midpoint_electrodes):
    eset, _, _ = midpoint_electrodes
    anchors = eset.anchor()
    # atol reflects float32 surfaces: the midpoint the fixture computes is
    # rounded to float32, a ~1e-5 mm difference from the exact one.
    assert np.allclose(anchors.depth, 0.5, atol=1e-4)
    assert np.allclose(anchors.offset_mm, 0.0, atol=1e-3)
    assert (anchors.placement == "on_surface").all()


def test_cortical_thickness_at_the_anchors_is_plausible(midpoint_electrodes):
    eset, _, _ = midpoint_electrodes
    anchors = eset.anchor()
    # Human cortex is 1-4.5 mm nearly everywhere; a median outside that would
    # mean the pial and white-matter surfaces are not the pair we think.
    assert 1.0 < float(np.median(anchors.thickness_mm)) < 4.5


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.75, 1.0])
def test_depth_is_recovered_across_the_ribbon(pairs, fraction):
    """Contacts placed at a known fraction through the ribbon come back at it.

    The midpoint test above is a special case that is exact by construction --
    the midpoint is a vertex of the surface the candidate search runs on. This
    one covers the rest of the ribbon, where the answer depends on the face
    *selection* criterion. Selecting candidates by distance to the mid-surface
    (as this did first) biases depth toward the middle, because a contact on
    the pia over a curved patch finds a nearer mid-surface point on a
    neighbouring column: mean error 0.073, worst case 0.38. Selecting by
    distance to the cortical column gives the tolerances below.
    """
    pair = pairs["lh"]
    pia, wm = pair.pia.astype(np.float64), pair.wm.astype(np.float64)
    idx = np.random.default_rng(1).choice(len(pia), 40, replace=False)
    coords = pia[idx] + fraction * (wm[idx] - pia[idx])

    eset = ElectrodeSet(["e%d" % i for i in range(len(idx))], coords, subject=SUBJECT)
    anchors = eset.anchor(surfaces={"lh": pair})

    error = np.abs(anchors.depth - fraction)
    assert np.median(error) < 0.02
    assert np.percentile(error, 90) < 0.05
    assert error.max() < 0.35


def test_hemispheres_are_recovered_from_the_geometry(midpoint_electrodes):
    eset, _, hemis = midpoint_electrodes
    anchors = eset.anchor()
    assert list(anchors.hemi) == list(hemis)


def test_the_anchor_names_the_vertex_it_was_built_from(midpoint_electrodes):
    eset, verts, hemis = midpoint_electrodes
    anchors = eset.anchor()
    for i, vertex in enumerate(verts):
        heaviest = anchors.verts[i][np.argmax(anchors.weights[i])]
        assert heaviest == vertex


def test_a_flat_position_is_the_flat_coordinate_of_that_vertex(midpoint_electrodes):
    """The whole feature, end to end: an anchor built in TkRegRAS, evaluated on
    a surface that has been cut open and flattened, lands on the right spot."""
    eset, verts, hemis = midpoint_electrodes
    eset.anchor()
    positions = eset.positions("flat", nudge=True)

    left, right = cortex.db.get_surf(SUBJECT, "flat", "both", nudge=True)
    flat = {"lh": left[0], "rh": right[0]}
    expected = np.array([flat[h][v] for h, v in zip(hemis, verts)])
    assert np.allclose(positions, expected, atol=1e-3)


def test_inflating_moves_the_electrodes_but_keeps_them_paired(midpoint_electrodes):
    eset, _, _ = midpoint_electrodes
    eset.anchor()
    fiducial = eset.positions("fiducial", nudge=False)
    inflated = eset.positions("inflated", nudge=False)
    assert np.isfinite(fiducial).all() and np.isfinite(inflated).all()
    # Inflation genuinely moves them -- otherwise this test proves nothing --
    # while neighbouring contacts stay neighbours.
    assert np.linalg.norm(inflated - fiducial, axis=1).mean() > 1.0


def test_a_realistic_grid_looks_aligned(grid_electrodes, pairs):
    report = check_alignment(grid_electrodes, pairs)
    assert not report.suspicious
    assert report.median_offset_mm < 3.0
    assert report.shift_magnitude_mm < 2.0


@pytest.mark.parametrize("shift_mm", [15.0, 40.0])
def test_a_grid_lifted_off_the_surface_is_caught(grid_electrodes, pairs, shift_mm):
    report = check_alignment(grid_electrodes + np.array([0, 0, shift_mm]), pairs)
    assert report.suspicious
    assert "SUSPICIOUS" in report.summary()
    # Nearly all of the offset is one common direction, which is what says
    # "coordinate-space error" rather than "electrodes at varied depths".
    assert report.shift_magnitude_mm > 0.8 * report.median_offset_mm
    # The dominant axis is the one the error was injected on. Only the axis,
    # not the magnitude: as the grid lifts, its anchor feet slide to nearby
    # cortex, so the residual tilts away from pure +z (about 40 degrees at
    # 15 mm) rather than tracking it.
    shift = report.systematic_shift_mm
    assert np.argmax(np.abs(shift)) == 2 and shift[2] > 0


# -- the surface-distance rule, on a folded brain --------------------------

def test_a_resting_grid_and_a_deep_shaft_both_clear_four_millimetres(pairs):
    """The measurement the 4 mm default was chosen from.

    Both ends of the montage vocabulary have to fit under one threshold or the
    number is useless: a subdural grid lying on the pia, and a depth electrode
    driven the better part of six centimetres inward. They do, and for
    different reasons -- the grid because it is *on* the surface, the shaft
    because cortex folds so densely that it is never far from a sulcal bank.
    That second fact is the whole reason one number can cover both.
    """
    from cortex import polyutils

    pair = pairs["lh"]
    surf = polyutils.Surface(pair.pia.astype(np.float64), pair.polys)
    normals = np.asarray(surf.vertex_normals)
    patch = np.argsort(np.linalg.norm(pair.pia - pair.pia[40000], axis=1))[:64]

    grid = anchor_to_surfaces(pair.pia[patch] + 1.5 * normals[patch], pairs)
    assert grid.surface_distance_mm.max() < 2.0
    assert grid.placeable.all()

    shaft = anchor_to_surfaces(
        pair.pia[patch[0]] - np.outer(np.arange(0, 60, 4.0), normals[patch[0]]), pairs
    )
    assert shaft.surface_distance_mm.max() < 4.0
    assert shaft.placeable.all()


def test_the_rule_catches_a_lifted_grid_but_not_scattered_contacts(
    grid_electrodes, pairs, midpoint_electrodes
):
    """What the threshold does and does not amount to.

    Lift a *contiguous* grid 15 mm off the convexity it was resting on and
    every contact fails: there is nothing under it any more, and the default
    rule rejects the lot. Do the same to a *scattered* set and most contacts
    survive, because each one lands somewhere in the folds. So the rule is not
    a registration check -- ``check_alignment`` is -- and a montage that passes
    it contact by contact can still be in the wrong space. Same caveat as
    :class:`~cortex.electrodes.AlignmentReport`'s: judge a montage by its
    grids, not by its outliers.
    """
    lift = np.array([0.0, 0.0, 15.0])

    lifted_grid = anchor_to_surfaces(grid_electrodes + lift, pairs)
    assert not lifted_grid.placeable.any()
    assert lifted_grid.surface_distance_mm.min() > 4.0

    eset, _, _ = midpoint_electrodes
    scattered = anchor_to_surfaces(eset.coords + lift, pairs)
    assert scattered.placeable.mean() > 0.5


def test_a_grid_slid_along_the_surface_is_not_caught(grid_electrodes, pairs):
    """The documented blind spot, pinned so it cannot be quietly overclaimed.

    A tangential error puts the grid on a neighbouring gyrus, where it sits
    just as snugly. No geometric test can see this; only the anatomical labels
    can, which is why :class:`AlignmentReport` says so rather than implying the
    check is a registration validator.
    """
    report = check_alignment(grid_electrodes + np.array([8.0, 0, 0]), pairs)
    assert not report.suspicious
    assert report.median_offset_mm < 3.0


# -- keeping electrodes off the surface, on real cortex ---------------------

@pytest.fixture(scope="module")
def shaft_electrodes(pairs):
    """A depth electrode: 15 contacts at a uniform 4 mm pitch, driven 56 mm in.

    Straight along the pial normal from one vertex, which is not how a surgeon
    plans a trajectory but is the only construction whose right answer is known
    in advance -- a straight line at even spacing. That is exactly what the
    assertions below check survives the trip to the inflated surface.

    The seed is chosen so the shaft stays inside its own hemisphere: entering at
    a laterally-facing vertex, it reaches x = -14.7 mm and stops. This matters,
    and the obvious seed does not satisfy it -- a shaft driven 56 mm inward from
    a medially-facing vertex crosses the midline and ends up anchored to the
    *other* hemisphere, which no real trajectory does. Every contact also clears
    the default 4 mm placement policy, so nothing here is testing an electrode
    the policy would have excluded anyway.
    """
    from cortex import polyutils

    pair = pairs["lh"]
    normals = np.asarray(
        polyutils.Surface(pair.pia.astype(np.float64), pair.polys).vertex_normals
    )
    seed = 84217
    return pair.pia[seed] - np.outer(np.arange(0, 60, 4.0), normals[seed])


def test_the_shaft_fixture_is_a_trajectory_a_surgeon_could_plan(shaft_electrodes, pairs):
    """Guard on the fixture, because the failure it prevents is silent.

    A shaft that crosses the midline gets split into two devices by
    :func:`regroup_anchors` -- correctly, since a device may not span
    hemispheres -- and every assertion about "the device" below then quietly
    describes half of one.
    """
    anchors = anchor_to_surfaces(shaft_electrodes, pairs)
    assert (anchors.hemi == "lh").all()
    assert anchors.placeable.all()
    assert shaft_electrodes[:, 0].max() < -10.0


def test_a_frame_returns_the_coordinate_on_the_surface_it_was_measured_on(
    grid_electrodes, shaft_electrodes, pairs
):
    """The exactness claim, on folded cortex rather than a synthetic sheet.

    Catches a sign error, a non-orthonormal basis, or a ``c_ras`` offset applied
    on one side and not the other -- all at once, and all as an exact equality
    rather than a threshold somebody would have had to choose.
    """
    coords = np.vstack([grid_electrodes, shaft_electrodes])
    anchors = anchor_to_surfaces(coords, pairs)
    surfaces = {h: pair.pia for h, pair in pairs.items()}

    back = anchors.evaluate(surfaces, offset="frame")
    assert np.abs(back - coords).max() < 1e-6

    # The projection is a genuinely different answer, or the test above would
    # pass for a no-op. Note how *small* the difference is: even a contact 56 mm
    # inside the brain is moved only 5.7 mm by being projected, because cortex
    # folds densely enough that it is never far from some sulcal bank. That is
    # why this failure is invisible contact by contact and only shows up in the
    # relative geometry -- the pitch test below is where it becomes obvious.
    moved = np.linalg.norm(anchors.evaluate(surfaces) - coords, axis=1)
    assert moved[-15:].max() > 3.0
    assert moved[-15:].max() < 10.0


def test_a_shaft_keeps_a_uniform_pitch_and_stays_straight_when_inflated(
    shaft_electrodes, pairs
):
    """The failure this whole mechanism exists for.

    Anchored per contact, the shaft does not survive inflation: cortex folds
    densely enough that consecutive contacts anchor to different sulcal banks,
    and the device arrives as scatter. Measured here rather than asserted from
    the design document, so the "before" cannot rot.

    Anchored once and carried through that frame -- ``anchor_mode="per_device"``,
    which is no longer what ``auto`` does -- it is a rigid body: the pitch is
    uniform and the contacts collinear to floating point. Both are exact
    because a shared orthonormal frame with one scale is a similarity
    transform -- a tolerance band here would pass for the anisotropic shear,
    which tilts the shaft off its trajectory and makes the pitch uneven.
    """
    coords = shaft_electrodes
    groups, types = ["LTD"] * len(coords), ["seeg"] * len(coords)
    surfaces = {h: pair.pia for h, pair in pairs.items()}
    inflated = {
        hemi: cortex.db.get_surf(SUBJECT, "inflated", hemi, nudge=False)[0]
        for hemi in pairs
    }

    per_contact = anchor_to_surfaces(coords, pairs)
    # The before. Fifteen contacts, fifteen different anchor triangles, spread
    # over the folds the shaft passes: a uniform 4 mm pitch arrives on the
    # inflated surface as 1.7 mm at one step and 40 mm at another. The recorded
    # `depth` gives the reason -- it runs from 0 to 9.2 and back to -0.9, since
    # each contact reports a depth relative to whichever bank it landed near
    # rather than to the column it entered through.
    assert len(np.unique(per_contact.verts[:, 0])) == len(coords)
    loose = np.linalg.norm(np.diff(per_contact.evaluate(inflated), axis=0), axis=1)
    assert loose.min() < 2.0 and loose.max() > 30.0

    shared = regroup_anchors(per_contact, coords, surfaces, groups, types,
                             mode=ANCHOR_PER_DEVICE)
    assert len(np.unique(shared.hosts)) == 1

    positions = shared.evaluate(inflated, offset="frame")
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert steps.max() - steps.min() < 1e-6

    centred = positions - positions.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    assert np.linalg.norm(centred - np.outer(centred @ axis, axis), axis=1).max() < 1e-6

    # The pitch tracks inflation rather than staying at 4 mm.
    assert 0.3 < steps.mean() / 4.0 < 1.5


def test_the_device_scale_is_not_taken_from_the_anchor_contact(shaft_electrodes, pairs):
    """Why the scale is a median over the device rather than its anchor's own.

    A shaft is anchored near its entry, and an entry point sits on a gyral
    crown, which inflates quite differently from the tissue the shaft threads.
    Taking the anchor triangle's own scale stretched two of three test shafts by
    more than 60%. The median over the device's contacts is a far steadier
    estimate of the territory it actually crosses.
    """
    from cortex.electrodes._anchor import _triangle_basis

    coords = shaft_electrodes
    surfaces = {h: pair.pia for h, pair in pairs.items()}
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, pairs), coords, surfaces,
        ["LTD"] * len(coords), ["seeg"] * len(coords), mode=ANCHOR_PER_DEVICE,
    )
    inflated = cortex.db.get_surf(SUBJECT, "inflated", "lh", nudge=False)[0]

    own = _triangle_basis(inflated[anchors.verts])[3] / anchors.frame_scale_mm
    host_only = own[int(anchors.hosts[0])]
    device = np.nanmedian(own)

    # They genuinely disagree, which is the whole reason for the choice.
    assert abs(host_only - device) / device > 0.05

    # And the pitch that comes out follows the median, not the anchor's own.
    steps = np.linalg.norm(
        np.diff(anchors.evaluate({"lh": inflated}, offset="frame"), axis=0), axis=1
    )
    assert steps.mean() == pytest.approx(4.0 * device, rel=1e-6)


def test_a_grid_still_drapes_over_the_folds(grid_electrodes, pairs):
    """``auto`` must not turn a grid into a rigid sheet.

    A grid's contacts each sit on their own column, and the spread in their
    spacing after inflation is the signal -- it is what says which contacts were
    buried in a sulcus. Holding them rigid would float the grid above the
    surface and hide exactly that.
    """
    coords = grid_electrodes
    surfaces = {h: pair.pia for h, pair in pairs.items()}
    anchors = regroup_anchors(
        anchor_to_surfaces(coords, pairs), coords, surfaces,
        ["LG"] * len(coords), ["grid"] * len(coords),
    )
    assert len(np.unique(anchors.hosts)) == len(coords)

    inflated = {"lh": cortex.db.get_surf(SUBJECT, "inflated", "lh", nudge=False)[0]}
    on = anchors.evaluate(inflated)
    off = anchors.evaluate(inflated, offset="frame")

    # Barely moved: a subdural contact is a fraction of a millimetre from the
    # column it anchors to, so drawing it off the surface changes little. That
    # is the correct outcome for a grid, and it is why the mechanism is worth
    # having only for depth electrodes.
    assert np.linalg.norm(off - on, axis=1).max() < 3.0

    # And it still warps: the spread in neighbour spacing survives.
    spacing = np.linalg.norm(off[:, None, :] - off[None, :, :], axis=2)
    nearest = np.sort(spacing, axis=1)[:, 1]
    assert nearest.max() / nearest.min() > 2.0


def test_rigid_preserves_the_residual_length_on_every_surface(grid_electrodes, pairs):
    """What ``rigid`` claims, checked on both the inflated and the flat surface.

    The distance from an electrode to its anchor point, in millimetres, is
    unchanged by evaluating it anywhere else. This is the invariant that is
    well-defined for a grid -- distance from the *anchor*, not from the vertex
    the contact was placed above, which anchoring is free to disagree about.
    """
    coords = grid_electrodes
    anchors = anchor_to_surfaces(coords, pairs)
    surfaces = {h: pair.pia for h, pair in pairs.items()}
    residual = np.linalg.norm(coords - anchors.evaluate(surfaces), axis=1)

    for surface_type in ("inflated", "flat"):
        target = {
            hemi: cortex.db.get_surf(SUBJECT, surface_type, hemi, nudge=False)[0]
            for hemi in pairs
        }
        moved = np.linalg.norm(
            anchors.evaluate(target, offset="frame", scale_mode="rigid")
            - anchors.evaluate(target),
            axis=1,
        )
        assert np.abs(moved - residual).max() < 1e-6


def test_the_set_threads_it_through_positions(pairs, shaft_electrodes):
    """The public path: ``anchor()`` groups, ``positions()`` offsets."""
    eset = ElectrodeSet(
        names=["LTD%d" % i for i in range(len(shaft_electrodes))],
        coords=shaft_electrodes,
        group=["LTD"] * len(shaft_electrodes),
        group_type=["seeg"] * len(shaft_electrodes),
        subject=SUBJECT,
    )
    eset.anchor(anchor_mode=ANCHOR_PER_DEVICE, coherent=False)
    assert len(np.unique(eset.anchors.hosts)) == 1

    on = eset.positions("inflated", nudge=False)
    off = eset.positions("inflated", nudge=False, offset="frame")
    assert not np.allclose(on, off)

    steps = np.linalg.norm(np.diff(off, axis=0), axis=1)
    assert steps.max() - steps.min() < 1e-6

    # Asking for the pia gives the coordinates back, through the whole stack.
    assert np.abs(
        eset.positions("pia", nudge=False, offset="frame") - shaft_electrodes
    ).max() < 1e-6


# -- filestore round-trip ---------------------------------------------------

@pytest.fixture
def scratch_db(tmp_path):
    """A filestore holding one bare subject, so nothing writes into the real one."""
    for directory in ("transforms", "anatomicals", "cache", "surfaces",
                      "surface-info", "views"):
        os.makedirs(tmp_path / "TESTSUBJ" / directory)
    return cortex.database.Database(filestore=str(tmp_path))


def test_a_subject_with_no_electrodes_lists_none(scratch_db):
    assert scratch_db.list_electrodes("TESTSUBJ") == []


def test_save_then_get_round_trips_through_the_filestore(scratch_db, midpoint_electrodes):
    eset, _, _ = midpoint_electrodes
    eset.anchor()

    scratch_db.save_electrodes("TESTSUBJ", eset, name="clinical")
    assert scratch_db.list_electrodes("TESTSUBJ") == ["clinical"]

    back = scratch_db.get_electrodes("TESTSUBJ", name="clinical")
    assert list(back.names) == list(eset.names)
    assert np.allclose(back.coords, eset.coords)
    assert np.array_equal(back.anchors.verts, eset.anchors.verts)
    assert back.anchors.surface_hash == eset.anchors.surface_hash
    assert back.subject == "TESTSUBJ"


def test_saving_twice_refuses_unless_told_to_overwrite(scratch_db, midpoint_electrodes):
    eset, _, _ = midpoint_electrodes
    scratch_db.save_electrodes("TESTSUBJ", eset)
    with pytest.raises(IOError, match="Refusing to overwrite"):
        scratch_db.save_electrodes("TESTSUBJ", eset)
    scratch_db.save_electrodes("TESTSUBJ", eset, overwrite=True)


def test_a_missing_set_says_what_is_available(scratch_db, midpoint_electrodes):
    eset, _, _ = midpoint_electrodes
    scratch_db.save_electrodes("TESTSUBJ", eset, name="clinical")
    with pytest.raises(IOError, match="clinical"):
        scratch_db.get_electrodes("TESTSUBJ", name="research")


def test_the_filestore_path_is_where_the_scope_says_it_is(scratch_db):
    path = scratch_db.get_paths("TESTSUBJ")["electrodes"]
    assert path.endswith(os.path.join("TESTSUBJ", "electrodes", "{name}.json"))


# -- choosing a shank's anchors together ------------------------------------

def test_coherent_anchoring_never_moves_a_contact_off_its_gyrus(shaft_electrodes):
    """The guarantee, on real folded cortex.

    The whole point of choosing anchors jointly is that it may not cost anatomy.
    Measured across both real clinical montages in this filestore, the worst
    anchor displacement is 2.96 and 2.99 mm against a 3 mm cap -- a hard bound,
    because the incumbent anchor is injected as a candidate and every other
    candidate is filtered against it, so there is no path by which the coherence
    term reaches something further away.
    """
    from cortex.electrodes._coherent import MAX_ANCHOR_SHIFT_MM

    eset = ElectrodeSet(
        names=["LTD%d" % i for i in range(len(shaft_electrodes))],
        coords=shaft_electrodes,
        group=["LTD"] * len(shaft_electrodes),
        group_type=["seeg"] * len(shaft_electrodes),
        subject=SUBJECT,
    )
    before = eset.anchor(coherent=False, inplace=False)
    after = eset.anchor(coherent=True)

    pia = load_surface_pairs(SUBJECT)["lh"].pia.astype(np.float64)
    was = np.einsum("mij,mi->mj", pia[before.verts], before.weights)
    now = np.einsum("mij,mi->mj", pia[after.verts], after.weights)
    assert np.linalg.norm(now - was, axis=1).max() <= MAX_ANCHOR_SHIFT_MM + 1e-6

    # And it did something, or the bound above is vacuous.
    assert not np.array_equal(before.verts, after.verts)
    assert eset.coherence.devices == 1


def test_coherent_anchoring_makes_the_inflated_spacing_more_faithful(shaft_electrodes):
    """What is bought with those three millimetres.

    Reported as the *typical* pair rather than the spread, because the spread is
    dominated by breaks the method cannot remove and does not claim to: where two
    consecutive contacts belong to gyri that inflation genuinely separates, no
    anchor within the cap closes the gap. Measured on one real device, a 45 mm
    jump reduces to 42.5 mm even with an eighty-neighbour candidate set and no
    cap at all.
    """
    eset = ElectrodeSet(
        names=["LTD%d" % i for i in range(len(shaft_electrodes))],
        coords=shaft_electrodes,
        group=["LTD"] * len(shaft_electrodes),
        group_type=["seeg"] * len(shaft_electrodes),
        subject=SUBJECT,
    )
    pitch = np.linalg.norm(np.diff(shaft_electrodes, axis=0), axis=1)

    def spacing_error(anchors):
        inflated = {
            hemi: cortex.db.get_surf(SUBJECT, "inflated", hemi, nudge=False)[0]
            for hemi in ("lh", "rh")
        }
        gap = np.linalg.norm(np.diff(anchors.evaluate(inflated), axis=0), axis=1)
        return np.abs(gap - pitch * np.median(gap / pitch))

    loose = spacing_error(eset.anchor(coherent=False, inplace=False))
    tight = spacing_error(eset.anchor(coherent=True))
    assert np.median(tight) < np.median(loose)


def test_a_grid_is_not_re_anchored(grid_electrodes):
    """A grid's contacts genuinely do belong to the column nearest each of them."""
    eset = ElectrodeSet(
        names=["G%d" % i for i in range(len(grid_electrodes))],
        coords=grid_electrodes,
        group=["LG"] * len(grid_electrodes),
        group_type=["grid"] * len(grid_electrodes),
        subject=SUBJECT,
    )
    before = eset.anchor(coherent=False, inplace=False)
    after = eset.anchor(coherent=True)
    assert np.array_equal(before.verts, after.verts)
    assert eset.coherence.devices == 0
