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
    check_alignment,
    load_surface_pairs,
)

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
