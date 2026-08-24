"""Anchoring geometry, against synthetic surfaces rather than a subject.

Every test here builds its own two-plane "cortex" -- a pial sheet and a
white-matter sheet three millimetres below it -- so the arithmetic is checkable
by hand and no pycortex database is needed. Real surfaces are curved and folded,
but nothing in the anchoring cares: it is a nearest-triangle search plus a
projection onto one segment.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex.electrodes import (
    ON_SURFACE,
    PROJECTED,
    TOO_FAR,
    UNKNOWN_ANATOMY,
    PlacementPolicy,
    SurfacePair,
    anchor_to_surfaces,
    check_alignment,
    classify_placement,
    surface_hash,
)
from cortex.electrodes._anchor import _closest_point_weights

THICKNESS = 3.0


def plane(x0, x1, y0=-10.0, y1=10.0, n=11, z=0.0):
    """An ``n`` by ``n`` triangulated sheet in the z = ``z`` plane."""
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, float(z))], axis=1
    )
    idx = np.arange(n * n).reshape(n, n)
    polys = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            polys += [[a, b, c], [a, c, d]]
    return pts, np.array(polys, dtype=np.int32)


def plane_pair(x0, x1, thickness=THICKNESS):
    """A pial sheet at z = 0 with white matter ``thickness`` below it."""
    pia, polys = plane(x0, x1, z=0.0)
    wm, _ = plane(x0, x1, z=-thickness)
    return SurfacePair(pia=pia, polys=polys, wm=wm)


@pytest.fixture
def hemis():
    """Two sheets, left at negative x as in RAS, right at positive."""
    return {"lh": plane_pair(-30.0, -10.0), "rh": plane_pair(10.0, 30.0)}


# -- the closest-point primitive -------------------------------------------

@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.25, 0.25, 1.0), (0.5, 0.25, 0.25)),   # interior, straight down
        ((-1.0, -1.0, 0.0), (1.0, 0.0, 0.0)),     # past vertex A
        ((2.0, -1.0, 0.0), (0.0, 1.0, 0.0)),      # past vertex B
        ((-1.0, 2.0, 0.0), (0.0, 0.0, 1.0)),      # past vertex C
        ((0.5, -1.0, 0.0), (0.5, 0.5, 0.0)),      # edge AB
        ((2.0, 2.0, 0.0), (0.0, 0.5, 0.5)),       # edge BC
        ((-1.0, 0.5, 0.0), (0.5, 0.0, 0.5)),      # edge AC
    ],
)
def test_closest_point_covers_every_voronoi_region(point, expected):
    tri = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]])
    weights = _closest_point_weights(tri, np.array(point))
    assert np.allclose(weights[0], expected)


def test_closest_point_weights_sum_to_one():
    rng = np.random.default_rng(0)
    tri = rng.normal(size=(64, 3, 3))
    weights = _closest_point_weights(tri, np.array([0.3, -0.2, 0.7]))
    assert np.allclose(weights.sum(axis=1), 1.0)


# -- depth ------------------------------------------------------------------

def test_depth_is_zero_on_pia_and_one_at_white_matter(hemis):
    coords = np.array([[-20.0, 0.0, 0.0], [-20.0, 0.0, -THICKNESS]])
    anchors = anchor_to_surfaces(coords, hemis)
    assert np.allclose(anchors.depth, [0.0, 1.0])
    assert np.allclose(anchors.depth_mm, [0.0, THICKNESS])
    assert np.allclose(anchors.thickness_mm, THICKNESS)


def test_a_contact_above_the_pia_gets_negative_depth(hemis):
    # A subdural grid sits a millimetre or two proud of the pia; that has to
    # come out as "outside", not clamp to zero, or the viewer cannot tell a
    # grid from something inside the ribbon.
    coords = np.array([[-20.0, 0.0, 2.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    assert np.isclose(anchors.depth[0], -2.0 / THICKNESS)
    assert np.isclose(anchors.depth_mm[0], -2.0)
    assert np.isclose(anchors.offset_mm[0], 0.0, atol=1e-9)
    assert anchors.placement[0] == PROJECTED


def test_a_deep_contact_gets_depth_past_one(hemis):
    coords = np.array([[20.0, 0.0, -9.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    assert np.isclose(anchors.depth[0], 3.0)
    assert np.isclose(anchors.depth_mm[0], 9.0)
    # Directly beneath its own column, so it is deep rather than misplaced.
    assert np.isclose(anchors.offset_mm[0], 0.0, atol=1e-9)


def test_how_deep_is_too_deep_is_the_surface_distance_rule(hemis):
    """Six millimetres past the white matter of an *unfolded* test sheet.

    On real cortex a contact this deep is still a millimetre or two from some
    sulcal bank and stays placed; these sheets are flat, so nothing is near it
    and the default four-millimetre rule excludes it. That is the rule working,
    not a quirk of the fixture -- and it is one number away from not applying.
    """
    coords = np.array([[20.0, 0.0, -9.0]])
    assert anchor_to_surfaces(coords, hemis).placement[0] == TOO_FAR
    relaxed = anchor_to_surfaces(
        coords, hemis, policy=PlacementPolicy(max_surface_distance_mm=10.0)
    )
    assert relaxed.placement[0] == PROJECTED


def test_mid_ribbon_contact_is_on_surface(hemis):
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, -1.5]]), hemis)
    assert np.isclose(anchors.depth[0], 0.5)
    assert anchors.placement[0] == ON_SURFACE


def test_depth_is_nan_without_a_white_matter_surface():
    pia, polys = plane(-30.0, -10.0)
    anchors = anchor_to_surfaces(
        np.array([[-20.0, 0.0, 2.0]]), {"lh": SurfacePair(pia=pia, polys=polys)}
    )
    assert np.isnan(anchors.depth[0])
    assert np.isnan(anchors.thickness_mm[0])
    # The offset still means something: it is the distance to the surface.
    assert np.isclose(anchors.offset_mm[0], 2.0)


# -- hemispheres ------------------------------------------------------------

def test_hemisphere_comes_from_geometry_not_from_a_label(hemis):
    coords = np.array([[-20.0, 0.0, 1.0], [20.0, 0.0, 1.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    assert list(anchors.hemi) == ["lh", "rh"]


def test_one_hemisphere_is_enough():
    anchors = anchor_to_surfaces(
        np.array([[-20.0, 0.0, 1.0]]), {"lh": plane_pair(-30.0, -10.0)}
    )
    assert anchors.hemi[0] == "lh"


def test_an_unknown_hemisphere_name_is_refused():
    with pytest.raises(ValueError, match="hemisphere"):
        anchor_to_surfaces(np.zeros((1, 3)), {"left": plane_pair(-30.0, -10.0)})


# -- placement policy -------------------------------------------------------

def test_a_contact_far_off_its_column_is_flagged(hemis):
    # 30 mm sideways from the nearest sheet edge: no cortical column can claim
    # it, and it must not quietly acquire the boundary's anatomy.
    anchors = anchor_to_surfaces(np.array([[60.0, 0.0, 0.0]]), hemis)
    assert anchors.placement[0] == TOO_FAR
    assert anchors.offset_mm[0] > 10.0
    # Excluded by the surface-distance rule, which is on by default, rather
    # than by the column-offset bound, which is not.
    assert anchors.surface_distance_mm[0] > 4.0


def test_a_contact_far_above_the_pia_is_flagged(hemis):
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, 40.0]]), hemis)
    assert anchors.placement[0] == TOO_FAR


# -- the surface-distance rule ---------------------------------------------

@pytest.mark.parametrize(
    "height, expected",
    [(3.5, PROJECTED), (4.5, TOO_FAR)],
)
def test_four_millimetres_from_a_surface_is_the_default_rule(hemis, height, expected):
    """The one number that decides whether a contact is drawn at all.

    Straight above the pia, so the distance to it is the height and there is no
    second effect to disentangle.
    """
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, height]]), hemis)
    assert np.isclose(anchors.dist_pia_mm[0], height, atol=1e-6)
    assert anchors.placement[0] == expected


def test_the_rule_is_a_number_the_caller_sets(hemis):
    coords = np.array([[-20.0, 0.0, 6.0]])
    assert anchor_to_surfaces(coords, hemis).placement[0] == TOO_FAR
    for threshold in (8.0, np.inf):
        anchors = anchor_to_surfaces(
            coords, hemis,
            policy=PlacementPolicy(max_surface_distance_mm=threshold),
        )
        assert anchors.placement[0] == PROJECTED


def test_distance_is_measured_to_both_surfaces(hemis):
    """A contact just under the white matter is far from the pia and near it.

    ``surface_distance_mm`` takes the nearer, which is what makes one threshold
    cover a subdural grid and a depth contact without special-casing either.
    """
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, -4.0]]), hemis)
    assert np.isclose(anchors.dist_pia_mm[0], 4.0, atol=1e-6)
    assert np.isclose(anchors.dist_wm_mm[0], 1.0, atol=1e-6)
    assert np.isclose(anchors.surface_distance_mm[0], 1.0, atol=1e-6)
    assert anchors.placement[0] == PROJECTED


def test_the_rule_still_works_without_a_white_matter_surface():
    """``fmin``, not ``minimum``: a NaN white-matter distance must not swallow
    the pial one and reject every contact the subject has."""
    pia, polys = plane(-30.0, -10.0)
    hemis = {"lh": SurfacePair(pia=pia, polys=polys)}
    near = anchor_to_surfaces(np.array([[-20.0, 0.0, 2.0]]), hemis)
    assert np.isnan(near.dist_wm_mm[0])
    assert np.isclose(near.surface_distance_mm[0], 2.0, atol=1e-6)
    assert near.placement[0] == PROJECTED
    far = anchor_to_surfaces(np.array([[-20.0, 0.0, 9.0]]), hemis)
    assert far.placement[0] == TOO_FAR


def test_the_column_offset_bound_is_opt_in(hemis):
    """Off by default, and still the honest way to ask "over its own column?".

    These flat sheets cannot produce a contact that is near cortex *and* far
    off its own column -- that needs folding -- so the offset is set by hand to
    isolate the bound from the surface-distance rule that would otherwise fire
    first.
    """
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, -0.5]]), hemis)
    assert anchors.placement[0] == ON_SURFACE
    anchors.offset_mm = np.array([5.0])
    assert classify_placement(anchors, PlacementPolicy())[0] == ON_SURFACE
    assert classify_placement(
        anchors, PlacementPolicy(max_offset_mm=1.0)
    )[0] == TOO_FAR


def test_policy_is_reapplied_without_redoing_the_geometry(hemis):
    coords = np.array([[60.0, 0.0, 0.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    assert anchors.placement[0] == TOO_FAR

    relaxed = classify_placement(
        anchors, PlacementPolicy(max_surface_distance_mm=100.0)
    )
    assert relaxed[0] == ON_SURFACE
    # and the expensive part is untouched
    assert np.array_equal(anchors.verts[0], anchor_to_surfaces(coords, hemis).verts[0])


def test_the_anatomy_rule_is_off_by_default(hemis):
    anchors = anchor_to_surfaces(
        np.array([[-20.0, 0.0, -1.5]]), hemis, anatomy=np.array(["Unknown"])
    )
    assert anchors.placement[0] == ON_SURFACE


def test_the_anatomy_rule_marks_rather_than_deletes(hemis):
    coords = np.array([[-20.0, 0.0, -1.5], [-22.0, 0.0, -1.5]])
    anchors = anchor_to_surfaces(
        coords, hemis,
        policy=PlacementPolicy(drop_unknown_anatomy=True),
        anatomy=np.array(["Unknown", "STG"]),
    )
    assert list(anchors.placement) == [UNKNOWN_ANATOMY, ON_SURFACE]
    # The excluded electrode keeps a usable anchor, so the decision is reversible.
    assert np.isfinite(anchors.depth[0])
    assert list(anchors.placeable) == [False, True]


def test_the_anatomy_rule_needs_labels(hemis):
    with pytest.raises(ValueError, match="anatomy"):
        anchor_to_surfaces(
            np.array([[-20.0, 0.0, 0.0]]), hemis,
            policy=PlacementPolicy(drop_unknown_anatomy=True),
        )


def test_below_white_matter_bound_is_a_separate_opt_in_measure(hemis):
    """``max_below_wm_mm`` asks how deep, not how far from cortex.

    The two coincide on these unfolded sheets and part company on a real brain,
    where a deep contact beside a sulcal bank is far past the white matter and
    still close to cortex. Keeping both means a montage can be filtered by
    either question.
    """
    coords = np.array([[20.0, 0.0, -40.0]])
    relaxed = PlacementPolicy(max_surface_distance_mm=np.inf)
    assert anchor_to_surfaces(coords, hemis, policy=relaxed).placement[0] == PROJECTED
    bounded = anchor_to_surfaces(
        coords, hemis,
        policy=PlacementPolicy(max_surface_distance_mm=np.inf, max_below_wm_mm=10.0),
    )
    assert bounded.placement[0] == TOO_FAR


# -- evaluating on other surfaces ------------------------------------------

def test_the_anchor_follows_a_deformed_surface(hemis):
    """The whole point: one anchor, valid on every surface of the subject."""
    coords = np.array([[-20.0, 4.0, 2.0]])
    anchors = anchor_to_surfaces(coords, hemis)

    pial = {h: hemis[h].pia for h in hemis}
    assert np.allclose(anchors.evaluate(pial)[0], [-20.0, 4.0, 0.0])

    # An "inflated" surface: same vertices, twice as far apart in y.
    stretched = {}
    for hemi, pair in hemis.items():
        pts = pair.pia.copy()
        pts[:, 1] *= 2
        stretched[hemi] = pts
    assert np.allclose(anchors.evaluate(stretched)[0], [-20.0, 8.0, 0.0])


def test_evaluate_leaves_absent_hemispheres_as_nan(hemis):
    coords = np.array([[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    anchors = anchor_to_surfaces(coords, hemis)
    only_left = {"lh": hemis["lh"].pia}
    positions = anchors.evaluate(only_left)
    assert np.isfinite(positions[0]).all()
    assert np.isnan(positions[1]).all()


# -- alignment and hashing --------------------------------------------------

def test_aligned_coordinates_look_aligned(hemis):
    coords = np.array([[-20.0, 0.0, 1.0], [-22.0, 2.0, 1.5], [20.0, -4.0, 0.5]])
    report = check_alignment(coords, hemis)
    assert not report.suspicious
    assert report.median_offset_mm < 2.0
    assert report.n_electrodes == 3


def test_a_uniform_shift_is_caught(hemis):
    # What a missing c_ras correction looks like: every electrode off by the
    # same vector, which is invisible one electrode at a time.
    # Start on the pia, so the whole centroid shift is the injected error.
    coords = np.array([[-20.0, 0.0, 0.0], [-22.0, 2.0, 0.0]]) + np.array([0, 0, 50.0])
    report = check_alignment(coords, hemis)
    assert report.suspicious
    assert "SUSPICIOUS" in report.summary()
    assert np.isclose(report.systematic_shift_mm[2], 50.0, atol=1.0)
    assert np.isclose(report.shift_magnitude_mm, 50.0, atol=1.0)


def test_placeholder_rows_are_excluded_rather_than_propagated(hemis):
    """One NaN row used to turn every statistic in the report into NaN.

    Real montages carry these in quantity -- placeholder rows for unconnected
    amplifier channels -- so a report that cannot survive one is a report that
    says nothing about most real files. Measured on a clinical montage: 14 of
    100 rows, and the whole summary came back NaN.
    """
    coords = np.array([
        [-20.0, 0.0, 1.0],
        [np.nan, np.nan, np.nan],
        [-22.0, 2.0, 1.5],
        [20.0, -4.0, np.nan],
    ])
    report = check_alignment(coords, hemis)
    assert np.isfinite(report.median_offset_mm)
    assert np.isfinite(report.systematic_shift_mm).all()
    assert report.n_electrodes == 2 and report.n_skipped == 2
    assert "2 rows skipped" in report.summary()


def test_a_single_skipped_row_reads_as_singular(hemis):
    coords = np.array([[-20.0, 0.0, 1.0], [np.nan, 0.0, 0.0]])
    assert "1 row skipped" in check_alignment(coords, hemis).summary()


def test_a_montage_with_no_finite_coordinate_is_refused(hemis):
    """Nothing was measured, so there is no report to make."""
    with pytest.raises(ValueError, match="finite coordinate"):
        check_alignment(np.full((3, 3), np.nan), hemis)


def test_a_clean_montage_reports_nothing_skipped(hemis):
    report = check_alignment(np.array([[-20.0, 0.0, 1.0]]), hemis)
    assert report.n_skipped == 0
    assert "skipped" not in report.summary()


def test_the_surface_hash_tracks_the_surfaces(hemis):
    before = surface_hash(hemis)
    assert before == surface_hash(hemis)
    moved = dict(hemis)
    moved["lh"] = SurfacePair(
        pia=hemis["lh"].pia + 1.0, polys=hemis["lh"].polys, wm=hemis["lh"].wm
    )
    assert surface_hash(moved) != before


def test_anchors_carry_the_hash_of_what_built_them(hemis):
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, 0.0]]), hemis)
    assert anchors.surface_hash == surface_hash(hemis)


def test_summary_counts_every_outcome(hemis):
    coords = np.array([[-20.0, 0.0, -1.5], [-20.0, 0.0, 2.0], [60.0, 0.0, 0.0]])
    text = anchor_to_surfaces(coords, hemis).summary()
    assert "3 electrodes anchored" in text
    for outcome in (ON_SURFACE, PROJECTED, TOO_FAR):
        assert outcome in text


def test_bad_coordinate_shape_is_refused(hemis):
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        anchor_to_surfaces(np.zeros((4, 2)), hemis)
