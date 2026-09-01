"""Spreading a contact's value over the cortex around it.

Two halves, for the two things that can be wrong. The synthetic half runs on
flat sheets, where geodesic distance and straight-line distance are the same
number analytically -- so the kernel, the truncation and the two metrics can be
checked against arithmetic rather than against each other. The S1 half runs on
real folded cortex, where they are emphatically not the same number, and checks
the things only folding and a real transform exercise: that a blob stays in its
own hemisphere, that the geodesic footprint sits inside the euclidean one, and
that a voxel blob lands where the transform says it should.
"""

from __future__ import annotations

import numpy as np
import pytest

import cortex
from cortex.electrodes import (
    ElectrodeSet,
    surface_weights,
    total_weight,
    volume_weights,
    weighted_mean,
)

from .test_electrode_anchor import plane, plane_pair

SUBJECT = "S1"
SIGMA = 3.0
RADIUS = 9.0


def sheet_pair(x0, x1, n=41, span=20.0):
    """A finer sheet than the anchoring tests need, for measuring distances on."""
    pia, polys = plane(x0, x1, y0=-span, y1=span, n=n, z=0.0)
    wm, _ = plane(x0, x1, y0=-span, y1=span, n=n, z=-3.0)
    return type(plane_pair(0.0, 1.0))(pia=pia, polys=polys, wm=wm)


@pytest.fixture
def sheets():
    """Two flat sheets, left at negative x and right at positive, as in RAS."""
    return {"lh": sheet_pair(-40.0, 0.0), "rh": sheet_pair(0.0, 40.0)}


def one_contact(sheets, xyz=(-20.0, 0.0, 0.0)):
    return ElectrodeSet(["A1"], np.array([xyz], dtype=float))


# -- the kernel -------------------------------------------------------------

@pytest.mark.parametrize("metric", ["geodesic", "euclidean"])
def test_the_kernel_is_a_gaussian_of_the_distance(sheets, metric):
    """On a flat sheet the answer is arithmetic, so check it against arithmetic.

    Both metrics run: a plane is the one place geodesic distance *is* Euclidean
    distance, which is what makes this a test of the kernel rather than a
    comparison of the two paths with each other.
    """
    eset = one_contact(sheets)
    weights = surface_weights(
        eset, sigma=SIGMA, radius=RADIUS, metric=metric, surfaces=sheets
    )

    mid = (sheets["lh"].pia + sheets["lh"].wm) / 2.0
    foot = eset.anchors.evaluate({"lh": mid, "rh": mid})[0]
    distance = np.linalg.norm(mid - foot, axis=1)

    row = np.asarray(weights[0, : len(mid)].todense()).ravel()
    inside = distance <= RADIUS - 1e-9
    expected = np.exp(-distance[inside] ** 2 / (2 * SIGMA**2))

    # Euclidean is exact. Geodesic is not and cannot be: the heat method
    # (Crane et al. 2012) recovers distance from a diffusion, and on a 1 mm mesh
    # it comes in a few percent short, which shows up as a slightly fat blob.
    # The tolerance here is that discretization error, measured -- not slack.
    assert np.allclose(row[inside], expected,
                       atol=0.06 if metric == "geodesic" else 1e-12)


def test_the_kernel_peaks_at_one(sheets):
    """Peak-one is what makes coverage read as a count of contacts."""
    weights = surface_weights(one_contact(sheets), sigma=SIGMA, surfaces=sheets)
    assert weights.max() == pytest.approx(1.0, abs=1e-9)


def test_nothing_reaches_past_the_radius(sheets):
    eset = one_contact(sheets)
    weights = surface_weights(eset, sigma=SIGMA, radius=RADIUS, surfaces=sheets)

    mid = (sheets["lh"].pia + sheets["lh"].wm) / 2.0
    foot = eset.anchors.evaluate({"lh": mid, "rh": mid})[0]
    reached = np.flatnonzero(np.asarray(weights[0, : len(mid)].todense()).ravel())
    assert np.all(np.linalg.norm(mid[reached] - foot, axis=1) <= RADIUS + 1e-9)


def test_the_radius_defaults_to_three_sigma(sheets):
    """One knob, not two -- and a wider blob must not come back squared off."""
    eset = one_contact(sheets)
    assert (
        surface_weights(eset, sigma=SIGMA, surfaces=sheets).nnz
        == surface_weights(eset, sigma=SIGMA, radius=3 * SIGMA, surfaces=sheets).nnz
    )


@pytest.mark.parametrize("bad", [{"sigma": 0}, {"sigma": -1}, {"radius": 0}])
def test_a_kernel_with_no_width_is_refused(sheets, bad):
    with pytest.raises(ValueError):
        surface_weights(one_contact(sheets), surfaces=sheets, **bad)


def test_an_unknown_metric_is_refused(sheets):
    with pytest.raises(ValueError, match="geodesic"):
        surface_weights(one_contact(sheets), metric="taxicab", surfaces=sheets)


# -- the matrix's shape and indexing ----------------------------------------

def test_rows_stay_aligned_with_the_channels(sheets):
    """A contact that reaches nothing keeps its row, rather than shifting the rest.

    Montages carry placeholder rows for unconnected amplifier channels, and the
    data array is recorded on the same channels; a dropped row would silently
    re-index everything below it.
    """
    eset = ElectrodeSet(
        ["A1", "NaN1", "A2"],
        np.array([[-20.0, 0, 0], [np.nan, np.nan, np.nan], [-20.0, 8.0, 0]]),
    )
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)

    assert weights.shape[0] == 3
    assert weights[1].nnz == 0
    assert weights[0].nnz and weights[2].nnz


def test_the_vertex_axis_runs_left_then_right(sheets):
    """The column order every Vertex uses, so the result can be handed to one."""
    eset = ElectrodeSet(["L", "R"], np.array([[-20.0, 0, 0], [20.0, 0, 0]]))
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)
    llen = len(sheets["lh"].pia)

    assert list(eset.anchors.hemi) == ["lh", "rh"]
    assert weights[0, llen:].nnz == 0
    assert weights[1, :llen].nnz == 0
    assert weights.shape[1] == llen + len(sheets["rh"].pia)


def test_a_blob_stays_in_its_own_hemisphere(sheets):
    """The bug a merged both-hemisphere search has.

    The two medial walls come within a couple of millimetres of each other, so a
    search that does not know about hemispheres puts a left-hemisphere contact's
    value on the right hemisphere. Here the sheets meet at x = 0 and the contact
    sits 1 mm from the join, well inside the radius of the other sheet.
    """
    eset = ElectrodeSet(["A1"], np.array([[-1.0, 0.0, 0.0]]))
    weights = surface_weights(eset, sigma=SIGMA, radius=RADIUS, surfaces=sheets)

    assert eset.anchors.hemi[0] == "lh"
    assert weights[0, len(sheets["lh"].pia):].nnz == 0


# -- what gets left out -----------------------------------------------------

def test_a_contact_too_far_from_cortex_is_left_out(sheets):
    """And is left out by policy, not by the kernel running out of reach."""
    eset = ElectrodeSet(["A1", "A2"], np.array([[-20.0, 0, 0], [-20.0, 0, 50.0]]))
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)

    assert eset.anchors.placement[1] == "too_far"
    assert weights[1].nnz == 0
    assert weights[0].nnz


def test_placeable_false_spreads_every_contact_that_has_a_coordinate(sheets):
    eset = ElectrodeSet(["A1", "A2"], np.array([[-20.0, 0, 0], [-20.0, 0, 50.0]]))
    weights = surface_weights(eset, sigma=SIGMA, placeable=False, surfaces=sheets)
    assert weights[1].nnz


# -- the reductions ---------------------------------------------------------

def test_constant_values_come_back_unchanged(sheets):
    """The one assertion that catches every normalisation error at once.

    A weighted mean of values that are all 5.0 is 5.0, whatever the weights are,
    so anything wrong with the denominator shows up here.
    """
    eset = ElectrodeSet(["A%d" % i for i in range(4)],
                        np.array([[-20.0 + 4 * i, 0, 0] for i in range(4)]))
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)
    mean = weighted_mean(np.full(4, 5.0), weights)

    assert np.allclose(mean[np.isfinite(mean)], 5.0)
    assert np.isfinite(mean).any()


def test_halfway_between_two_contacts_is_halfway_between_their_values(sheets):
    eset = ElectrodeSet(["A1", "A2"], np.array([[-22.0, 0, 0], [-18.0, 0, 0]]))
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)

    mid = (sheets["lh"].pia + sheets["lh"].wm) / 2.0
    between = int(np.argmin(np.linalg.norm(mid - np.array([-20.0, 0.0, -1.5]), axis=1)))
    mean = weighted_mean(np.array([0.0, 1.0]), weights)
    assert mean[between] == pytest.approx(0.5, abs=0.02)


def test_a_dead_channel_does_not_blank_the_cortex_around_it(sheets):
    """NaN drops out of the mean rather than counting as zero.

    A NaN treated as data would pull the whole neighbourhood towards zero; a NaN
    that poisoned the sum would leave a hole. It should do neither: the answer is
    the mean over the contacts that did report.
    """
    eset = ElectrodeSet(["A1", "A2"], np.array([[-22.0, 0, 0], [-18.0, 0, 0]]))
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)

    with_nan = weighted_mean(np.array([np.nan, 1.0]), weights)
    alone = weighted_mean(np.array([1.0]), weights[1:])

    covered = np.isfinite(alone)
    assert covered.any()
    assert np.allclose(with_nan[covered], alone[covered])


def test_min_weight_decides_how_far_a_lone_value_is_carried(sheets):
    eset = one_contact(sheets)
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)

    counts = [np.isfinite(weighted_mean(np.array([1.0]), weights, mw)).sum()
              for mw in (0.001, 0.1, 0.5)]
    assert counts[0] > counts[1] > counts[2]


def test_coverage_counts_contacts(sheets):
    """One contact peaks at about one; two on the same spot at about two."""
    one = total_weight(surface_weights(one_contact(sheets), sigma=SIGMA, surfaces=sheets))
    pair = ElectrodeSet(["A1", "A2"], np.array([[-20.0, 0, 0], [-20.0, 0, 0.001]]))
    two = total_weight(surface_weights(pair, sigma=SIGMA, surfaces=sheets))

    assert one.max() == pytest.approx(1.0, abs=1e-6)
    assert two.max() == pytest.approx(2.0, abs=1e-3)
    assert one.min() == 0.0  # no contact nearby is an answer, not a hole


def test_a_movie_keeps_its_time_axis_in_front(sheets):
    """The vertex axis has to come last; that is what Vertex reads as a movie."""
    eset = ElectrodeSet(["A1", "A2"], np.array([[-22.0, 0, 0], [-18.0, 0, 0]]))
    weights = surface_weights(eset, sigma=SIGMA, surfaces=sheets)
    frames = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    movie = weighted_mean(frames, weights)
    assert movie.shape == (3, weights.shape[1])
    for t, frame in enumerate(frames):
        np.testing.assert_allclose(movie[t], weighted_mean(frame, weights), equal_nan=True)


# -- against the real subject ------------------------------------------------

@pytest.fixture(scope="module")
def s1_contacts():
    """Contacts sitting exactly on known mid-ribbon vertices of both hemispheres.

    Exact rather than approximate: the midpoint of a pial and a white-matter
    vertex *is* a vertex of the surface the blob is measured on, so the peak
    weight has a known location and a known value of exactly one.
    """
    pairs = cortex.electrodes.load_surface_pairs(SUBJECT)
    rng = np.random.default_rng(0)
    coords, verts, hemis = [], [], []
    for hemi in ("lh", "rh"):
        pair = pairs[hemi]
        idx = rng.choice(len(pair.pia), 3, replace=False)
        coords.append((pair.pia[idx] + pair.wm[idx]) / 2.0)
        verts.append(idx)
        hemis += [hemi] * 3
    eset = ElectrodeSet(
        ["%s%d" % (h, i) for h in "LR" for i in range(3)],
        np.vstack(coords), subject=SUBJECT,
    )
    llen = len(pairs["lh"].pia)
    columns = np.concatenate([verts[0], verts[1] + llen])
    return eset, columns, llen


@pytest.fixture(scope="module")
def s1_weights(s1_contacts):
    eset = s1_contacts[0]
    return surface_weights(eset, sigma=SIGMA, radius=RADIUS, subject=SUBJECT)


def test_on_real_cortex_each_blob_peaks_on_its_own_contact(s1_contacts, s1_weights):
    _, columns, _ = s1_contacts
    for row, column in enumerate(columns):
        assert s1_weights[row, column] == pytest.approx(1.0, abs=1e-9)
        assert s1_weights[row].max() == pytest.approx(1.0, abs=1e-9)


def test_on_real_cortex_a_blob_stays_in_its_own_hemisphere(s1_contacts, s1_weights):
    eset, _, llen = s1_contacts
    left = np.flatnonzero(eset.anchors.hemi == "lh")
    right = np.flatnonzero(eset.anchors.hemi == "rh")
    assert len(left) and len(right)
    assert s1_weights[left][:, llen:].nnz == 0
    assert s1_weights[right][:, :llen].nnz == 0


def test_the_geodesic_footprint_sits_inside_the_euclidean_one(s1_contacts, s1_weights):
    """Geodesic distance is never shorter than straight-line distance.

    So on folded cortex the geodesic blob must reach a subset of the vertices the
    euclidean one does -- and a strictly smaller one, since folding brings
    vertices near through space that are far across the sheet. This is the single
    best check that the geodesic path is doing anything at all: a degenerate
    solve returning zeros would fail it immediately by reaching *more*.
    """
    eset = s1_contacts[0]
    euclidean = surface_weights(
        eset, sigma=SIGMA, radius=RADIUS, metric="euclidean", subject=SUBJECT
    )
    geodesic_reach = (s1_weights != 0)
    euclidean_reach = (euclidean != 0)

    assert (geodesic_reach > euclidean_reach).nnz == 0, "geodesic reached past euclidean"
    assert geodesic_reach.nnz < euclidean_reach.nnz, "folding should cost the geodesic blob vertices"


def test_a_vertex_the_solver_could_not_reach_gets_no_weight(s1_contacts, s1_weights):
    """`geodesic_distance` fills unsolved rows with *zero*, not infinity.

    Unguarded, such a vertex takes the full weight of every blob -- anywhere on
    the hemisphere. The guard is that straight-line distance bounds geodesic
    distance from below, so nothing beyond the radius through space can be within
    it across the surface.
    """
    eset, _, llen = s1_contacts
    left, _ = cortex.db.get_surf(SUBJECT, "fiducial", "both", nudge=False)
    feet = eset.anchors.evaluate(
        dict(zip(("lh", "rh"), [s[0] for s in
                                cortex.db.get_surf(SUBJECT, "fiducial", "both", nudge=False)]))
    )
    for row in np.flatnonzero(eset.anchors.hemi == "lh"):
        reached = s1_weights[row, :llen].indices
        assert np.all(np.linalg.norm(left[0][reached] - feet[row], axis=1) <= RADIUS + 1e-9)


def test_the_view_gives_back_an_ordinary_vertex(s1_contacts):
    eset = s1_contacts[0]
    elec = cortex.Electrode(np.arange(6.0), SUBJECT, "native",
                            electrodes=eset, cmap="viridis")
    vertex = elec.to_vertex(sigma=SIGMA, radius=RADIUS)

    assert isinstance(vertex, cortex.Vertex)
    assert vertex.subject == SUBJECT
    assert vertex.data.shape == (cortex.db.get_surf(SUBJECT, "fiducial", merge=True)[0].shape[0],)
    assert not vertex.movie
    # The mean lies between the contacts' own bounds, so the view's bounds are
    # the field's bounds -- and the blobs colour-match the markers drawn on top.
    assert vertex.cmap == "viridis"
    assert (vertex.vmin, vertex.vmax) == (elec.vmin, elec.vmax)
    covered = np.isfinite(vertex.data)
    assert covered.any()
    assert vertex.data[covered].min() == pytest.approx(0.0, abs=1e-9) or vertex.data[covered].min() > 0
    assert vertex.data[covered].max() <= 5.0 + 1e-9


def test_coverage_is_its_own_map_not_the_data(s1_contacts):
    eset = s1_contacts[0]
    elec = cortex.Electrode(np.arange(6.0), SUBJECT, "native", electrodes=eset)
    coverage = elec.coverage(sigma=SIGMA, radius=RADIUS)

    assert coverage.vmin == 0
    assert not np.isnan(coverage.data).any()
    assert coverage.data.max() == pytest.approx(1.0, abs=1e-6)


# -- the volume path ---------------------------------------------------------

def test_a_voxel_blob_lands_where_the_transform_says(s1_contacts):
    """The assertion that pins the whole volumetric coordinate path.

    Electrode coordinates are TkRegRAS; `surface_space_offset` carries them into
    the space the subject's surfaces are in; and a pycortex transform is defined
    against those same surfaces, so that is the whole conversion. Getting it
    wrong -- by going via the anatomical's scanner affine, say -- moves the blob
    by tens of millimetres, which is what this catches.
    """
    from cortex.electrodes import surface_space_offset

    eset = s1_contacts[0]
    xfm = cortex.db.get_xfm(SUBJECT, "fullhead")
    coverage = total_weight(
        volume_weights(eset[[0]], "fullhead", sigma=SIGMA, radius=RADIUS, subject=SUBJECT)
    ).reshape(xfm.shape)

    x, y, z = xfm(eset.coords[:1] + surface_space_offset(SUBJECT))[0]
    assert np.unravel_index(np.argmax(coverage), coverage.shape) == (
        int(round(z)), int(round(y)), int(round(x)),
    )


def test_a_voxel_blob_is_round_in_the_head_not_in_the_array(s1_contacts):
    """S1's `fullhead` grid is 2.24 x 2.24 x 4.13 mm, so the two differ.

    Measuring in voxels instead of millimetres would give a footprint that is
    isotropic in voxel counts and squashed in the head. This asserts the reverse:
    anisotropic in voxels, isotropic in millimetres.
    """
    eset = s1_contacts[0]
    xfm = cortex.db.get_xfm(SUBJECT, "fullhead")
    coverage = total_weight(
        volume_weights(eset[[0]], "fullhead", sigma=SIGMA, radius=RADIUS, subject=SUBJECT)
    ).reshape(xfm.shape)

    zs, ys, xs = np.nonzero(coverage)
    extent_voxels = np.array([np.ptp(xs), np.ptp(ys), np.ptp(zs)]) + 1
    voxel_mm = np.linalg.norm(np.linalg.inv(np.asarray(xfm.xfm)[:3, :3]), axis=0)
    extent_mm = extent_voxels * voxel_mm

    assert extent_voxels[2] < extent_voxels[0]           # fewer, fatter slices
    assert np.ptp(extent_mm) / extent_mm.mean() < 0.15   # but the same size in mm


def test_the_volume_view_matches_its_transform(s1_contacts):
    eset = s1_contacts[0]
    elec = cortex.Electrode(np.arange(6.0), SUBJECT, "native", electrodes=eset)
    volume = elec.to_volume("fullhead", sigma=SIGMA, radius=RADIUS)

    assert isinstance(volume, cortex.Volume)
    assert volume.data.shape == cortex.db.get_xfm(SUBJECT, "fullhead").shape
    assert not volume.linear
    assert np.isfinite(volume.data).any()


def test_a_mask_gives_back_a_masked_volume(s1_contacts):
    eset = s1_contacts[0]
    elec = cortex.Electrode(np.arange(6.0), SUBJECT, "native", electrodes=eset)
    shape = cortex.db.get_xfm(SUBJECT, "fullhead").shape

    reached = np.isfinite(elec.to_volume("fullhead", sigma=SIGMA, radius=RADIUS).data)
    volume = elec.to_volume("fullhead", sigma=SIGMA, radius=RADIUS, mask=reached)

    assert volume.linear
    assert volume.data.shape == (int(reached.sum()),)
    assert np.isfinite(volume.data).all()


def test_a_mask_of_the_wrong_shape_is_refused(s1_contacts):
    with pytest.raises(ValueError, match="shape"):
        volume_weights(s1_contacts[0], "fullhead", sigma=SIGMA,
                       subject=SUBJECT, mask=np.ones((3, 3, 3), bool))
