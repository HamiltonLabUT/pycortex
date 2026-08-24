"""``add_electrodes`` on a real flatmap.

Assertions are on the marker collections rather than on pixels: what matters is
which electrodes were drawn, where, in what shape and what colour. The reference
images in ``test_quickflat.py`` cover the rendering itself.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import cortex
from cortex.electrodes import ElectrodeSet, PlacementPolicy, load_surface_pairs
from cortex.quickflat import add_electrodes
from cortex.quickflat.composite import MARKER_BY_GROUP_TYPE

SUBJECT = "S1"


@pytest.fixture(scope="module")
def eset():
    """Twelve contacts at known cortical depths, half a grid and half a shaft.

    Built from vertices rather than arbitrary points so the anchors, and
    therefore the flat positions, are exact.
    """
    pair = load_surface_pairs(SUBJECT)["lh"]
    pia, wm = pair.pia.astype(np.float64), pair.wm.astype(np.float64)
    flat_verts = np.unique(cortex.db.get_surf(SUBJECT, "flat", "lh", nudge=False)[1])
    idx = np.random.default_rng(1).choice(flat_verts, 12, replace=False)

    # Six on the pia (depth 0) and six halfway through the ribbon (depth 0.5),
    # which the anchor recovers to within about 0.02 -- see
    # test_electrodes_subject.test_depth_is_recovered_across_the_ribbon.
    coords = np.vstack([pia[idx[:6]], (pia[idx[6:]] + wm[idx[6:]]) / 2.0])
    eset = ElectrodeSet(
        names=["LG%d" % (i + 1) for i in range(6)] + ["LD%d" % (i + 1) for i in range(6)],
        coords=coords,
        subject=SUBJECT,
        group_type=["grid"] * 6 + ["seeg"] * 6,
        size=[2.3] * 6 + [0.8] * 6,
    )
    eset.anchor()
    return eset


@pytest.fixture
def flatfig():
    view = cortex.Vertex.random(SUBJECT)
    fig = cortex.quickflat.make_figure(view, with_rois=False, with_colorbar=False)
    yield fig
    plt.close(fig)


def _offsets(collections):
    return np.vstack([c.get_offsets() for c in collections.values()])


def _n_drawn(collections):
    return sum(len(c.get_offsets()) for c in collections.values())


# -- what gets drawn, and where --------------------------------------------

def test_every_electrode_is_drawn_by_default(flatfig, eset):
    collections = add_electrodes(flatfig, eset)
    assert _n_drawn(collections) == len(eset)


def test_markers_land_at_the_anchors_flat_positions(flatfig, eset):
    collections = add_electrodes(flatfig, eset, marker="o")
    expected = eset.positions("flat", nudge=True)[:, :2]
    assert np.allclose(collections["o"].get_offsets(), expected)


def test_markers_land_inside_the_flatmap(flatfig, eset):
    """A wrong extents convention would put every marker off the brain."""
    collections = add_electrodes(flatfig, eset)
    xy = _offsets(collections)
    xlim, ylim = flatfig.gca().get_xlim(), flatfig.gca().get_ylim()
    assert (xy[:, 0] > xlim[0]).all() and (xy[:, 0] < xlim[1]).all()
    assert (xy[:, 1] > ylim[0]).all() and (xy[:, 1] < ylim[1]).all()


def test_electrodes_sit_above_the_roi_layer(flatfig, eset):
    collections = add_electrodes(flatfig, eset)
    assert all(c.get_zorder() > 1000 for c in collections.values())


# -- shapes ----------------------------------------------------------------

def test_group_type_picks_the_marker(flatfig, eset):
    collections = add_electrodes(flatfig, eset)
    assert set(collections) == {MARKER_BY_GROUP_TYPE["grid"], MARKER_BY_GROUP_TYPE["seeg"]}
    assert len(collections[MARKER_BY_GROUP_TYPE["grid"]].get_offsets()) == 6


def test_one_marker_for_everything(flatfig, eset):
    collections = add_electrodes(flatfig, eset, marker="^")
    assert set(collections) == {"^"}


def test_a_marker_table_overrides_the_default(flatfig, eset):
    collections = add_electrodes(flatfig, eset, marker={"grid": "*"})
    assert set(collections) == {"*", "o"}          # seeg falls back to the default


# -- depth selection -------------------------------------------------------

def test_depth_accepts_a_range(flatfig, eset):
    """Counts come from the anchored depths, not from the nominal ones.

    The fixture places contacts at depth 0 and 0.5, but the anchor recovers
    those to a median of ~0.02 with a worst case near 0.35, so asserting
    "six in the band" would be testing anchor precision rather than the filter.
    Split at the midpoint of the depths actually present instead.
    """
    d = eset.depth
    lo, hi = float(d.min()), float(d.max())
    split = (lo + hi) / 2
    lower = int((d <= split).sum())

    assert _n_drawn(add_electrodes(flatfig, eset, depth=(lo - 0.01, hi + 0.01))) == len(eset)
    assert _n_drawn(add_electrodes(flatfig, eset, depth=(lo - 0.01, split))) == lower
    assert _n_drawn(add_electrodes(flatfig, eset, depth=(split, hi + 0.01))) == len(eset) - lower


def test_a_scalar_depth_is_a_band_of_depth_tol_either_side(flatfig, eset):
    d = eset.depth
    target = float(d[0])
    expected = int((np.abs(d - target) <= 0.1).sum())
    drawn = _n_drawn(add_electrodes(flatfig, eset, depth=target, depth_tol=0.1))
    assert drawn == expected >= 1


def test_a_depth_band_with_nothing_in_it_draws_nothing(flatfig, eset):
    assert add_electrodes(flatfig, eset, depth=(5.0, 6.0)) == {}


def test_unknown_depth_is_kept_rather_than_dropped(flatfig, eset):
    """A subject with no white-matter surface has NaN depths. Those electrodes
    cannot fail a depth test, so adding a depth argument must not make them
    silently vanish."""
    partial = eset[:]
    partial.anchors.depth = np.full(len(partial), np.nan)
    assert _n_drawn(add_electrodes(flatfig, partial, depth=0.5)) == len(partial)


# -- placement filtering ---------------------------------------------------

def test_rejected_electrodes_are_not_drawn(flatfig, eset):
    rejected = eset[:]
    rejected.reclassify(PlacementPolicy(max_offset_mm=-1.0))     # rejects everything
    assert add_electrodes(flatfig, rejected) == {}
    assert _n_drawn(add_electrodes(flatfig, rejected, placeable_only=False)) == len(eset)


def test_the_surface_distance_threshold_is_settable_per_figure(flatfig, eset):
    """How near cortex a contact must be to be drawn, as a figure argument.

    Half these contacts are *on* the pia and half are mid-ribbon, about 1.0 to
    1.7 mm from either bounding surface, so a threshold between the two splits
    the montage exactly in half. All twelve clear the 4 mm default.
    """
    assert _n_drawn(add_electrodes(flatfig, eset)) == len(eset)
    assert _n_drawn(
        add_electrodes(flatfig, eset, max_surface_distance_mm=np.inf)
    ) == len(eset)
    assert _n_drawn(add_electrodes(flatfig, eset, max_surface_distance_mm=0.5)) == 6


def test_a_contact_the_default_rejects_can_be_drawn_by_raising_it(flatfig, eset):
    """Two centimetres out along +z, which is what it takes on a real brain.

    These twelve contacts are scattered rather than contiguous, and cortex
    folds densely enough that most of them still find a bank a millimetre or
    two away after the shift. A contiguous grid does not -- see
    ``test_electrodes_subject.test_the_rule_catches_a_lifted_grid_but_not_scattered_contacts``.
    """
    lifted = eset[:]
    lifted.coords = lifted.coords + np.array([0.0, 0.0, 20.0])
    lifted.anchor()
    default = _n_drawn(add_electrodes(flatfig, lifted))
    assert 0 < default < len(lifted)
    assert _n_drawn(
        add_electrodes(flatfig, lifted, max_surface_distance_mm=np.inf)
    ) == len(lifted)


def test_drawing_a_figure_does_not_reclassify_the_set(flatfig, eset):
    """A threshold passed to one figure must not follow the set to the next one,
    or to a viewer. `placement` belongs to the anchor, not to a drawing."""
    before = list(eset.anchors.placement)
    add_electrodes(flatfig, eset, max_surface_distance_mm=0.001)
    assert list(eset.anchors.placement) == before


# -- values and colours ----------------------------------------------------

def test_values_are_colormapped(flatfig, eset):
    values = np.linspace(0, 1, len(eset))
    collections = add_electrodes(flatfig, eset, values=values, marker="o", cmap="viridis")
    coll = collections["o"]
    assert np.allclose(coll.get_array(), values)
    assert coll.norm.vmin == 0 and coll.norm.vmax == 1


def test_explicit_colour_limits_are_kept(flatfig, eset):
    collections = add_electrodes(flatfig, eset, values=np.zeros(len(eset)),
                                 marker="o", vmin=-2, vmax=7)
    assert (collections["o"].norm.vmin, collections["o"].norm.vmax) == (-2, 7)


def test_values_follow_their_electrodes_through_filtering(flatfig, eset):
    """The subtle one: values are given for the whole set, so they have to be
    subset by the same mask the positions are."""
    values = np.arange(len(eset), dtype=float)
    collections = add_electrodes(flatfig, eset, values=values, depth=0.5, depth_tol=0.1,
                                 marker="o")
    assert np.allclose(collections["o"].get_array(), values[6:])


def test_a_mismatched_values_length_is_refused(flatfig, eset):
    with pytest.raises(ValueError, match="values has 3 entries"):
        add_electrodes(flatfig, eset, values=np.zeros(3))


# -- sizes -----------------------------------------------------------------

def test_size_is_constant_by_default(flatfig, eset):
    collections = add_electrodes(flatfig, eset, size=50, marker="o")
    assert np.allclose(collections["o"].get_sizes(), 50)


def test_size_by_maps_a_field_onto_the_size_range(flatfig, eset):
    collections = add_electrodes(flatfig, eset, size_by="size", size_range=(10, 100),
                                 marker="o")
    sizes = collections["o"].get_sizes()
    assert np.isclose(sizes.min(), 10) and np.isclose(sizes.max(), 100)
    # the 0.8 mm seeg contacts are the small ones
    assert np.allclose(sizes[6:], 10)


# -- guardrails ------------------------------------------------------------

def test_electrodes_from_another_subject_are_refused(flatfig, eset):
    other = eset[:]
    other.subject = "SOMEONE_ELSE"
    with pytest.raises(ValueError, match="SOMEONE_ELSE"):
        add_electrodes(flatfig, other, subject=SUBJECT)


def test_something_that_is_not_an_electrode_set_is_refused(flatfig):
    with pytest.raises(TypeError, match="ElectrodeSet"):
        add_electrodes(flatfig, np.zeros((4, 3)))


def test_an_unanchored_set_anchors_itself(flatfig, eset):
    fresh = ElectrodeSet(eset.names, eset.coords, subject=SUBJECT)
    assert fresh.anchors is None
    assert _n_drawn(add_electrodes(flatfig, fresh)) == len(fresh)
    assert fresh.anchors is not None


# -- through make_figure ---------------------------------------------------

def test_make_figure_draws_them(eset):
    view = cortex.Vertex.random(SUBJECT)
    fig = cortex.quickflat.make_figure(
        view, with_rois=False, with_colorbar=False, with_electrodes=eset,
        electrode_kwargs=dict(marker="o"),
    )
    scatters = [c for c in fig.gca().collections if len(c.get_offsets()) == len(eset)]
    assert len(scatters) == 1
    plt.close(fig)


def test_make_figure_passes_values_and_kwargs_through(eset):
    view = cortex.Vertex.random(SUBJECT)
    fig = cortex.quickflat.make_figure(
        view, with_rois=False, with_colorbar=False, with_electrodes=eset,
        electrode_values=np.arange(len(eset), dtype=float),
        electrode_kwargs=dict(depth=0.5, depth_tol=0.1, marker="o"),
    )
    scatter = [c for c in fig.gca().collections if len(c.get_offsets()) == 6]
    assert len(scatter) == 1
    assert np.allclose(scatter[0].get_array(), np.arange(6, 12))
    plt.close(fig)


def test_make_figure_is_unchanged_when_the_flag_is_off(eset):
    view = cortex.Vertex.random(SUBJECT)
    fig = cortex.quickflat.make_figure(view, with_rois=False, with_colorbar=False)
    assert len(fig.gca().collections) == 0
    plt.close(fig)


# -- an Electrode view as the thing being drawn -----------------------------
#
# P2 made electrode data a real Dataview, so the two entry points below are
# what "plot my electrode data" now means. `add_electrodes` gains a view that
# carries its own values and colormap, and `make_figure` gains a branch for the
# case where electrode data is the *subject* of the figure rather than an
# annotation over somebody else's.


@pytest.fixture
def montage(eset, tmp_path, monkeypatch):
    """`eset`, saved as S1's native montage in a scratch filestore.

    A scratch one because these tests write, and the real filestore is checked
    into the repo. The surfaces still come from the real S1 -- the anchors were
    computed against them in `eset` and are carried in the file.
    """
    import os
    import shutil

    real = cortex.db.filestore
    for entry in os.listdir(os.path.join(real, SUBJECT)):
        src = os.path.join(real, SUBJECT, entry)
        dst = str(tmp_path / SUBJECT / entry)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)

    monkeypatch.setattr(cortex.db, "filestore", str(tmp_path))
    cortex.db.reload_subjects()
    cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)
    yield cortex.db
    cortex.db.reload_subjects()


@pytest.fixture
def view(montage, eset):
    return cortex.Electrode(
        np.linspace(0.0, 1.0, len(eset)), SUBJECT, "native",
        cmap="viridis", vmin=0.0, vmax=1.0,
    )


def test_a_view_supplies_its_own_values_and_colormap(flatfig, view):
    """Which is the point of the view existing: one object, not four arguments."""
    collections = add_electrodes(flatfig, view)
    drawn = [c for c in collections.values() if c.get_array() is not None]
    assert drawn, "markers should be colormapped, not flat-coloured"
    for coll in drawn:
        assert coll.cmap.name == "viridis"
        assert (coll.norm.vmin, coll.norm.vmax) == (0.0, 1.0)


def test_an_explicit_argument_still_beats_the_view(flatfig, view):
    collections = add_electrodes(flatfig, view, cmap="hot", vmax=0.5)
    coll = next(c for c in collections.values() if c.get_array() is not None)
    assert coll.cmap.name == "hot"
    assert coll.norm.vmax == 0.5
    assert coll.norm.vmin == 0.0     # the one not overridden still comes from the view


def test_a_view_and_its_set_draw_the_same_markers(flatfig, view, eset):
    from_view = _offsets(add_electrodes(flatfig, view))
    from_set = _offsets(add_electrodes(flatfig, eset, values=view.data))
    assert np.allclose(from_view, from_set)


def test_a_movie_view_draws_its_first_frame(flatfig, montage, eset):
    """As `add_data` takes the first frame of an image: a figure is one instant."""
    frames = np.tile(np.arange(float(len(eset))), (4, 1))
    collections = add_electrodes(flatfig, cortex.Electrode(frames, SUBJECT, "native"))
    # One collection per marker shape, so gather them all before comparing.
    drawn = np.concatenate([np.asarray(c.get_array()) for c in collections.values()])
    assert np.allclose(np.sort(drawn), frames[0])


def test_a_2d_view_brings_finished_colours_rather_than_a_scale(flatfig, montage, eset):
    """It has already colormapped itself, so there is nothing left to map.

    The markers therefore carry per-contact RGBA and no norm -- and so no
    colorbar, which is right: a 2D colormap has no one-dimensional scale to show.
    """
    n = len(eset)
    twod = cortex.Electrode2D(np.arange(float(n)), np.arange(float(n))[::-1],
                              SUBJECT, "native")
    collections = add_electrodes(flatfig, twod)
    assert all(c.get_array() is None for c in collections.values())
    faces = np.vstack([c.get_facecolors() for c in collections.values()])
    assert faces.shape[1] == 4
    assert len(np.unique(faces, axis=0)) > 1     # not one colour repeated


def test_something_that_is_neither_a_set_nor_a_view_is_rejected(flatfig):
    with pytest.raises(TypeError, match="ElectrodeSet or a cortex.Electrode"):
        add_electrodes(flatfig, "not electrodes")


# -- make_figure with electrode data as the primary view --------------------


def test_electrode_data_can_be_the_figure(view):
    """`make_figure(elec)` draws the contacts, with no `add_data` call.

    An `Electrode` is a `RenderableView` like any other, so `as_renderable`
    accepts it -- but its array is one value per contact, and handing that to
    `make_flatmap_image` would index the flatmap cache with a few hundred values.
    The branch is what turns that into the intended picture.
    """
    fig = cortex.quickflat.make_figure(view, with_rois=False)
    try:
        ax = fig.get_axes()[0]
        markers = [c for c in ax.collections if c.get_label() == "electrodes"]
        assert markers
        assert sum(len(c.get_offsets()) for c in markers) > 0
        assert next(c for c in markers if c.get_array() is not None).cmap.name == "viridis"
    finally:
        plt.close(fig)


def test_an_electrode_figure_draws_curvature_without_being_asked(view):
    """Markers on a blank page say nothing, so the default flips for this kind.

    `with_curvature` defaults to None -- "not asked about" -- rather than False,
    so this can differ by data kind while an explicit False still wins.
    """
    fig = cortex.quickflat.make_figure(view, with_rois=False)
    plain = cortex.quickflat.make_figure(view, with_rois=False, with_curvature=False)
    try:
        assert len(fig.get_axes()[0].get_images()) == 1
        assert len(plain.get_axes()[0].get_images()) == 0
    finally:
        plt.close(fig)
        plt.close(plain)


def test_an_electrode_figure_gets_a_colorbar_from_its_markers(view):
    """There being no image to take one from."""
    fig = cortex.quickflat.make_figure(view, with_rois=False, with_colorbar=True)
    bare = cortex.quickflat.make_figure(view, with_rois=False, with_colorbar=False)
    try:
        assert len(fig.get_axes()) == len(bare.get_axes()) + 1
    finally:
        plt.close(fig)
        plt.close(bare)


def test_other_data_still_draws_no_curvature_and_no_electrodes_by_default():
    """The two flipped defaults are scoped to electrode data and nothing else."""
    fig = cortex.quickflat.make_figure(
        cortex.Vertex.random(SUBJECT), with_rois=False, with_colorbar=False
    )
    try:
        ax = fig.get_axes()[0]
        assert [c for c in ax.collections if c.get_label() == "electrodes"] == []
        assert len(ax.get_images()) == 1        # the data layer, not curvature
    finally:
        plt.close(fig)


def test_a_png_of_electrode_data_is_written(view, tmp_path):
    """`make_png` forwards **kwargs, so it follows `make_figure` for free."""
    out = str(tmp_path / "elec.png")
    cortex.quickflat.make_png(out, view, with_rois=False)
    import os
    assert os.path.getsize(out) > 1000


# -- per-view marker vectors ------------------------------------------------

@pytest.fixture
def marked(eset):
    """The same montage, wrapped in a view that overrides its markers."""
    import cortex
    n = len(eset)
    cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)
    return cortex.Electrode(
        np.arange(float(n)), SUBJECT, "native",
        marker_size=np.linspace(1.0, 5.0, n),
        marker_shape=["cube"] * (n // 2) + ["diamond"] * (n - n // 2),
    )


def test_a_view_supplies_its_marker_shapes(flatfig, marked):
    """Shapes come from the view rather than from group_type -- contrast with
    test_group_type_picks_the_marker, which is the montage-only path."""
    collections = add_electrodes(flatfig, marked)
    assert set(collections) == {"s", "D"}          # cube, diamond


def test_an_explicit_marker_still_beats_the_view(flatfig, marked):
    collections = add_electrodes(flatfig, marked, marker="^")
    assert set(collections) == {"^"}


def test_a_view_supplies_its_marker_sizes(flatfig, marked):
    collections = add_electrodes(flatfig, marked, marker="o", size_range=(10, 100))
    sizes = collections["o"].get_sizes()
    # Normalised onto size_range, so the ends land on it and the ramp is monotone.
    assert np.isclose(sizes.min(), 10) and np.isclose(sizes.max(), 100)
    assert np.all(np.diff(sizes) > 0)


def test_size_by_still_beats_the_view(flatfig, marked):
    """Explicit argument wins, the same rule the rest of add_electrodes follows.

    The montage records two distinct diameters and the view's vector is a
    monotone ramp, so which one was used is legible in the number of distinct
    marker areas.
    """
    from_view = add_electrodes(flatfig, marked, marker="o")["o"].get_sizes()
    from_field = add_electrodes(flatfig, marked, marker="o",
                                size_by="size")["o"].get_sizes()

    assert len(set(np.round(from_view, 6))) == len(from_view)      # the ramp
    assert len(set(np.round(from_field, 6))) < len(from_field)     # the montage
    assert not np.allclose(from_view, from_field)


def test_view_markers_follow_filtering(flatfig, marked):
    """The vectors are montage-length like the values, so they take the same
    mask -- otherwise a filtered figure sizes the wrong contacts."""
    drawn = add_electrodes(flatfig, marked, marker="o", depth=(-10.0, 10.0))
    everything = add_electrodes(flatfig, marked, marker="o")
    assert len(drawn["o"].get_offsets()) <= len(everything["o"].get_offsets())


def test_a_view_without_markers_draws_what_it_always_did(flatfig, eset):
    import cortex
    cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)
    plain = cortex.Electrode(np.arange(float(len(eset))), SUBJECT, "native")
    collections = add_electrodes(flatfig, plain, size=40)
    assert set(collections) == {MARKER_BY_GROUP_TYPE["grid"],
                                MARKER_BY_GROUP_TYPE["seeg"]}
    for coll in collections.values():
        assert np.allclose(coll.get_sizes(), 40)


def test_the_shape_table_covers_the_vocabulary():
    """quickflat and the viewer must agree on what a shape name means."""
    from cortex.dataset.electrode_views import MARKER_SHAPES
    from cortex.quickflat.composite import MARKER_BY_SHAPE

    assert set(MARKER_SHAPES) <= set(MARKER_BY_SHAPE)
