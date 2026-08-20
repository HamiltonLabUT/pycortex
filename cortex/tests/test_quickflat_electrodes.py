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
