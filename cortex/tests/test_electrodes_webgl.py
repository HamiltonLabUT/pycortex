"""Electrode markers in the webgl viewer.

The assertion that matters here is a *distance*, not a pixel count. Markers on
the wrong vertices still land on the surface and still track the morph -- they
are simply in the wrong places, scattered over the hemisphere. That renders
without error and photographs plausibly, so the only thing that catches it is
measuring a spacing whose right answer is known independently: the contacts of
an 8x8 grid at 8 mm pitch are ~7 mm apart on cortex, and if the vertex indices
are wrong they come out at ~50 mm.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

import cortex
from cortex.electrodes import ElectrodeSet, load_surface_pairs
from cortex.electrodes._webgl import ctm_vertex_index, to_viewer_json
from cortex.tests.testing_utils import has_playwright

SUBJECT = "S1"
PITCH = 8.0


def build_grid(subject=SUBJECT, n=8, pitch=PITCH):
    """An n-by-n subdural grid draped over lateral temporal cortex.

    Contacts are conformed to the folded surface, so their spacing is set by the
    cortex rather than by the plane they started on -- which is what makes the
    measured spacing a meaningful check rather than a restatement of the input.
    """
    pair = load_surface_pairs(subject)["lh"]
    pia, wm = pair.pia.astype(np.float64), pair.wm.astype(np.float64)
    column = pia - wm
    column /= np.linalg.norm(column, axis=1)[:, None]

    near = np.flatnonzero(np.linalg.norm(pia - np.array([-62.0, -10.0, 0.0]), axis=1) < 25)
    seed = int(near[np.argmax(column[near] @ np.array([-1.0, 0.0, 0.0]))])
    out = column[seed]
    u = np.cross(out, [0, 0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(out, u)

    planar = np.array([
        pia[seed] + (i - (n - 1) / 2) * pitch * u + (j - (n - 1) / 2) * pitch * v + 6 * out
        for i in range(n) for j in range(n)
    ])
    draped = ElectrodeSet(["g%d" % i for i in range(n * n)], planar, subject=subject)
    draped.anchor(surfaces={"lh": pair})
    coords = draped.anchors.evaluate({"lh": pia})

    eset = ElectrodeSet(
        names=["G%d" % (i + 1) for i in range(n * n)], coords=coords,
        subject=subject, group_type=["grid"] * (n * n),
    )
    eset.anchor()
    return eset


def _unwrap_list(value):
    """Peel the python bridge's wrapping off a list it returned.

    Every value comes back inside a one-element list, and an array comes back
    with that wrapping nested. A genuine one-element result survives, because
    peeling stops as soon as the single element is not itself a list.
    """
    while (isinstance(value, list) and len(value) == 1
           and isinstance(value[0], list)):
        value = value[0]
    return value


def row_spacing(positions, n=8):
    """Distances between neighbouring contacts along each row of the grid."""
    positions = np.asarray(positions)
    return np.array([
        np.linalg.norm(positions[i * n + j] - positions[i * n + j + 1])
        for i in range(n) for j in range(n - 1)
    ])


# -- the python half, which needs no browser -------------------------------

@pytest.fixture(scope="module")
def ctmfile():
    return cortex.utils.get_ctmpack(SUBJECT, ("inflated",), method="mg2", level=9)


def test_the_ctm_reorder_is_a_permutation(ctmfile):
    index = ctm_vertex_index(ctmfile)
    assert len(np.unique(index)) == len(index)
    assert index.min() == 0 and index.max() == len(index) - 1


def test_the_ctm_reorder_keeps_the_hemispheres_apart(ctmfile):
    """Which is what lets a per-hemisphere index be offset into the merged
    permutation and back out again."""
    index = ctm_vertex_index(ctmfile)
    left_n = len(cortex.db.get_surf(SUBJECT, "pia", "lh")[0])
    assert index[:left_n].max() < left_n
    assert index[left_n:].min() >= left_n


def test_the_reorder_actually_moves_vertices(ctmfile):
    """If this ever became the identity the bug it guards against would be
    invisible, because sending pycortex indices unconverted would start working
    by accident."""
    index = ctm_vertex_index(ctmfile)
    assert (index != np.arange(len(index))).mean() > 0.5


def test_serialised_vertices_are_converted(ctmfile):
    eset = build_grid()
    index = ctm_vertex_index(ctmfile)
    payload = to_viewer_json(eset, ctm_index=index)
    sent = np.array([c["verts"] for c in payload["electrodes"]])
    assert sent.shape == (len(eset), 3)
    assert not np.array_equal(sent, eset.anchors.verts)      # genuinely converted
    assert sent.min() >= 0


def test_serialisation_needs_anchors():
    eset = ElectrodeSet(["A1"], np.zeros((1, 3)), subject=SUBJECT)
    with pytest.raises(ValueError, match="anchored"):
        to_viewer_json(eset)


def test_make_static_embeds_the_electrodes(tmp_path):
    """`make_static` rewrites its `ctms` values to bundle-relative names, and
    under `anonymize` renames the keys too, so the electrode payload has to be
    built before that happens or the CTM permutation is unreachable. Only the
    static path exercises this -- `show` keeps real paths throughout."""
    eset = build_grid()
    out = str(tmp_path / "bundle")
    cortex.webgl.make_static(out, cortex.Vertex.random(SUBJECT),
                             electrodes=eset, overlays_visible=())

    with open(os.path.join(out, "index.html")) as page:
        html = page.read()
    assert "electrodes.Electrodes" in html            # the module was inlined
    assert '"electrodes"' in html                     # ...and given a payload
    assert eset.names[0] in html                      # ...naming real contacts


# -- the browser half ------------------------------------------------------

@pytest.mark.skipif(not has_playwright, reason="playwright and chromium are required")
class TestMarkersInTheViewer:
    """One headless session, several assertions."""

    @pytest.fixture(autouse=True, scope="class")
    def _viewer(self, request):
        eset = build_grid()
        cls = request.cls
        cls.eset = eset
        cls.expected = row_spacing(eset.positions("fiducial", nudge=False))
        with cortex.export.headless_viewer(
            cortex.Vertex.random(SUBJECT), viewer_params=dict(electrodes=eset)
        ) as handle:
            time.sleep(3)
            cls.handle = handle
            reported = handle.surfs[0].surf.electrodes.describe()
            while isinstance(reported, list) and len(reported) == 1 and isinstance(reported[0], list):
                reported = reported[0]
            cls.reported = reported
            yield

    def test_every_contact_is_drawn(self):
        assert len(type(self).reported) == len(type(self).eset)

    def test_markers_land_on_the_right_vertices(self):
        """The regression this file exists for.

        Vertex order is shuffled twice between python and the browser -- the
        CTM's spatial sort, then three.js's chunk-local one -- and applying only
        one of them scatters the grid across the hemisphere while still drawing
        it neatly on the surface.
        """
        drawn = row_spacing([c["position"] for c in type(self).reported])
        expected = type(self).expected
        assert np.median(drawn) == pytest.approx(np.median(expected), abs=1.0)
        assert drawn.max() < 3 * np.median(expected)

    def test_the_grid_stays_a_grid_and_not_a_scatter(self):
        positions = np.array([c["position"] for c in type(self).reported])
        extent = positions.max(0) - positions.min(0)
        # A 64-contact grid at 8 mm pitch spans well under half a hemisphere.
        assert extent.max() < 90.0

    def test_at_unfold_zero_markers_sit_at_their_true_coordinates(self):
        """On the anatomical surface the measured coordinate is the truth.

        A depth contact belongs inside the brain; snapping it to the nearest
        gyrus would draw a position it does not have. The anchor only takes
        over once the surface deforms and the coordinate stops meaning
        anything.
        """
        drawn = np.array([c["position"] for c in type(self).reported])
        assert np.allclose(drawn, type(self).eset.coords, atol=1e-3)

    def test_the_depth_window_follows_the_sampled_surface(self):
        """A deformed surface shows one depth through the ribbon at a time.

        A contact that is not at that depth is not on the sheet being drawn, so
        it is hidden rather than projected onto a surface it is nowhere near --
        the same reasoning as the placement policy. Which contacts survive has
        to change when the depth slider moves from pia to white matter; if it
        does not, the slider is writing the uniform without telling anything
        that positions itself on the CPU, which is exactly the bug this pins.
        """
        handle = type(self).handle
        handle.ui.set("surface.S1.unfold", 0.5)
        time.sleep(1.5)

        def visible_at(thickmix):
            handle.ui.set("surface.S1.depth", thickmix)
            time.sleep(1.5)
            reported = handle.surfs[0].surf.electrodes.describe()
            while (isinstance(reported, list) and len(reported) == 1
                   and isinstance(reported[0], list)):
                reported = reported[0]
            return {c["name"] for c in reported if c["visible"]}

        at_pia, at_wm = visible_at(0.0), visible_at(1.0)
        handle.ui.set("surface.S1.unfold", 0.0)
        time.sleep(1.5)

        # A grid sits at the pia, so sampling there shows it and sampling the
        # white matter does not.
        assert at_pia, "nothing visible when sampling the pial surface"
        assert at_pia != at_wm, "the depth slider did not change what is drawn"
        assert len(at_wm) < len(at_pia)

    def test_channel_name_labels_toggle(self):
        """Off by default, and only shown for markers that are themselves
        visible -- otherwise a filtered-out contact leaves its name floating
        over nothing."""
        e = type(self).handle.surfs[0].surf.electrodes
        assert e.setLabels() in (False, [False])
        e.setLabels(True)
        time.sleep(1)
        assert e.setLabels() in (True, [True])
        e.setLabels(False)
        time.sleep(0.5)

    def test_hovering_a_contact_shows_that_one_label(self):
        """Hover works with the global labels toggle off, one at a time.

        Aimed by projecting a contact to screen coordinates and hit-testing
        there, so this exercises the real raycast rather than poking the
        hovered contact in directly.
        """
        e = type(self).handle.surfs[0].surf.electrodes

        def unwrap(v):
            # The python bridge returns every value wrapped in a one-element
            # list, and nests that wrapping for arrays. Peel until it is not a
            # singleton -- a genuine [x, y] pair has length two and survives.
            while isinstance(v, list) and len(v) == 1:
                v = v[0]
            return v

        xy = unwrap(e.projectContact(0))
        name = unwrap(e.pickNameAt(xy[0], xy[1]))
        assert name, "raycast found nothing where a contact was projected"

        assert unwrap(e.hoverAt(xy[0], xy[1])) == name
        time.sleep(0.5)
        # ...and pointing at empty space clears it.
        assert unwrap(e.hoverAt(-0.98, -0.98)) == ""

    def test_contacts_on_the_same_device_are_joined_by_a_line(self):
        """The lattice, drawn.

        An 8x8 grid has 2*8*7 edge neighbours, and a wrong lattice does not hit
        that number by accident: too many means the diagonals crept in and the
        grid reads as a sheet of triangles, too few means it was torn where the
        cortex stretched it.
        """
        e = type(self).handle.surfs[0].surf.electrodes
        edges = _unwrap_list(e.describeEdges())
        assert len(edges) == 2 * 8 * 7

        names = {c["name"] for c in type(self).reported}
        assert all(edge["a"] in names and edge["b"] in names for edge in edges)
        assert all(edge["a"] != edge["b"] for edge in edges)

    def test_the_lines_join_contacts_that_are_actually_adjacent(self):
        """Not contacts on opposite sides of the grid.

        Same regression as the markers themselves: the edges are indices into
        the contact list, so an off-by-one anywhere between python and here
        draws a neat mesh over the wrong pairs. A segment has to be about as
        long as the within-row spacing measured independently.
        """
        e = type(self).handle.surfs[0].surf.electrodes
        drawn = np.array([edge["length"] for edge in _unwrap_list(e.describeEdges())])
        expected = type(self).expected
        assert np.median(drawn) == pytest.approx(np.median(expected), abs=1.0)
        # A diagonal would be sqrt(2) spacings long, and a wild pair much more.
        assert drawn.max() < 3 * np.median(expected)

    def test_the_connection_lines_toggle(self):
        """On by default: which device a contact belongs to is the first thing
        a montage has to say, so it should not need turning on."""
        e = type(self).handle.surfs[0].surf.electrodes
        assert e.setConnections() in (True, [True])
        e.setConnections(False)
        time.sleep(1)
        assert e.setConnections() in (False, [False])
        e.setConnections(True)
        time.sleep(0.5)

    def test_the_page_raises_no_errors(self):
        errors = [e for e in type(self).handle._pw_thread.browser_errors
                  if "[pageerror]" in e]
        assert errors == [], "JS errors: %s" % errors

    @pytest.mark.parametrize("unfold", [0.0, 0.5, 1.0])
    def test_it_renders_at_every_unfold_state(self, unfold, tmp_path):
        from cortex.export.save_views import angle_view_params, default_view_params

        handle = type(self).handle
        params = dict(default_view_params)
        params.update(angle_view_params["flatmap" if unfold == 1.0 else "left"])
        params["surface.{subject}.unfold"] = unfold
        handle._set_view(**params)
        time.sleep(2)

        out = str(tmp_path / ("unfold_%s.png" % unfold))
        handle.getImage(out, (500, 400))
        for _ in range(300):
            if os.path.exists(out) and os.path.getsize(out) > 0:
                break
            time.sleep(0.1)
        assert os.path.getsize(out) > 0


# -- markers coloured from data --------------------------------------------
#
# Colour is asserted on pixels rather than on `describe()`, because a colour
# that never reaches the framebuffer is not a colour. A screenshot also catches
# the failure this whole file exists for -- markers drawn somewhere plausible
# but wrong -- in the one case where the grid is easiest to read.
#
# The markers are found by saturation: the cortex renders as grey and the page
# background is white, so anything with a spread between its RGB channels is a
# marker. That holds for every colormap the viewer ships except a greyscale one.


def _marker_pixels(path):
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), dtype=int)
    return img[(img.max(2) - img.min(2)) > 40]


@pytest.mark.skipif(not has_playwright, reason="playwright and chromium are required")
class TestMarkerColoursFollowTheData:
    """One headless session showing an Electrode movie, several assertions."""

    @classmethod
    def _shot(cls, name):
        out = os.path.join(cls.tmpdir, name)
        if os.path.exists(out):
            os.remove(out)
        cls.handle.getImage(out, (700, 550))
        for _ in range(300):
            if os.path.exists(out) and os.path.getsize(out) > 0:
                break
            time.sleep(0.1)
        time.sleep(0.3)
        return _marker_pixels(out)

    @pytest.fixture(autouse=True, scope="class")
    def _viewer(self, request, tmp_path_factory):
        from cortex.export.save_views import angle_view_params, default_view_params

        cls = request.cls
        cls.tmpdir = str(tmp_path_factory.mktemp("shots"))

        eset = build_grid()
        eset.anchor()
        # A scratch filestore, since a montage has to be saved for `Electrode`
        # to find it and the real one is checked into the repo.
        store = str(tmp_path_factory.mktemp("store"))
        real = cortex.db.filestore
        import shutil
        shutil.copytree(os.path.join(real, SUBJECT), os.path.join(store, SUBJECT))
        old = cortex.db.filestore
        cortex.db.filestore = store
        cortex.db.reload_subjects()
        cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)

        n = len(eset)
        # Two frames that run the ramp in opposite directions, so a frame change
        # has to alter what is on screen.
        frames = np.vstack([np.linspace(0, 1, n), np.linspace(1, 0, n)])
        view = cortex.Electrode(frames, SUBJECT, "native",
                                cmap="viridis", vmin=0.0, vmax=1.0)
        cls.n = n

        with cortex.export.headless_viewer(view, viewer_params={}) as handle:
            time.sleep(4)
            cls.handle = handle
            params = dict(default_view_params)
            params.update(angle_view_params["left"])
            params["surface.{subject}.unfold"] = 0.0
            handle._set_view(**params)
            time.sleep(2)
            cls.baseline = cls._shot("frame0.png")
            yield

        cortex.db.filestore = old
        cortex.db.reload_subjects()

    def test_the_page_raises_no_errors(self):
        errors = [e for e in type(self).handle._pw_thread.browser_errors
                  if "[pageerror]" in e]
        assert errors == [], "JS errors: %s" % errors

    def test_the_markers_span_the_colormap(self):
        """Rather than all sharing one flat colour, as P3a's did.

        A ramp through viridis is purple at one end and yellow at the other, so
        the drawn pixels have to vary in every channel. A single flat colour --
        which is what a failure to bind the data looks like -- would give a
        standard deviation near zero.
        """
        pixels = type(self).baseline
        assert len(pixels) > 100, "no coloured markers were drawn"
        assert pixels.std(0).min() > 15, (
            "marker colours barely vary (%s); they are probably still flat"
            % pixels.std(0)
        )

    def test_the_data_layer_is_not_drawn_on_the_cortex(self):
        """Electrode data has no per-vertex array, so the cortex stays curvature.

        `ElectrodeData.set` dispatches no attribute event, which leaves the
        surface's data attributes at the empty arrays it was initialised with;
        the shader reads `nanmask` as 0 there and discards the data layer. If
        that ever stopped holding, the cortex would come back covered in
        whatever the empty buffer decoded to, and these grey pixels would be
        coloured.
        """
        from PIL import Image

        img = np.asarray(
            Image.open(os.path.join(type(self).tmpdir, "frame0.png")).convert("RGB"),
            dtype=int,
        )
        sat = img.max(2) - img.min(2)
        # The cortex is the large mid-grey region; markers are a few hundred px.
        greyish = ((sat <= 40) & (img.mean(2) > 40) & (img.mean(2) < 220)).sum()
        assert greyish > 20 * (sat > 40).sum(), "the cortex looks coloured, not grey"

    def test_scrubbing_the_movie_recolours_the_markers(self):
        cls = type(self)
        cls.handle.setFrame(1)
        time.sleep(1.5)
        moved = cls._shot("frame1.png")
        cls.handle.setFrame(0)
        time.sleep(1.0)
        assert not np.allclose(cls.baseline.mean(0), moved.mean(0), atol=6), (
            "frame 1 looks identical to frame 0 (%s vs %s)"
            % (cls.baseline.mean(0), moved.mean(0))
        )

    def test_the_vminvmax_sliders_recolour_the_markers(self):
        """The property ELECTRODES_DESIGN.md's decision 3 is really about.

        Clipping the range to its bottom fifth saturates almost every contact at
        the top of the colormap, so the markers go overwhelmingly yellow -- a
        change no amount of luck would produce from a static colour.
        """
        cls = type(self)
        cls.handle.setVminmax(0.0, 0.2)
        time.sleep(1.5)
        try:
            clipped = cls._shot("clipped.png")
            assert not np.allclose(cls.baseline.mean(0), clipped.mean(0), atol=6)
            red, green, blue = clipped.mean(0)
            assert red > 150 and green > 150 and blue < 120, (
                "clipping to the low end should drive viridis to yellow, got %s"
                % clipped.mean(0)
            )
        finally:
            cls.handle.setVminmax(0.0, 1.0)
            time.sleep(1.0)


# -- what the viewer is handed ---------------------------------------------


def test_an_electrode_view_supplies_its_own_markers(tmp_path, monkeypatch):
    """`cortex.webshow(elec)` needs no second `electrodes=` argument.

    Requiring one would be a way to get the drawn markers and the drawn values
    out of step with each other.
    """
    from cortex.webgl.view import _electrodes_from

    store = str(tmp_path)
    import shutil
    shutil.copytree(os.path.join(cortex.db.filestore, SUBJECT),
                    os.path.join(store, SUBJECT))
    monkeypatch.setattr(cortex.db, "filestore", store)
    cortex.db.reload_subjects()

    eset = build_grid()
    eset.anchor()
    cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)
    view = cortex.Electrode(np.arange(float(len(eset))), SUBJECT, "native")

    assert list(_electrodes_from(view, None).names) == list(eset.names)
    assert list(_electrodes_from(cortex.Dataset(x=view), None).names) == list(eset.names)
    # ...and an explicit set still wins, which is how P3a drew a montage over
    # somebody else's data.
    other = eset.select(group_type="grid")[:4]
    assert len(_electrodes_from(view, other)) == 4

    # A dataset with no electrode views says nothing about electrodes.
    assert _electrodes_from(cortex.Vertex.random(SUBJECT), None) is None


def test_a_dataset_over_two_montages_refuses_to_guess(tmp_path, monkeypatch):
    """One set of markers is drawn, so two montages have no single answer."""
    from cortex.webgl.view import _electrodes_from

    store = str(tmp_path)
    import shutil
    shutil.copytree(os.path.join(cortex.db.filestore, SUBJECT),
                    os.path.join(store, SUBJECT))
    monkeypatch.setattr(cortex.db, "filestore", store)
    cortex.db.reload_subjects()

    eset = build_grid()
    eset.anchor()
    cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)
    cortex.db.save_montage(SUBJECT, eset, "research", overwrite=True)

    values = np.arange(float(len(eset)))
    both = cortex.Dataset(
        a=cortex.Electrode(values, SUBJECT, "native"),
        b=cortex.Electrode(values, SUBJECT, "research"),
    )
    with pytest.raises(ValueError, match="different montages"):
        _electrodes_from(both, None)
