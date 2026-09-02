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


def build_shaft(subject=SUBJECT, n=15, pitch=4.0):
    """A depth electrode: contacts at a uniform pitch along one straight track.

    The seed enters laterally so the shaft stays inside its own hemisphere -- a
    trajectory driven inward from a medially-facing vertex crosses the midline,
    which no real one does and which splits the device in two.
    """
    from cortex import polyutils

    pair = load_surface_pairs(subject)["lh"]
    normals = np.asarray(
        polyutils.Surface(pair.pia.astype(np.float64), pair.polys).vertex_normals
    )
    seed = 84217
    coords = pair.pia[seed] - np.outer(np.arange(n) * pitch, normals[seed])

    eset = ElectrodeSet(
        names=["LTD%d" % (i + 1) for i in range(n)],
        coords=coords,
        subject=subject,
        group=["LTD"] * n,
        group_type=["seeg"] * n,
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

        Markers are drawn exactly there and never stood off, so a depth
        contact is inside the brain because that is where it is. Turn
        `surface_opacity` down to see those; do not move the marker.
        """
        drawn = np.array([c["position"] for c in self._describe()])
        assert np.allclose(drawn, type(self).eset.coords, atol=1e-3)

    def _describe(self):
        reported = type(self).handle.surfs[0].surf.electrodes.describe()
        while (isinstance(reported, list) and len(reported) == 1
               and isinstance(reported[0], list)):
            reported = reported[0]
        return reported

    def _visible_at(self, thickmix):
        handle = type(self).handle
        handle.ui.set("surface.S1.depth", thickmix)
        time.sleep(1.5)
        reported = handle.surfs[0].surf.electrodes.describe()
        while (isinstance(reported, list) and len(reported) == 1
               and isinstance(reported[0], list)):
            reported = reported[0]
        return {c["name"] for c in reported if c["visible"]}

    def test_the_depth_window_is_off_by_default(self):
        """Only the placement policy decides whether a contact is drawn.

        The window used to default to 2 mm and to ride the surface's depth
        slider, which itself defaults to mid-ribbon. That hid 53 of 143
        placeable contacts of a real montage on load -- every one of the 35
        sitting outside the pia among them -- and said nothing. Worse, measured
        from the pia it asks the same question as the policy's
        `max_surface_distance_mm` and answered it more strictly, so a contact
        that passed the 4 mm projection gate was hidden by a 2 mm window
        anyway. Two gates, one of them invisible.
        """
        handle = type(self).handle
        electrodes = handle.surfs[0].surf.electrodes
        assert electrodes.setDepthWindow() in (20, [20]), "window is not off"

        handle.ui.set("surface.S1.unfold", 0.5)
        time.sleep(1.5)
        try:
            at_pia, at_wm = self._visible_at(0.0), self._visible_at(1.0)
        finally:
            handle.ui.set("surface.S1.unfold", 0.0)
            time.sleep(1.5)

        assert len(at_pia) == len(type(self).eset), "the default hid contacts"
        assert at_pia == at_wm, "the depth slider changed what is drawn"

    def test_the_depth_window_is_measured_from_the_pia(self):
        """Set, it is a distance from the pia rather than from the slider.

        The slider's mid-ribbon default is a fine place to render a surface and
        a bad place to look for electrodes: a subdural contact sits above the
        pia by construction and is never near the middle of the ribbon.
        """
        handle = type(self).handle
        electrodes = handle.surfs[0].surf.electrodes
        handle.ui.set("surface.S1.unfold", 0.5)
        time.sleep(1.5)
        electrodes.setDepthWindow(2.0)
        time.sleep(1)
        try:
            at_pia, at_wm = self._visible_at(0.0), self._visible_at(1.0)
        finally:
            electrodes.setDepthWindow(20)
            handle.ui.set("surface.S1.unfold", 0.0)
            time.sleep(1.5)

        assert at_pia, "nothing visible when sampling the pial surface"
        assert at_pia == at_wm, (
            "the depth slider changed what is drawn, so the window is riding "
            "it rather than the pia"
        )

    def test_the_depth_window_can_be_asked_to_follow_the_slider(self):
        """The old behaviour, kept for sweeping through the ribbon deliberately.

        Turned on, a deformed surface shows one depth at a time and a contact
        that is not at that depth is not on the sheet being drawn. Which
        contacts survive then has to change as the slider moves; if it does
        not, the slider is writing the uniform without telling anything that
        positions itself on the CPU, which is the bug the original version of
        this test pinned.
        """
        handle = type(self).handle
        electrodes = handle.surfs[0].surf.electrodes
        handle.ui.set("surface.S1.unfold", 0.5)
        time.sleep(1.5)
        electrodes.setDepthWindow(2.0)
        electrodes.setDepthFollowsSlider(True)
        time.sleep(1)
        try:
            at_pia, at_wm = self._visible_at(0.0), self._visible_at(1.0)
        finally:
            electrodes.setDepthFollowsSlider(False)
            electrodes.setDepthWindow(20)
            handle.ui.set("surface.S1.unfold", 0.0)
            time.sleep(1.5)

        # The fixture's grid sits at the pia, so sampling there shows it and
        # sampling the white matter does not.
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

    def test_filters_compose_and_clear(self):
        """Each filter is one criterion and a contact has to pass all of them."""
        e = type(self).handle.surfs[0].surf.electrodes

        def U(v):
            while isinstance(v, list) and len(v) == 1:
                v = v[0]
            return v

        everything = U(e.countVisible())
        assert everything > 0

        e.filter_group_type("grid")
        time.sleep(0.5)
        grids = U(e.countVisible())
        assert 0 < grids <= everything

        e.filter_group_type("all")
        time.sleep(0.5)
        assert U(e.countVisible()) == everything

    def test_shape_control(self):
        e = type(self).handle.surfs[0].surf.electrodes

        def U(v):
            while isinstance(v, list) and len(v) == 1:
                v = v[0]
            return v

        e.setShape("cube")
        time.sleep(0.5)
        assert U(e.setShape()) == "cube"
        e.setShape("sphere")
        time.sleep(0.5)

    def test_marker_radius_comes_from_the_recorded_diameter(self):
        """Half the contact's own diameter, or the fallback where the montage
        records none. No global radius control: the montage already knows how
        big its electrodes are."""
        e = type(self).handle.surfs[0].surf.electrodes

        def U(v):
            while isinstance(v, list) and len(v) == 1:
                v = v[0]
            return v

        eset = type(self).eset
        for i in range(min(4, len(eset))):
            expected = (eset.size[i] / 2 if np.isfinite(eset.size[i])
                        else U(e.radius))
            assert U(e.contacts[i].radius) == pytest.approx(expected)

    def test_clicking_a_contact_opens_the_metadata_panel(self):
        """Driven through the real hit test, by projecting a contact to screen
        coordinates rather than poking the selection in directly."""
        e = type(self).handle.surfs[0].surf.electrodes

        def U(v):
            while isinstance(v, list) and len(v) == 1:
                v = v[0]
            return v

        xy = U(e.projectContact(0))
        name = U(e.selectAt(xy[0], xy[1]))
        assert name, "clicking a projected contact selected nothing"
        assert U(e.selected()) == name
        # ...and clicking empty space clears it
        assert U(e.selectAt(-0.98, -0.98)) == ""

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


# -- a depth electrode drawn off the surface -------------------------------

@pytest.mark.skipif(not has_playwright, reason="playwright and chromium are required")
class TestAShaftKeepsItsShapeInTheViewer:
    """The end-to-end version of the frame machinery, in a real browser.

    The python tests assert that a shared frame keeps a shaft rigid. This one
    asserts the browser agrees, which is a separate claim: the frame is rebuilt
    there from three ``get_position`` calls against a morphing surface, through
    two vertex permutations, with the scale recomputed per mix event. Any of
    those going wrong still draws markers on a brain.

    Asserted as a *pitch*, not a picture. A shaft drawn with the wrong indices
    or the wrong basis still photographs like a depth electrode -- that is the
    lesson the grid tests in this file were written from, and it applies harder
    here because a scatter of dots inside a translucent brain looks much like a
    track.
    """

    @pytest.fixture(autouse=True, scope="class")
    def _viewer(self, request):
        eset = build_shaft()
        cls = request.cls
        cls.eset = eset
        with cortex.export.headless_viewer(
            cortex.Vertex.random(SUBJECT), viewer_params=dict(electrodes=eset)
        ) as handle:
            time.sleep(3)
            cls.handle = handle
            yield

    def _positions(self):
        reported = type(self).handle.surfs[0].surf.electrodes.describe()
        while (isinstance(reported, list) and len(reported) == 1
               and isinstance(reported[0], list)):
            reported = reported[0]
        order = {name: i for i, name in enumerate(type(self).eset.names)}
        reported = sorted(reported, key=lambda c: order[c["name"]])
        return np.array([c["position"] for c in reported])

    def _unfold(self, value):
        type(self).handle.ui.set("surface.S1.unfold", value)
        time.sleep(2)
        return self._positions()

    def test_the_payload_carries_one_frame_for_the_whole_device(self):
        payload = to_viewer_json(type(self).eset)
        devices = {c["device"] for c in payload["electrodes"]}
        assert len(devices) == 1
        assert all(c["frame"] is not None for c in payload["electrodes"])
        assert all(c["frame_scale_mm"] for c in payload["electrodes"])

    def test_the_frame_triangle_is_converted_like_any_other_vertex(self, ctmfile):
        """``frame_verts`` makes the same CTM trip ``verts`` does.

        Sending it unconverted is the exact failure mode section 5 of the design
        document describes: the markers still land on the surface and still
        track the morph, and are simply anchored to the wrong triangle. Nothing
        about the picture says so, which is why this is asserted on the payload.
        """
        eset = type(self).eset
        payload = to_viewer_json(eset, ctm_index=ctm_vertex_index(ctmfile))
        sent = [c["frame_verts"] for c in payload["electrodes"]]

        # Every contact of this shaft shares one frame, so every payload row
        # names the same triangle.
        assert all(row == sent[0] for row in sent)

        # And it is the converted form of the host's triangle, not the raw one.
        host = int(eset.anchors.hosts[0])
        assert sent[0] != [int(v) for v in eset.anchors.verts[host]]
        expected = payload["electrodes"][
            [c["index"] for c in payload["electrodes"]].index(host)
        ]["verts"]
        assert sent[0] == expected

    def test_on_the_anatomical_surface_the_contacts_are_where_they_were_measured(self):
        """Reconstruction, not a special case.

        At unfold 0 the frame is rebuilt against the pial triangle, so the
        residual reproduces the measured coordinate exactly -- the same identity
        the python tests assert, arrived at through the browser's own arithmetic.
        """
        assert np.allclose(self._unfold(0.0), type(self).eset.coords, atol=1e-2)

    def test_inflating_keeps_the_pitch_uniform(self):
        """What the whole mechanism is for.

        Projected, this shaft's consecutive spacings on the inflated surface run
        from 1.7 to 40.5 mm. Carried through one frame they are all equal, and
        the tolerance here is loose only because the value comes back through a
        JSON bridge -- the arithmetic itself is exact.
        """
        positions = self._unfold(1.0)
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        assert steps.min() > 0.5
        assert (steps.max() - steps.min()) < 0.05 * steps.mean()

        centred = positions - positions.mean(0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        off_line = np.linalg.norm(centred - np.outer(centred @ axis, axis), axis=1)
        assert off_line.max() < 0.05 * steps.mean()

    def test_turning_offsets_off_gives_the_projected_positions_back(self):
        """The control, and the proof the two paths genuinely differ.

        Without it a bug that quietly ignored the frame would pass every
        assertion above that does not measure the pitch.
        """
        self._unfold(1.0)
        electrodes = type(self).handle.surfs[0].surf.electrodes
        electrodes.setOffsets(False)
        time.sleep(1.5)
        projected = self._positions()
        electrodes.setOffsets(True)
        time.sleep(1.5)
        carried = self._positions()

        spread = lambda p: np.ptp(  # noqa: E731
            np.linalg.norm(np.diff(p, axis=0), axis=1)
        )
        assert spread(projected) > 10.0
        assert spread(carried) < 0.5

    def test_the_page_raises_no_errors(self):
        errors = [e for e in type(self).handle._pw_thread.browser_errors
                  if "[pageerror]" in e]
        assert errors == [], "JS errors: %s" % errors


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
        # The cortex is the large mid-grey region; the coloured things on top of
        # it are the markers and the lines joining them.
        greyish = ((sat <= 40) & (img.mean(2) > 40) & (img.mean(2) < 220)).sum()
        coloured = (sat > 40).sum()
        # A ratio rather than a pixel count, because the claim is "the cortex is
        # grey" and the brain's size on screen depends on the camera.
        #
        # The allowance was 20x when this was written, when markers were the only
        # coloured thing. Connection lines are a second one, drawn in the marker
        # colour: measured on this view the ratio is 22.7 with the lines off and
        # 15.0 with them on, while the grey count barely moves (36529 vs 35776).
        # So the surface is not what changed. 5x keeps the failure this test is
        # for -- a data layer leaking onto the cortex would not shade this
        # number, it would invert it, colouring most of what is counted as grey.
        assert greyish > 5 * coloured, "the cortex looks coloured, not grey"

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


# -- per-view marker vectors ------------------------------------------------

def _unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def test_each_contact_carries_its_montage_index():
    """The payload drops unplaceable contacts, so a contact's position in the
    list is not its position in the montage. A view's marker arrays are
    montage-length, so the index is what makes them readable at all."""
    from cortex.electrodes import to_viewer_json

    eset = build_grid()
    eset.anchor()
    payload = to_viewer_json(eset)

    assert payload["nelec"] == len(eset)
    assert [c["index"] for c in payload["electrodes"]] == list(
        np.flatnonzero(eset.anchors.placeable)
    )


@pytest.mark.skipif(not has_playwright, reason="playwright and chromium are required")
class TestPerViewMarkers:
    """A view whose marker vectors override the montage's own diameters."""

    SIZES = [1.0, 2.0, 4.0, np.nan, 8.0, 10.0]
    SHAPES = ["sphere", "cube", "diamond", "cube", "sphere", "diamond"]
    MONTAGE_DIAMETER = 3.0

    @pytest.fixture(autouse=True, scope="class")
    def _viewer(self, request, tmp_path_factory):
        import shutil

        from cortex.electrodes import ElectrodeSet, load_surface_pairs

        cls = request.cls
        pair = load_surface_pairs(SUBJECT)["lh"]
        pia = pair.pia.astype(np.float64)
        n = len(cls.SIZES)
        eset = ElectrodeSet(
            ["E%d" % (i + 1) for i in range(n)],
            pia[np.arange(n) * 500 + 40000],
            subject=SUBJECT, group_type=["grid"] * n,
            size=[cls.MONTAGE_DIAMETER] * n,
        )
        eset.anchor()

        store = str(tmp_path_factory.mktemp("markerstore"))
        shutil.copytree(os.path.join(cortex.db.filestore, SUBJECT),
                        os.path.join(store, SUBJECT))
        old = cortex.db.filestore
        cortex.db.filestore = store
        cortex.db.reload_subjects()
        cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)

        view = cortex.Electrode(np.arange(float(n)), SUBJECT, "native",
                                marker_size=cls.SIZES, marker_shape=cls.SHAPES)
        cls.plain = cortex.electrodes.blank(SUBJECT) if hasattr(
            cortex.electrodes, "blank") else cortex.Vertex.random(SUBJECT)

        with cortex.export.headless_viewer(view, viewer_params={}) as handle:
            time.sleep(4)
            cls.handle = handle
            yield

        cortex.db.filestore = old
        cortex.db.reload_subjects()

    def _described(self):
        return _unwrap(type(self).handle.surfs[0].surf.electrodes.describe())

    def test_marker_size_overrides_the_montage_diameter(self):
        """...and a NaN entry means "not this one", falling back to the montage
        rather than to zero."""
        radii = [c["radius"] for c in self._described()]
        expected = [
            type(self).MONTAGE_DIAMETER / 2 if np.isnan(s) else s / 2
            for s in type(self).SIZES
        ]
        assert radii == pytest.approx(expected)

    def test_marker_shape_is_per_contact(self):
        assert [c["shape"] for c in self._described()] == type(self).SHAPES

    def test_the_shape_menu_overrides_and_auto_gives_the_view_back(self):
        e = type(self).handle.surfs[0].surf.electrodes
        e.setShape("cube")
        time.sleep(0.7)
        assert {c["shape"] for c in self._described()} == {"cube"}

        e.setShape("auto")
        time.sleep(0.7)
        assert [c["shape"] for c in self._described()] == type(self).SHAPES

    def test_the_page_raises_no_errors(self):
        errors = [e for e in type(self).handle._pw_thread.browser_errors
                  if "[pageerror]" in e]
        assert errors == [], "JS errors: %s" % errors


@pytest.mark.skipif(not has_playwright, reason="playwright and chromium are required")
def test_the_shape_vocabulary_matches_the_viewer(tmp_path_factory):
    """Python names the shapes and the viewer builds them; nothing links the two
    but this test. Drive every name and check the viewer accepts it."""
    import shutil

    from cortex.dataset.electrode_views import MARKER_SHAPES
    from cortex.electrodes import ElectrodeSet, load_surface_pairs

    pair = load_surface_pairs(SUBJECT)["lh"]
    pia = pair.pia.astype(np.float64)
    eset = ElectrodeSet(["E1", "E2"], pia[[40000, 40500]], subject=SUBJECT,
                        group_type=["grid"] * 2)
    eset.anchor()

    store = str(tmp_path_factory.mktemp("vocabstore"))
    shutil.copytree(os.path.join(cortex.db.filestore, SUBJECT),
                    os.path.join(store, SUBJECT))
    old = cortex.db.filestore
    cortex.db.filestore = store
    cortex.db.reload_subjects()
    cortex.db.save_montage(SUBJECT, eset, "native", overwrite=True)
    try:
        view = cortex.Electrode(np.zeros(2), SUBJECT, "native")
        with cortex.export.headless_viewer(view, viewer_params={}) as handle:
            time.sleep(4)
            e = handle.surfs[0].surf.electrodes
            for name in MARKER_SHAPES:
                e.setShape(name)
                time.sleep(0.6)
                drawn = {c["shape"] for c in _unwrap(e.describe())}
                assert drawn == {name}, "viewer would not draw %r" % name
    finally:
        cortex.db.filestore = old
        cortex.db.reload_subjects()
