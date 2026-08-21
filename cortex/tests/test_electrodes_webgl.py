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
