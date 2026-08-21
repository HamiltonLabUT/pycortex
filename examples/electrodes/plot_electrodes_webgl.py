"""
==========================================
Show intracranial electrodes in the viewer
==========================================

Electrode markers in the webgl viewer, which keep their place on the cortex as
you inflate and flatten it.

Nothing about a stored 3-D position could survive that, so what is sent to the
browser is not a position: it is the *anchor* :mod:`cortex.electrodes` computes
-- the triangle a contact sits over, three barycentric weights within it, and a
normalised cortical depth. The browser re-derives a position from that anchor
for whatever state the sliders are in, the same way the picker's axes and the
ROI labels already do.

Drag the ``unfold`` slider in the ``surface`` folder and the markers travel with
the cortex. The ``electrodes`` folder has their visibility, radius, and how far
they stand off the surface.

The demo subject has no real electrodes, so this builds a synthetic 64-contact
grid and a 9-contact depth electrode. For a real montage, replace the whole
"Build a montage" block with one line::

    eset = cortex.electrodes.load_electrodes("TDT_elecs_all.mat", subject="S1")

which reads the ``img_pipe`` layout (``elecmatrix`` / ``eleclabels`` /
``anatomy``), or point it at a BIDS ``*_electrodes.tsv``. The subject has to be
imported into the pycortex filestore, not just present in FreeSurfer.
"""

import numpy as np

import cortex
from cortex.electrodes import ElectrodeSet, load_surface_pairs

subject = "S1"

# ---------------------------------------------------------------------------
# Build a montage. Replace this block with load_electrodes() for a real one.
# ---------------------------------------------------------------------------
pair = load_surface_pairs(subject)["lh"]
pia, wm = pair.pia.astype(np.float64), pair.wm.astype(np.float64)

# The white-matter-to-pia direction is the unambiguous "outward", and the axis
# cortical depth is measured along. Vertex normals are not usable: in a sulcal
# fundus they point along the sulcal wall rather than out of the head.
column = pia - wm
column /= np.linalg.norm(column, axis=1)[:, None]

near = np.flatnonzero(np.linalg.norm(pia - np.array([-62.0, -10.0, 0.0]), axis=1) < 25)
seed = int(near[np.argmax(column[near] @ np.array([-1.0, 0.0, 0.0]))])
out = column[seed]
u = np.cross(out, [0, 0, 1.0]); u /= np.linalg.norm(u)
v = np.cross(out, u)

# An 8x8 grid, laid on a plane then draped onto the folded surface.
planar = np.array([pia[seed] + (i - 3.5) * 8 * u + (j - 3.5) * 8 * v + 6 * out
                   for i in range(8) for j in range(8)])
draped = ElectrodeSet(["g%d" % i for i in range(64)], planar, subject=subject)
draped.anchor(surfaces={"lh": pair})
grid = draped.anchors.evaluate({"lh": pia})

# A depth electrode driven medially from the same gyral crown.
shaft = np.array([pia[seed] + 2 * out - k * 6 * out for k in range(9)])

electrodes = ElectrodeSet(
    names=["LSTG%d" % (i + 1) for i in range(64)] + ["LAMY%d" % (i + 1) for i in range(9)],
    coords=np.vstack([grid, shaft]),
    subject=subject,
    group_type=["grid"] * 64 + ["seeg"] * 9,
    size=[2.3] * 64 + [0.8] * 9,
)

# ---------------------------------------------------------------------------
# Anchor them, and read the report before drawing anything. Nothing is ever
# dropped silently: every contact keeps its coordinate and its anchor and is
# merely marked, so this is a summary rather than a casualty list.
# ---------------------------------------------------------------------------
anchors = electrodes.anchor()
print(anchors.summary())
print("groups:", electrodes.groups)

# ---------------------------------------------------------------------------
# Show it. Marker shape follows group type -- a sphere for a grid, a cube for a
# strip, a diamond for a depth electrode -- so a reader can tell at a glance
# which positions to trust. A depth contact's surface position is a locality,
# not a location.
#
# This blocks and opens a browser tab. Ctrl-C in the terminal to stop the server.
# ---------------------------------------------------------------------------
cortex.webgl.show(
    cortex.Vertex.random(subject),
    electrodes=electrodes,
    overlays_visible=(),
)

# For a static bundle instead of a live server, which is what you would send
# someone or drop in a paper's supplement:
#
#     cortex.webgl.make_static("/tmp/electrode_viewer", cortex.Vertex.random(subject),
#                              electrodes=electrodes)
