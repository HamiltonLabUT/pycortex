"""
=========================================
Plot intracranial electrodes on a flatmap
=========================================

An electrode has no vertex identity, so nothing about its position survives
inflation or flattening on its own. :mod:`cortex.electrodes` fixes that by
*anchoring* each contact: recording which triangle it sits over, where inside
that triangle, and how deep through the cortical ribbon. One anchor then
evaluates on any surface the subject has.

This example builds a synthetic 64-contact subdural grid and a 9-contact sEEG
shaft, since the pycortex demo subject has no real electrodes, then draws them
with :func:`cortex.quickflat.add_electrodes`. Swap the construction for

    eset = cortex.electrodes.load_electrodes("sub-01_electrodes.tsv", subject="S1")

to use a real BIDS montage.
"""

import matplotlib.pyplot as plt
import numpy as np

import cortex
from cortex.electrodes import ElectrodeSet, load_surface_pairs

subject = "S1"

# ---------------------------------------------------------------------------
# Build a synthetic montage. Skip to "Draw them" for the part that matters.
# ---------------------------------------------------------------------------
pair = load_surface_pairs(subject)["lh"]
pia, wm = pair.pia.astype(np.float64), pair.wm.astype(np.float64)

# The white-matter-to-pia direction is the unambiguous "outward", and it is the
# axis cortical depth is measured along. Vertex normals are not usable here: in
# a sulcal fundus they point along the sulcal wall rather than out of the head.
column = pia - wm
column /= np.linalg.norm(column, axis=1)[:, None]

near = np.flatnonzero(np.linalg.norm(pia - np.array([-62.0, -10.0, 0.0]), axis=1) < 25)
seed = int(near[np.argmax(column[near] @ np.array([-1.0, 0.0, 0.0]))])
out = column[seed]
u = np.cross(out, [0, 0, 1.0]); u /= np.linalg.norm(u)
v = np.cross(out, u)

planar = np.array([pia[seed] + (i - 3.5) * 8 * u + (j - 3.5) * 8 * v + 6 * out
                   for i in range(8) for j in range(8)])
draped = ElectrodeSet(["g%d" % i for i in range(64)], planar, subject=subject)
draped.anchor(surfaces={"lh": pair})
feet = draped.anchors.evaluate({"lh": pia})
normal = np.einsum("nij,ni->nj", column[draped.anchors.verts], draped.anchors.weights)
grid = feet + 2.0 * normal / np.linalg.norm(normal, axis=1)[:, None]
shaft = np.array([pia[seed] + 2 * out - k * 6 * out for k in range(9)])

electrodes = ElectrodeSet(
    names=["LSTG%d" % (i + 1) for i in range(64)] + ["LAMY%d" % (i + 1) for i in range(9)],
    coords=np.vstack([grid, shaft]),
    subject=subject,
    group_type=["grid"] * 64 + ["seeg"] * 9,
    size=[2.3] * 64 + [0.8] * 9,
)

# ---------------------------------------------------------------------------
# Anchor them, and look at what the placement policy did before drawing
# anything. Nothing is ever dropped silently: every contact keeps its
# coordinate and its anchor, and is merely marked.
# ---------------------------------------------------------------------------
anchors = electrodes.anchor()
print(anchors.summary())
print(electrodes.groups)

# Are the coordinates even in the surfaces' space? pycortex keeps FreeSurfer's
# surface RAS, so TkRegRAS coordinates should land directly on the surfaces.
# Run the check on the surface contacts: depth electrodes are legitimately
# centimetres from the pia and would read as suspicious.
from cortex.electrodes import check_alignment
print(check_alignment(electrodes.select(group_type="grid").coords,
                      load_surface_pairs(subject)).summary())

# ---------------------------------------------------------------------------
# Draw them. Marker shape follows group type by default -- a circle for a grid,
# a diamond for a depth electrode -- so a reader can tell at a glance which
# positions to trust, without consulting a legend.
# ---------------------------------------------------------------------------
# An all-NaN data layer renders transparent, leaving the curvature background
# on its own -- the usual way to put something on a bare flatmap.
nverts = cortex.db.get_surf(subject, "fiducial", merge=True)[0].shape[0]
blank = cortex.Vertex(np.full(nverts, np.nan), subject)

fig = cortex.quickflat.make_figure(
    blank, with_rois=False, with_colorbar=False, with_curvature=True,
    with_electrodes=electrodes,
    electrode_values=electrodes.depth,
    electrode_kwargs=dict(cmap="RdYlBu_r", vmin=-0.6, vmax=3.0, size=55),
)
fig.suptitle("colour = cortical depth: 0 at the pia, 1 at the white matter", y=0.02)
plt.show()

# ---------------------------------------------------------------------------
# Depth is the same coordinate as the webgl viewer's depth slider, so a figure
# and a viewer can be made to agree. Here, only the contacts inside the
# cortical ribbon, scaled by their real contact diameter.
# ---------------------------------------------------------------------------
fig = cortex.quickflat.make_figure(
    blank, with_rois=False, with_colorbar=False, with_curvature=True,
    with_electrodes=electrodes,
    electrode_kwargs=dict(depth=(0.0, 1.0), size_by="size", size_range=(25, 130),
                          color="white"),
)
fig.suptitle("only the contacts inside grey matter, sized by contact diameter", y=0.02)
plt.show()
