"""
==========================================
Electrode data as a field on the cortex
==========================================

A montage drawn as markers is a montage drawn as markers: fine for sixty-four
contacts, unreadable for six hundred, and impossible to compare against a
surface or volumetric result from the same subject. :meth:`cortex.Electrode.to_vertex`
spreads each contact's value over the cortex within a few millimetres of it and
averages the overlaps, so the montage becomes an ordinary :class:`cortex.Vertex`.

The distances are measured **along the cortical surface**, not through space.
That is the whole reason this is not two lines of KD-tree: the two banks of a
sulcus are millimetres apart through space and centimetres apart across cortex,
and a subdural contact on this bank is not recording from that one. The second
figure below shows what the difference costs you if you ignore it.

This example builds the same synthetic grid as
:doc:`plot_electrodes_flatmap`, since the pycortex demo subject has no real
electrodes. Swap the construction for

    eset = cortex.electrodes.load_electrodes("sub-01_electrodes.tsv", subject="S1")

to use a real BIDS montage.
"""

import matplotlib.pyplot as plt
import numpy as np

import cortex
from cortex.electrodes import ElectrodeSet, load_surface_pairs

subject = "S1"

# ---------------------------------------------------------------------------
# Build a synthetic 64-contact subdural grid. Skip to "The field" for the part
# that matters.
# ---------------------------------------------------------------------------
pair = load_surface_pairs(subject)["lh"]
pia, wm = pair.pia.astype(np.float64), pair.wm.astype(np.float64)
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

electrodes = ElectrodeSet(
    names=["LSTG%d" % (i + 1) for i in range(64)],
    coords=feet + 2.0 * normal / np.linalg.norm(normal, axis=1)[:, None],
    subject=subject,
    group_type=["grid"] * 64,
    size=[2.3] * 64,
)

# A plausible result to look at: one focus in the middle of the grid.
rows, cols = np.arange(64) // 8, np.arange(64) % 8
values = np.exp(-((rows - 4.0) ** 2 + (cols - 3.0) ** 2) / 8.0)
elec = cortex.Electrode(values, subject, "native", electrodes=electrodes, cmap="RdBu_r")

# ---------------------------------------------------------------------------
# The field. `sigma` is the only number that matters: it is the blob's width in
# millimetres, and roughly the pitch of the grid is a good starting point --
# adjacent contacts then overlap and get averaged, distant ones do not reach
# each other. The cutoff follows at three sigma unless you set `radius`.
#
# The markers are drawn on top, and colour-match the field: a weighted mean of
# non-negative weights lies between the smallest and largest value that fed it,
# so `to_vertex` keeps the view's own colour bounds.
# ---------------------------------------------------------------------------
field = elec.to_vertex(sigma=4)

fig = cortex.quickflat.make_figure(
    field, with_rois=False, with_curvature=True, height=600,
    with_electrodes=electrodes,
    electrode_kwargs=dict(color="k", size=14),
)
fig.suptitle("each contact's value, spread along the cortical surface", y=0.04)
plt.show()

# ---------------------------------------------------------------------------
# Coverage: how much electrode there is at each vertex, in contacts. About 1
# under a lone contact, about 2 where two overlap, 0 where none reaches.
#
# Show this beside any figure built from `to_vertex`. The field alone cannot
# distinguish cortex that six contacts agree about from cortex that one contact
# is speaking for at the far edge of its range; this is the map that can.
# ---------------------------------------------------------------------------
fig = cortex.quickflat.make_figure(
    elec.coverage(sigma=4, cmap="inferno"),
    with_rois=False, with_curvature=True, height=600,
    with_electrodes=electrodes, electrode_kwargs=dict(color="w", size=14),
)
fig.suptitle("coverage: how many contacts reach each vertex", y=0.04)
plt.show()

# ---------------------------------------------------------------------------
# What the metric buys. Euclidean distance is about ten times faster and is the
# right choice for a first look at a large montage -- but it does not know that
# cortex is folded, so it pours a contact's value straight through a sulcus onto
# the bank opposite, and through the fissure onto the other hemisphere. The
# euclidean field below is visibly smoother, and that smoothness is invented.
# ---------------------------------------------------------------------------
fig = cortex.quickflat.make_figure(
    elec.to_vertex(sigma=4, metric="euclidean"),
    with_rois=False, with_curvature=True, height=600,
    with_electrodes=electrodes, electrode_kwargs=dict(color="k", size=14),
)
fig.suptitle("the same data measured through space: smoother, and wrong", y=0.04)
plt.show()

# ---------------------------------------------------------------------------
# The same thing over a voxel grid, for comparison against volumetric data.
# There is no surface here, so no metric to choose: a blob crosses a sulcus and
# the fissure without noticing. When that matters, spread on the surface and
# project back instead -- `elec.to_vertex(sigma=4).volume("fullhead")`.
# ---------------------------------------------------------------------------
volume = elec.to_volume("fullhead", sigma=4)
print(volume.data.shape, np.isfinite(volume.data).sum(), "voxels filled")

# The weights themselves are available, if you want to do your own reduction:
# a sparse (n_contacts, n_vertices) matrix that both methods above are built on.
weights = cortex.electrodes.surface_weights(electrodes, sigma=4)
print("%d contacts x %d vertices, %d nonzero" % (*weights.shape, weights.nnz))
