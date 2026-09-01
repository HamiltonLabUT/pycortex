"""Spreading a value recorded at one contact over the cortex around it.

An electrode reports from a point. A figure of two hundred of them is two
hundred dots, which is the wrong shape for the question usually being asked of
it -- where on cortex is this effect, and how does it compare with a surface or
volumetric result drawn from the same subject. What answers that is a *field*:
each contact's value smeared over the cortex within a few millimetres of it, so
the montage becomes an ordinary :class:`cortex.Vertex` or :class:`cortex.Volume`
and every tool that already works on those works on it.

The whole module is one matrix. :func:`surface_weights` returns a sparse
``(n_contacts, n_vertices)`` array whose entry ``(i, v)`` is how much contact
``i`` has to say about vertex ``v`` -- a Gaussian of the distance between them,
zero past a cutoff. :func:`volume_weights` is the same thing over voxels. Every
map anyone wants is a reduction of that matrix against the data:

    coverage        W.sum(0)                    how many contacts speak here
    weighted mean   (values @ W) / W.sum(0)     what they say, on average

Keeping the matrix separate from the reductions is what makes the second one
honest. A weighted mean of two contacts 40 mm apart is a number, and it means
nothing; the denominator is the only thing that says so, and a caller who has it
can threshold on it. :meth:`cortex.Electrode.to_vertex` does, at ``min_weight``.

Distance is measured **along the surface** by default rather than through space.
The difference is not small and not a refinement: the two banks of a sulcus are
routinely 3 mm apart in 3-D and 15 mm apart across cortex, and a subdural
contact on a gyral crown is not recording from the bank underneath it. Euclidean
distance is available, and is roughly an order of magnitude faster; it is the
right choice for a first look at a large montage and the wrong one for a figure.

Nothing here transforms coordinates. :meth:`cortex.electrodes.ElectrodeSet.anchor`
has already put the contacts into the space the subject's surfaces are in --
including the TkRegRAS offset that :func:`~cortex.electrodes.surface_space_offset`
reads out of the surface files -- and pycortex's transforms are defined against
those same surfaces, so the volumetric path needs no separate correction either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Optional

import numpy as np
import numpy.typing as npt
from scipy import sparse

from ._anchor import SurfacePair, surface_hash
from ._set import load_surface_pairs, surface_space_offset

if TYPE_CHECKING:
    from ._set import ElectrodeSet

HEMIS = ("lh", "rh")

#: Width of the blob, in millimetres. Half maximum at 3.5 mm, which is about one
#: contact's worth of spread for the 4-10 mm pitch of a clinical grid: adjacent
#: contacts overlap and are averaged, distant ones do not reach each other.
DEFAULT_SIGMA = 3.0

#: The cutoff, as a multiple of sigma, when none is given. Three sigma leaves
#: 1.1% of the peak at the edge -- close enough to zero that the truncation is
#: invisible, near enough that the geodesic patch stays small. Deriving it from
#: sigma rather than fixing it is what stops a wider blob from being silently
#: squared off: at sigma=10 a fixed 10 mm cutoff would chop the kernel at its
#: half maximum and draw a disc with a hard edge.
RADIUS_SIGMAS = 3.0

Metric = Literal["geodesic", "euclidean"]


def _resolve(sigma: float, radius: Optional[float]) -> tuple[float, float]:
    """Validate the kernel's two numbers, filling in the cutoff from sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive, got %r" % (sigma,))
    radius = RADIUS_SIGMAS * sigma if radius is None else radius
    if radius <= 0:
        raise ValueError("radius must be positive, got %r" % (radius,))
    return sigma, radius


def _gaussian(distances: npt.NDArray[np.floating], sigma: float) -> npt.NDArray[np.floating]:
    """The kernel, peaking at 1.

    Peak-one rather than unit-integral, so that the coverage map reads as a
    count -- about 1 under a lone contact, about 2 where two of them overlap --
    and so that a threshold on it means the same thing whatever ``sigma`` is.
    The weighted mean is unaffected either way: a constant factor cancels
    between numerator and denominator.
    """
    return np.exp(-np.square(distances) / (2.0 * sigma * sigma))


def _anchors_for(
    electrodes: "ElectrodeSet",
    subject: Optional[str],
    surfaces: Optional[Mapping[str, SurfacePair]] = None,
):
    """This set's anchors and the subject they belong to, computing them if needed.

    Anchoring is stored back onto the set, because it is a cache: it is derived
    from :attr:`~cortex.electrodes.ElectrodeSet.coords`, it can be recomputed at
    any time, and asking for a second radius should not pay for it twice. A set
    anchored against surfaces that have since changed is re-anchored rather than
    trusted; that is what the anchors' ``surface_hash`` is for. An empty hash
    means they came from somewhere that recorded none, and is taken at face
    value.
    """
    subject = subject or electrodes.subject
    if subject is None and surfaces is None:
        raise ValueError(
            "spreading electrode values over cortex needs a subject, either on "
            "the set or as an argument"
        )

    anchors = electrodes.anchors
    if anchors is None:
        anchors = electrodes.anchor(subject, surfaces=surfaces)
    elif surfaces is None and anchors.surface_hash and subject is not None:
        if anchors.surface_hash != surface_hash(load_surface_pairs(subject)):
            anchors = electrodes.anchor(subject)
    return subject, anchors


def _measuring_surfaces(
    subject: Optional[str],
    surface_type: str,
    surfaces: Optional[Mapping[str, SurfacePair]],
):
    """``{hemi: (pts, polys)}`` for the surface distances are measured on.

    Given surfaces directly, the mid-surface is synthesised the same way
    :meth:`cortex.database.Database.get_surf` synthesises a missing fiducial --
    halfway between white matter and pia -- so that passing surfaces and naming
    ``"fiducial"`` mean the same thing.
    """
    if surfaces is not None:
        return {
            hemi: (
                pair.pia if pair.wm is None else (pair.pia + pair.wm) / 2.0,
                pair.polys,
            )
            for hemi, pair in surfaces.items()
        }

    from ..database import db

    if subject is None:  # unreachable: `_anchors_for` has already refused this
        raise ValueError("a subject is needed to load surfaces")

    # nudge=False without exception: nudging shifts the hemispheres apart in x
    # for display, and every distance measured on a nudged surface between the
    # hemispheres would be a distance in a picture rather than in a head.
    return dict(zip(HEMIS, db.get_surf(subject, surface_type, "both", nudge=False)))


def surface_weights(
    electrodes: "ElectrodeSet",
    sigma: float = DEFAULT_SIGMA,
    radius: Optional[float] = None,
    metric: Metric = "geodesic",
    surface_type: str = "fiducial",
    subject: Optional[str] = None,
    placeable: bool = True,
    surfaces: Optional[Mapping[str, SurfacePair]] = None,
    m: float = 1.0,
) -> sparse.csr_matrix:
    """How much each contact has to say about each vertex.

    Parameters
    ----------
    electrodes : ElectrodeSet
        Anchored if it is not already, and the anchors are kept on it.
    sigma : float
        Width of the Gaussian, in millimetres.
    radius : float, optional
        Cutoff in millimetres; vertices further than this from a contact take
        nothing from it. Defaults to three sigma. This is the parameter that
        decides how long the call takes and how much memory the result needs;
        ``sigma`` is the one that decides what the picture looks like.
    metric : {"geodesic", "euclidean"}
        How to measure from a contact to a vertex. ``"geodesic"`` measures along
        the cortical surface, ``"euclidean"`` straight through space. They
        disagree wherever cortex folds, which is everywhere that matters: the
        far bank of a sulcus is a few millimetres away through space and a
        couple of centimetres away across cortex, and a contact on this bank is
        not recording from that one. Euclidean is roughly an order of magnitude
        faster and is the right choice for a first look at a large montage.
    surface_type : str
        Which of the subject's surfaces to measure on. Should be a closed
        anatomical surface -- ``"fiducial"``, ``"pia"``, ``"wm"``. Distances on
        ``"inflated"`` or ``"flat"`` are distances in a rendering, not in a brain.
    subject : str, optional
        Overrides the set's own subject.
    placeable : bool
        Leave out the contacts that have no honest surface position -- no
        coordinate, or further from cortex than their placement policy allows.
        See :attr:`cortex.electrodes.ElectrodeAnchors.placeable`, which keeps
        ``non_cortical`` contacts in: a hippocampal contact's anchor is sound,
        and where it projects to is usually what a reader wants shown.
    surfaces : mapping, optional
        Pre-loaded ``{"lh": SurfacePair, "rh": SurfacePair}``, which skips the
        database entirely, exactly as in
        :meth:`cortex.electrodes.ElectrodeSet.anchor`. Taken to be in the same
        space as the coordinates, so no offset is applied.
    m : float
        Reverse Euler step length, passed through to
        :meth:`cortex.polyutils.Surface.geodesic_distance`. Geodesic only.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_contacts, n_vertices)``, the vertex axis in the left-then-right
        order every :class:`cortex.Vertex` uses. Row ``i`` is contact ``i`` of
        ``electrodes``, so the matrix stays aligned with a data array recorded
        on the same channels; a contact that was left out gets an empty row
        rather than shifting every row beneath it.

    Notes
    -----
    Distances are measured from where a contact *anchors* on the surface, not
    from the contact itself. The two differ by the contact's distance from
    cortex: a couple of millimetres for a subdural grid, a couple of centimetres
    for a depth electrode driven through white matter. Measuring from the raw
    coordinate would fade a depth contact's blob out in proportion to how deep
    it was driven, which is a statement about the surgery rather than about the
    recording. Whether a contact is near enough to cortex to be saying anything
    about it is a different question, and
    :class:`~cortex.electrodes.PlacementPolicy` has already answered it.

    Under ``metric="geodesic"`` the blob is seeded from the nearest vertex of
    the anchor's triangle rather than from the exact barycentric point, because
    the heat method solves from vertices. That moves the centre by at most half
    an edge -- under a millimetre on a native surface, against a default
    ``sigma`` of three, and far less than the choice of ``surface_type`` moves it.

    Geodesic distance itself is approximate, which matters more. It comes from
    :meth:`cortex.polyutils.Surface.geodesic_distance`, which recovers distance
    from a simulated diffusion (Crane et al. 2012) and on a millimetre-scale mesh
    runs a few percent short -- measured at up to 0.05 of a weight on a flat test
    sheet, in the direction of a slightly fat blob. That is well inside the
    uncertainty already carried by the choice of ``sigma``, and it is not a
    reason to prefer euclidean: being a few percent wrong about a distance along
    cortex beats being exactly right about a distance through the skull.
    """
    from ..polyutils import Surface

    if metric not in ("geodesic", "euclidean"):
        raise ValueError("metric must be 'geodesic' or 'euclidean', got %r" % (metric,))
    sigma, radius = _resolve(sigma, radius)

    subject, anchors = _anchors_for(electrodes, subject, surfaces)
    surfs = _measuring_surfaces(subject, surface_type, surfaces)

    offsets = {"lh": 0, "rh": len(surfs["lh"][0])}
    nverts = offsets["rh"] + len(surfs["rh"][0])

    # Where each contact lands on the measuring surface. `evaluate` needs only
    # the points: an anchor names its triangle by vertex, and vertex indexing is
    # shared across all of a subject's surfaces, which is what makes it portable.
    feet = anchors.evaluate({hemi: surfs[hemi][0] for hemi in HEMIS})
    usable = anchors.placeable if placeable else np.isfinite(feet).all(axis=1)

    rows: list[npt.NDArray[np.integer]] = []
    cols: list[npt.NDArray[np.integer]] = []
    vals: list[npt.NDArray[np.floating]] = []

    for hemi in HEMIS:
        contacts = np.flatnonzero(usable & (anchors.hemi == hemi))
        if len(contacts) == 0:
            continue

        # Built once per hemisphere and kept for every contact in it. A Surface
        # memoises its polygon adjacency on the instance, which is what walks
        # each patch's connected component, and rebuilding the surface per
        # contact would throw that away every time.
        surf = Surface(*surfs[hemi])

        for i in contacts:
            if metric == "euclidean":
                near = np.flatnonzero(surf.get_euclidean_ball(feet[i], radius))
                distances = np.linalg.norm(surf.pts[near] - feet[i], axis=1)
            else:
                seed = int(anchors.verts[i][np.argmax(anchors.weights[i])])
                patch = surf.get_geodesic_patch(seed, radius, m=m)
                near = np.flatnonzero(patch["vertex_mask"])
                distances = np.asarray(patch["geodesic_distance"], dtype=np.float64)

                # A vertex the heat solve could not reach comes back as distance
                # *zero*, not infinity -- `geodesic_distance` fills its output
                # with zeros and writes only the rows it solved. Unguarded, such
                # a vertex takes the full weight of every blob. Geodesic distance
                # is never shorter than straight-line distance, so a vertex
                # further than `radius` through space cannot honestly be within
                # `radius` across the surface, whatever the solver returned.
                near, distances = _drop_unreachable(surf, feet[i], near, distances, radius)

            keep = np.isfinite(distances) & (distances <= radius)
            if not keep.any():
                continue
            near, distances = near[keep], distances[keep]

            rows.append(np.full(len(near), i))
            cols.append(near + offsets[hemi])
            vals.append(_gaussian(distances, sigma))

    return _assemble(rows, cols, vals, (len(electrodes), nverts))


def _drop_unreachable(surf, foot, near, distances, radius):
    """Cut vertices whose reported geodesic distance is contradicted by geometry.

    Only ever fires on a solver failure: for every real vertex the straight-line
    distance is the shorter of the two, so this rejects nothing that the
    geodesic cutoff would have kept.
    """
    straight = np.linalg.norm(surf.pts[near] - foot, axis=1)
    honest = straight <= radius
    return near[honest], distances[honest]


def _assemble(rows, cols, vals, shape) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix(shape)
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
    )


def volume_weights(
    electrodes: "ElectrodeSet",
    xfmname: str,
    sigma: float = DEFAULT_SIGMA,
    radius: Optional[float] = None,
    subject: Optional[str] = None,
    placeable: bool = True,
    mask: Optional[npt.NDArray[np.bool_]] = None,
) -> sparse.csr_matrix:
    """How much each contact has to say about each voxel.

    The volumetric counterpart of :func:`surface_weights`, and simpler in one
    way and cruder in another. Simpler: there is no surface, so no anchoring and
    no choice of metric -- distance is straight-line distance and nothing else.
    Cruder for the same reason: a blob here crosses a sulcus, the fissure and
    the pia without noticing, so it says "near this contact in the head" rather
    than "on the cortex this contact records from". For a volume that respects
    the folding, spread on the surface and project::

        elec.to_vertex(sigma=3).volume(xfmname)

    Parameters
    ----------
    electrodes : ElectrodeSet
    xfmname : str
        Which of the subject's transforms names the voxel grid to fill.
    sigma, radius : float
        As in :func:`surface_weights`, both in millimetres.
    subject : str, optional
    placeable : bool
        Leave out contacts with no honest position. Anchors the set if it is not
        already anchored, since that is what decides placement; pass ``False``
        to spread every contact that has a finite coordinate and skip anchoring
        entirely.
    mask : ndarray, optional
        Boolean ``(z, y, x)``, restricting which voxels are filled -- a cortical
        mask, say. Both faster and tidier than filling skull and ventricle.

    Returns
    -------
    scipy.sparse.csr_matrix
        ``(n_contacts, n_voxels)``, the voxel axis in the C order of the
        transform's ``(z, y, x)`` shape, or of the masked voxels when ``mask``
        is given. Row ``i`` is contact ``i``, as in :func:`surface_weights`.

    Notes
    -----
    Distances are true millimetres, not voxel counts: the voxel offset is mapped
    back through the transform before it is measured, so an anisotropic or
    oblique grid gives a blob that is round in the head rather than round in the
    array. Contact positions are the raw coordinates, shifted into the surfaces'
    space by :func:`~cortex.electrodes.surface_space_offset` -- which is all
    that is needed, because a pycortex transform is defined against those same
    surfaces. There is no separate scanner-space conversion to do.
    """
    from ..database import db

    sigma, radius = _resolve(sigma, radius)

    subject = subject or electrodes.subject
    if subject is None:
        raise ValueError(
            "spreading electrode values over a volume needs a subject, either "
            "on the set or as an argument"
        )

    coords = electrodes.coords + surface_space_offset(subject)
    usable = np.isfinite(coords).all(axis=1)
    if placeable:
        _, anchors = _anchors_for(electrodes, subject)
        usable &= anchors.placeable

    xfm = db.get_xfm(subject, xfmname)
    shape = tuple(int(n) for n in xfm.shape)  # (z, y, x)
    nz, ny, nx = shape
    dims = np.array([nx, ny, nz])

    centres = xfm(coords)  # (n, 3) in voxels, x-y-z
    forward = np.asarray(xfm.xfm, dtype=np.float64)[:3, :3]  # mm -> voxel
    backward = np.linalg.inv(forward)  # voxel -> mm

    # Half-extent of the sphere's bounding box, per voxel axis. For a
    # displacement p of at most `radius` millimetres, the largest the j-th voxel
    # component of `forward @ p` can be is `radius * norm(forward[j])` -- the
    # row norm, which is the support function of the image ellipsoid. Column
    # norms would under-cover an oblique transform and clip the blob silently.
    half = radius * np.linalg.norm(forward, axis=1)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(
                "mask must have the transform's shape %r, got %r" % (shape, mask.shape)
            )
        # Column j of the output is the j-th True voxel of the mask, which is
        # the layout `Volume(data, subject, xfm, mask=mask)` expects.
        columns = np.full(shape, -1, dtype=np.int64)
        columns[mask] = np.arange(int(mask.sum()))
        n_columns = int(mask.sum())
    else:
        columns = None
        n_columns = nx * ny * nz

    rows: list[npt.NDArray[np.integer]] = []
    cols: list[npt.NDArray[np.integer]] = []
    vals: list[npt.NDArray[np.floating]] = []

    for i in np.flatnonzero(usable):
        lo = np.maximum(np.floor(centres[i] - half), 0).astype(int)
        hi = np.minimum(np.ceil(centres[i] + half), dims - 1).astype(int)
        if np.any(lo > hi):
            continue  # entirely outside the field of view

        gx, gy, gz = np.meshgrid(
            *(np.arange(lo[a], hi[a] + 1) for a in range(3)), indexing="ij"
        )
        grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()])  # (3, m), x-y-z
        distances = np.linalg.norm(backward @ (grid - centres[i][:, None]), axis=0)

        keep = distances <= radius
        if not keep.any():
            continue
        zyx = (grid[2][keep], grid[1][keep], grid[0][keep])

        if columns is None:
            column = np.ravel_multi_index(zyx, shape)
            weight = _gaussian(distances[keep], sigma)
        else:
            column = columns[zyx]
            inside = column >= 0
            column, weight = column[inside], _gaussian(distances[keep][inside], sigma)
            if not len(column):
                continue

        rows.append(np.full(len(column), i))
        cols.append(column)
        vals.append(weight)

    return _assemble(rows, cols, vals, (len(electrodes), n_columns))


def weighted_mean(
    data: npt.ArrayLike,
    weights: sparse.csr_matrix,
    min_weight: float = 0.01,
) -> npt.NDArray[np.floating]:
    """The kernel-weighted mean of per-contact values over the weight matrix.

    The reduction both :meth:`cortex.Electrode.to_vertex` and
    :meth:`cortex.Electrode.to_volume` are: what the nearby contacts say, each
    counted by how near it is.

    Parameters
    ----------
    data : array
        ``(n_contacts,)``, or ``(t, n_contacts)`` for a movie.
    weights : sparse matrix
        ``(n_contacts, n_targets)``, from :func:`surface_weights` or
        :func:`volume_weights`.
    min_weight : float
        Below this much total weight, the mean is reported as NaN rather than as
        data. A target with almost no weight on it has a mean that is one distant
        contact's value carried further than it can honestly reach; NaN renders
        transparent, so the blob shows its own footprint and nothing else.

    Returns
    -------
    ndarray
        ``(n_targets,)`` or ``(t, n_targets)`` -- the target axis last, which is
        what :class:`cortex.Vertex` and :class:`cortex.Volume` require of a movie.

    Notes
    -----
    NaN in the data is excluded rather than counted as zero: a NaN contact drops
    out of both the numerator and the denominator, so the result is the mean over
    the contacts that did report, and one dead channel does not blank the cortex
    around it. This is the same rule :func:`cortex.mapper.utils.nanproject`
    applies to a volume-to-surface mapper, arrived at the same way and by sparse
    products rather than by a loop over rows.
    """
    values = np.asarray(data, dtype=np.float64)
    flat = values.ndim == 1
    values = np.atleast_2d(values)  # (t, n)

    finite = np.isfinite(values)
    numerator = np.where(finite, values, 0.0) @ weights  # (t, targets)

    if finite.all():
        # One row, which broadcasts over t -- worth the branch, since this is
        # the ordinary case and the alternative is a sparse product per frame.
        denominator = np.asarray(weights.sum(axis=0))
    else:
        denominator = finite.astype(np.float64) @ weights

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(numerator / denominator)
    out[np.broadcast_to(denominator, out.shape) < min_weight] = np.nan

    return out[0] if flat else out


def total_weight(weights: sparse.csr_matrix) -> npt.NDArray[np.floating]:
    """How much contact there is at each target: the coverage map.

    In units of contacts -- about 1 where a single contact sits, about 2 where
    two of them overlap -- because the kernel peaks at one. Zero rather than NaN
    where no contact reaches, since "no electrode near here" is an answer and
    should be drawn as one.
    """
    return np.asarray(weights.sum(axis=0), dtype=np.float64).ravel()
