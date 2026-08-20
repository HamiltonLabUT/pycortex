"""Tying an electrode's 3-D coordinate to the cortical surface.

An electrode has no vertex identity, so nothing about its position survives
inflation or flattening on its own. What survives is a *parameterisation*: which
triangle it sits over, and where inside that triangle. Every surface pycortex
holds for a subject shares vertex indexing, so three vertex indices plus three
barycentric weights name the same anatomical spot on the fiducial, inflated and
flat surfaces alike -- which is the whole trick.

Vertex indices rather than a face index, and the distinction is not
bookkeeping: flattening cuts the medial wall away, so a subject's flat surface
has thousands fewer triangles than its pial one and face ``k`` is a different
triangle on each. A face index would evaluate silently and wrongly.

The third coordinate is depth. The pial and white-matter surfaces are not cut,
so the same vertex triple names a triangle on *both*, and the two
barycentric points ``P`` (pial) and ``W`` (white matter) bound a short segment
through the cortical ribbon. An electrode's normalised depth is its projection
onto that segment:

    d = dot(e - P, W - P) / |W - P|**2

with ``d = 0`` on the pia, ``d = 1`` at the white-matter boundary, ``d < 0``
outside the brain and ``d > 1`` past the white matter. This is deliberately the
same quantity the webgl viewer already interpolates: ``brainctm`` builds the CTM
with the pial surface as its base and white matter as an auxiliary attribute, and
``mriview.get_position`` mixes the two by ``thickmix`` -- so an electrode's depth
and the viewer's depth slider are the same number, and the slider can select
which electrodes are drawn.

Nothing here mutates the electrode's TkRegRAS coordinate. Anchors are derived,
carry a hash of the surfaces they were computed against, and can be thrown away
and recomputed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

HEMIS = ("lh", "rh")

#: Placement outcomes. Every electrode gets exactly one, and none is ever
#: dropped silently: an excluded electrode keeps its anchor and its coordinate
#: and is merely marked, so a caller can count, inspect and override.
ON_SURFACE = "on_surface"
"""Inside the cortical ribbon (``0 <= d <= 1``) and over its own column."""

PROJECTED = "projected"
"""Outside the ribbon but within policy: a grid contact sitting above the pia,
or a depth contact below the white matter. The anchor is meaningful; the
electrode is not literally at the position drawn."""

TOO_FAR = "too_far"
"""Beyond the policy's tolerance -- no cortical column can honestly claim it."""

UNKNOWN_ANATOMY = "unknown_anatomy"
"""Excluded by the anatomy rule rather than by geometry."""

PLACEMENTS = (ON_SURFACE, PROJECTED, TOO_FAR, UNKNOWN_ANATOMY)


class SurfacePair(NamedTuple):
    """One hemisphere's geometry, as anchoring needs it.

    ``wm`` is optional because a subject imported without a white-matter surface
    still has a usable anchor -- it just has no depth axis, and ``brainctm``
    likewise falls back to a CTM with no ``wm`` attribute, whose depth slider
    does nothing. Depth then comes back as NaN rather than as a fabricated zero.
    """

    pia: npt.NDArray[np.floating]
    polys: npt.NDArray[np.integer]
    wm: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class PlacementPolicy:
    """Which electrodes are considered placeable, and how far is too far.

    The rule is geometric first. The design document's original formulation --
    drop an electrode whose anatomical label is ``Unknown`` -- keys on a field
    that is *optional* in the data specification, so with unlabelled or
    atypical anatomy it discards contacts that are merely undocumented. Here
    the anatomy rule is one configurable term (off by default) on top of a
    distance test, and everything it excludes is reported rather than removed.

    Parameters
    ----------
    max_offset_mm : float
        How far an electrode may sit from the cortical column it is assigned
        to, measured perpendicular to that column. This is the number that says
        whether a projection means anything: a depth contact directly beneath a
        gyral crown has a small offset however deep it is, while a contact in a
        ventricle or outside the head does not.
    max_above_pia_mm : float
        How far *outside* the pial surface an electrode may sit and still be
        placed. Subdural grid and strip contacts are legitimately a millimetre
        or two above the pia; a contact ten centimetres above it is a
        coordinate-space error.
    max_below_wm_mm : float or None
        The same bound below the white-matter boundary, or None -- the default
        -- for no bound, since sEEG contacts are legitimately deep.
    drop_unknown_anatomy : bool
        Apply the anatomy rule at all. Off by default.
    unknown_labels : sequence of str
        Anatomical labels counted as unknown, compared case-insensitively.
    """

    max_offset_mm: float = 10.0
    max_above_pia_mm: float = 10.0
    max_below_wm_mm: Optional[float] = None
    drop_unknown_anatomy: bool = False
    unknown_labels: Sequence[str] = ("", "unknown", "none", "n/a", "nan")


@dataclass
class ElectrodeAnchors:
    """Where each electrode sits on the surface, in surface-relative terms.

    Every array is ``n_electrodes`` long and index-aligned with the
    :class:`~cortex.electrodes.ElectrodeSet` that produced it.

    Attributes
    ----------
    hemi : (n,) array of str
        ``"lh"`` or ``"rh"``, chosen as whichever hemisphere's surface the
        electrode is nearer to, so an unlabelled input still resolves.
    verts : (n, 3) array of int
        The three vertices of the triangle the electrode sits over.

        Vertices rather than a face index, because a face index is *not*
        shared between a subject's surfaces: flattening cuts the medial wall
        away, so the flat surface has thousands fewer triangles than the pial
        one and face ``k`` is a different triangle on each. Vertex indexing is
        shared by every surface, which is the only thing that makes an anchor
        portable.
    weights : (n, 3) array of float
        Barycentric weights over ``verts``, summing to 1. Valid on every
        surface of the subject, which is what makes the electrode follow
        inflation and flattening.
    depth : (n,) array of float
        Normalised pia-to-white-matter depth, unclamped. NaN without a
        white-matter surface.
    depth_mm : (n,) array of float
        The same quantity in millimetres from the pia, signed outward-negative.
    thickness_mm : (n,) array of float
        Local cortical thickness at the anchor, which is what ``depth`` is
        normalised by.
    offset_mm : (n,) array of float
        Distance from the electrode to the cortical column it was assigned to,
        measured perpendicular to that column. Falls back to the distance to
        the mid-surface triangle when there is no white matter to define a
        column.
    placement : (n,) array of str
        One of the module-level placement constants.
    surface_hash : str
        Identifies the surfaces these anchors were computed against, so a
        cached anchor set can be recognised as stale.
    """

    hemi: npt.NDArray[np.str_]
    verts: npt.NDArray[np.intp]
    weights: npt.NDArray[np.floating]
    depth: npt.NDArray[np.floating]
    depth_mm: npt.NDArray[np.floating]
    thickness_mm: npt.NDArray[np.floating]
    offset_mm: npt.NDArray[np.floating]
    placement: npt.NDArray[np.str_]
    surface_hash: str = ""

    def __len__(self) -> int:
        return len(self.verts)

    def __getitem__(self, index: Union[int, slice, npt.NDArray]) -> "ElectrodeAnchors":
        idx = np.atleast_1d(np.asarray(index)) if isinstance(index, int) else index
        return ElectrodeAnchors(
            hemi=self.hemi[idx],
            verts=self.verts[idx],
            weights=self.weights[idx],
            depth=self.depth[idx],
            depth_mm=self.depth_mm[idx],
            thickness_mm=self.thickness_mm[idx],
            offset_mm=self.offset_mm[idx],
            placement=self.placement[idx],
            surface_hash=self.surface_hash,
        )

    @property
    def placeable(self) -> npt.NDArray[np.bool_]:
        """Electrodes the policy accepted -- everything not excluded."""
        return (self.placement == ON_SURFACE) | (self.placement == PROJECTED)

    def evaluate(
        self, surfaces: Mapping[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Positions on some other surface of the same subject.

        Parameters
        ----------
        surfaces : mapping
            ``{"lh": pts, "rh": pts}`` for the target surface -- fiducial,
            inflated, flat, anything sharing the subject's vertex indexing.
            Only the points are needed: the anchor already names its triangle
            by vertex, so the target surface's polygons are irrelevant, which
            is what lets a flat surface with a cut medial wall be evaluated at
            all.

        Returns
        -------
        (n, 3) array
            One position per electrode. Electrodes the policy excluded still
            get a position; filter on :attr:`placeable` if you do not want them.

        Notes
        -----
        Purely barycentric: the electrode lands *on* the target surface rather
        than at its own depth, because depth has no geometric meaning once the
        surface is inflated or flat. Depth is what decides whether an electrode
        is drawn at all, not where it is drawn -- see the module docstring.
        """
        out = np.full((len(self), 3), np.nan)
        for hemi in HEMIS:
            if hemi not in surfaces:
                continue
            sel = self.hemi == hemi
            if not sel.any():
                continue
            tri = np.asarray(surfaces[hemi])[self.verts[sel]]     # (m, 3, 3)
            out[sel] = np.einsum("mij,mi->mj", tri, self.weights[sel])
        return out

    def summary(self) -> str:
        """A one-line-per-outcome count, for reporting what a policy did."""
        lines = ["%d electrodes anchored:" % len(self)]
        for name in PLACEMENTS:
            count = int((self.placement == name).sum())
            if count:
                lines.append("  %-16s %d" % (name, count))
        return "\n".join(lines)


def _vertex_faces(
    polys: npt.NDArray[np.integer], nverts: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """CSR-style vertex-to-face adjacency: ``(indptr, indices)``.

    Built by hand rather than through :mod:`scipy.sparse` because only the
    adjacency lists are wanted, and a sparse matrix would allocate a values
    array the same size for nothing.
    """
    rows = np.asarray(polys).ravel()
    cols = np.repeat(np.arange(len(polys), dtype=np.intp), 3)
    order = np.argsort(rows, kind="stable")
    indices = cols[order]
    indptr = np.searchsorted(rows[order], np.arange(nverts + 1))
    return indptr.astype(np.intp), indices


def _closest_point_weights(
    tri: npt.NDArray[np.floating], point: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Barycentric weights of the closest point to ``point`` on each triangle.

    ``tri`` is ``(F, 3, 3)`` -- F triangles, three vertices, three coordinates.
    Returns ``(F, 3)`` weights that sum to 1, so the closest point itself is
    ``einsum("fij,fi->fj", tri, weights)``.

    The seven-region Voronoi test from Ericson, *Real-Time Collision Detection*
    section 5.1.5, evaluated for every triangle and resolved by priority rather
    than branched. Vertex and edge regions matter here: an electrode above a
    gyral crown projects into a triangle's interior, but one over a sulcal fold
    often lands on an edge, and clamping to the interior would slide it.
    """
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a

    dot = lambda u, v: np.einsum("ij,ij->i", u, v)  # noqa: E731
    ap, bp, cp = point - a, point - b, point - c
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _ratio(num: npt.NDArray, den: npt.NDArray) -> npt.NDArray:
        """num/den, zero where the region is degenerate -- masked out anyway."""
        return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    v_ab = _ratio(d1, d1 - d3)
    w_ac = _ratio(d2, d2 - d6)
    w_bc = _ratio(d4 - d3, (d4 - d3) + (d5 - d6))

    zero = np.zeros(len(tri))
    one = np.ones(len(tri))

    denom = va + vb + vc
    v_in = _ratio(vb, denom)
    w_in = _ratio(vc, denom)
    weights = np.stack([1 - v_in - w_in, v_in, w_in], axis=1)

    regions = [
        ((d1 <= 0) & (d2 <= 0), np.stack([one, zero, zero], axis=1)),
        ((d3 >= 0) & (d4 <= d3), np.stack([zero, one, zero], axis=1)),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - v_ab, v_ab, zero], axis=1)),
        ((d6 >= 0) & (d5 <= d6), np.stack([zero, zero, one], axis=1)),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - w_ac, zero, w_ac], axis=1)),
        (
            (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
            np.stack([zero, 1 - w_bc, w_bc], axis=1),
        ),
    ]
    # Reversed so that an earlier region wins, matching the reference's
    # short-circuiting if-chain.
    for condition, region_weights in reversed(regions):
        weights = np.where(condition[:, None], region_weights, weights)
    return weights


def _anchor_one_hemisphere(
    coords: npt.NDArray[np.floating], pair: SurfacePair, neighbours: int
) -> dict[str, npt.NDArray]:
    """Anchor every electrode to one hemisphere, whether or not it belongs there.

    Both hemispheres are anchored unconditionally and the nearer one wins later,
    which is how an electrode with no ``hemisphere`` column still resolves, and
    how a mislabelled one is corrected rather than trusted.
    """
    from scipy.spatial import cKDTree

    pia = np.asarray(pair.pia, dtype=np.float64)
    polys = np.asarray(pair.polys)
    wm = None if pair.wm is None else np.asarray(pair.wm, dtype=np.float64)

    # Search against the mid-surface. The pial and white-matter surfaces bound
    # the ribbon, so their midpoint is the least biased place to look for the
    # column an electrode belongs to; searching on the pia alone pulls
    # superficial contacts onto the near bank of a sulcus.
    mid = pia if wm is None else (pia + wm) / 2.0

    tree = cKDTree(mid)
    _, nearest = tree.query(coords, k=min(neighbours, len(mid)))
    # cKDTree drops the k axis entirely when k == 1, which a tiny test surface
    # can hit; keep it two-dimensional so the candidate gather below is uniform.
    nearest = np.asarray(nearest)
    if nearest.ndim == 1:
        nearest = nearest[:, None]

    indptr, indices = _vertex_faces(polys, len(mid))

    n = len(coords)
    face = np.zeros(n, dtype=np.intp)
    weights = np.zeros((n, 3))
    distance = np.full(n, np.inf)

    for i in range(n):
        cand = np.unique(
            np.concatenate([indices[indptr[v]:indptr[v + 1]] for v in nearest[i]])
        )
        w = _closest_point_weights(mid[polys[cand]], coords[i])
        foot = np.einsum("fij,fi->fj", mid[polys[cand]], w)
        dist = np.linalg.norm(foot - coords[i], axis=1)
        best = int(np.argmin(dist))
        face[i] = cand[best]
        weights[i] = w[best]
        distance[i] = dist[best]

    verts = polys[face]
    pial_point = np.einsum("nij,ni->nj", pia[verts], weights)

    if wm is None:
        nan = np.full(n, np.nan)
        return dict(
            verts=verts, weights=weights, distance=distance,
            depth=nan, depth_mm=nan.copy(), thickness_mm=nan.copy(),
            offset_mm=distance.copy(),
        )

    wm_point = np.einsum("nij,ni->nj", wm[verts], weights)
    column = wm_point - pial_point
    thickness = np.linalg.norm(column, axis=1)

    rel = coords - pial_point
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.einsum("ni,ni->n", rel, column) / thickness ** 2
    depth = np.where(thickness > 0, depth, np.nan)
    perpendicular = rel - depth[:, None] * column
    offset = np.linalg.norm(perpendicular, axis=1)

    return dict(
        verts=verts, weights=weights, distance=distance,
        depth=depth, depth_mm=depth * thickness, thickness_mm=thickness,
        offset_mm=np.where(np.isfinite(offset), offset, distance),
    )


def surface_hash(hemis: Mapping[str, SurfacePair]) -> str:
    """A short digest of the surfaces an anchor set was computed against."""
    digest = hashlib.sha1()
    for hemi in sorted(hemis):
        pair = hemis[hemi]
        digest.update(hemi.encode("utf-8"))
        for array in (pair.pia, pair.polys, pair.wm):
            if array is not None:
                digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def anchor_to_surfaces(
    coords: npt.NDArray[np.floating],
    hemis: Mapping[str, SurfacePair],
    policy: Optional[PlacementPolicy] = None,
    anatomy: Optional[npt.NDArray[np.str_]] = None,
    neighbours: int = 8,
) -> ElectrodeAnchors:
    """Anchor electrode coordinates to a subject's cortical surfaces.

    The geometry entry point, taking surfaces directly so it can be tested and
    used without a pycortex database. :meth:`cortex.electrodes.ElectrodeSet.anchor`
    is the version that loads the surfaces for a subject.

    Parameters
    ----------
    coords : (n, 3) array
        Electrode coordinates, in the same space as the surfaces -- TkRegRAS,
        for surfaces imported from FreeSurfer.
    hemis : mapping
        ``{"lh": SurfacePair(...), "rh": SurfacePair(...)}``. One hemisphere is
        allowed.
    policy : PlacementPolicy, optional
        Defaults to :class:`PlacementPolicy` with its own defaults.
    anatomy : (n,) array of str, optional
        Anatomical labels, read only when the policy's anatomy rule is on.
    neighbours : int
        How many nearby vertices seed the candidate faces for each electrode.
        Eight covers roughly a two-ring neighbourhood, which is ample; raise it
        only for unusually coarse surfaces.

    Returns
    -------
    ElectrodeAnchors
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    if coords.shape[1] != 3:
        raise ValueError("coords must be (n, 3), got %r" % (coords.shape,))
    if not hemis:
        raise ValueError("at least one hemisphere is needed to anchor electrodes")
    for hemi in hemis:
        if hemi not in HEMIS:
            raise ValueError("hemisphere must be one of %r, got %r" % (HEMIS, hemi))
    policy = policy or PlacementPolicy()

    per_hemi = {h: _anchor_one_hemisphere(coords, hemis[h], neighbours) for h in hemis}

    # Whichever hemisphere's mid-surface the electrode is closer to.
    names = sorted(per_hemi)
    stacked = np.stack([per_hemi[h]["distance"] for h in names], axis=1)
    winner = np.argmin(stacked, axis=1)

    n = len(coords)
    take = lambda key: np.stack(  # noqa: E731
        [per_hemi[h][key] for h in names], axis=1
    )[np.arange(n), winner]

    hemi_of = np.array([names[w] for w in winner], dtype="<U2")
    stack2d = lambda key: np.stack(  # noqa: E731
        [per_hemi[h][key] for h in names], axis=1
    )[np.arange(n), winner]

    anchors = ElectrodeAnchors(
        hemi=hemi_of,
        verts=stack2d("verts").astype(np.intp),
        weights=stack2d("weights"),
        depth=take("depth"),
        depth_mm=take("depth_mm"),
        thickness_mm=take("thickness_mm"),
        offset_mm=take("offset_mm"),
        placement=np.full(n, ON_SURFACE, dtype="<U16"),
        surface_hash=surface_hash(hemis),
    )
    anchors.placement = classify_placement(anchors, policy, anatomy)
    return anchors


def classify_placement(
    anchors: ElectrodeAnchors,
    policy: Optional[PlacementPolicy] = None,
    anatomy: Optional[npt.NDArray[np.str_]] = None,
) -> npt.NDArray[np.str_]:
    """Apply a placement policy to an anchored set, returning the outcomes.

    Separate from :func:`anchor_to_surfaces` because the anchor is expensive
    geometry and the policy is a cheap decision: changing a threshold should
    not mean re-running a nearest-face search over a whole hemisphere.
    """
    policy = policy or PlacementPolicy()
    n = len(anchors)

    depth = anchors.depth
    known_depth = np.isfinite(depth)
    inside = known_depth & (depth >= 0) & (depth <= 1)

    placement = np.where(inside, ON_SURFACE, PROJECTED).astype("<U16")

    too_far = anchors.offset_mm > policy.max_offset_mm
    above = -np.minimum(anchors.depth_mm, 0.0)          # mm outside the pia
    too_far |= np.nan_to_num(above) > policy.max_above_pia_mm
    if policy.max_below_wm_mm is not None:
        below = np.maximum(anchors.depth_mm - anchors.thickness_mm, 0.0)
        too_far |= np.nan_to_num(below) > policy.max_below_wm_mm
    placement[too_far] = TOO_FAR

    if policy.drop_unknown_anatomy:
        if anatomy is None:
            raise ValueError(
                "policy.drop_unknown_anatomy is set but no anatomy labels were given"
            )
        labels = np.array([str(a).strip().lower() for a in anatomy])
        unknown = np.isin(labels, [u.lower() for u in policy.unknown_labels])
        if len(unknown) != n:
            raise ValueError(
                "anatomy has %d entries but there are %d electrodes" % (len(unknown), n)
            )
        placement[unknown & ~too_far] = UNKNOWN_ANATOMY

    return placement


@dataclass
class AlignmentReport:
    """Evidence about whether electrode coordinates and surfaces share a space.

    Worth running before trusting any anchor. pycortex keeps FreeSurfer's own
    surface RAS -- ``freesurfer.parse_surf`` applies no ``c_ras`` shift -- so
    TkRegRAS coordinates should already land on the surfaces. When they do not,
    the error is systematic, and that is what makes it hard to see: one
    displaced electrode and a whole misregistered grid look alike when
    inspected one at a time.

    What this can and cannot detect, measured rather than assumed:

    - A shift that lifts electrodes **off** the cortical sheet is caught
      easily. On a real subject, a 15 mm error raises the median offset of a
      subdural grid from 1.6 mm to 12 mm.
    - A shift **along** the sheet is not caught at all, and cannot be: the grid
      slides to a different gyrus and sits just as snugly on it. An 8 mm
      tangential error leaves the median offset at 1.0 mm -- lower than the
      correct placement's. Nothing in the geometry distinguishes it; only the
      anatomical labels can.
    - Cortex folds, so a *scattered* set is far more forgiving than a
      contiguous one -- every stray point finds some nearby bank. Judge a
      montage by its grids and strips, not by its outliers.

    Depth electrodes are legitimately centimetres from the pia, so a montage
    that is mostly sEEG will report a large median offset and read as
    suspicious when it is fine. Run this on the surface contacts:
    ``check_alignment(eset.select(group_type=["grid", "strip"]).coords, ...)``.
    """

    median_offset_mm: float
    max_offset_mm: float
    systematic_shift_mm: npt.NDArray[np.floating] = field(
        default_factory=lambda: np.zeros(3)
    )
    n_electrodes: int = 0
    suspicious: bool = False

    @property
    def shift_magnitude_mm(self) -> float:
        """How much of the offset is a common direction rather than scatter.

        Close to the median offset when every electrode is displaced the same
        way, which is the signature of a coordinate-space error rather than of
        electrodes that are genuinely at different depths.
        """
        return float(np.linalg.norm(self.systematic_shift_mm))

    def summary(self) -> str:
        lines = [
            "%d electrodes vs. the pial surface:" % self.n_electrodes,
            "  median offset     %.2f mm" % self.median_offset_mm,
            "  max offset        %.2f mm" % self.max_offset_mm,
            "  systematic shift  (%.2f, %.2f, %.2f) mm, |%.2f|"
            % (*self.systematic_shift_mm, self.shift_magnitude_mm),
        ]
        if self.suspicious:
            lines.append(
                "  SUSPICIOUS: electrodes sit this far off the surface only when "
                "the coordinates are not in the surfaces' space (a c_ras shift, "
                "or scanner rather than surface RAS) -- or when the montage is "
                "mostly depth electrodes, which are legitimately deep."
            )
        return "\n".join(lines)


def check_alignment(
    coords: npt.NDArray[np.floating],
    hemis: Mapping[str, SurfacePair],
    threshold_mm: float = 10.0,
) -> AlignmentReport:
    """Test whether ``coords`` plausibly live in the same space as ``hemis``.

    See :class:`AlignmentReport` for what this does and does not catch -- in
    particular that it is blind to a shift along the cortical sheet.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    anchors = anchor_to_surfaces(
        coords, hemis,
        policy=PlacementPolicy(max_offset_mm=np.inf, max_above_pia_mm=np.inf),
    )
    residuals = coords - anchors.evaluate({h: hemis[h].pia for h in hemis})
    offsets = np.linalg.norm(residuals, axis=1)
    return AlignmentReport(
        median_offset_mm=float(np.median(offsets)),
        max_offset_mm=float(np.max(offsets)),
        systematic_shift_mm=residuals.mean(0),
        n_electrodes=len(coords),
        suspicious=bool(np.median(offsets) > threshold_mm),
    )
