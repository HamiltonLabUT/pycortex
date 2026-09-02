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
and the viewer's depth slider are the same number, and the slider *can* be asked
to select which electrodes are drawn.

It does not by default, and that is deliberate. The slider starts mid-ribbon,
where a subdural contact never is, so letting it gate visibility means the
montage arrives with a third of itself missing and nothing said about it. What
decides whether a contact is drawn is the placement policy, in python, where it
is configurable and reports what it excluded.

Nothing here mutates the electrode's TkRegRAS coordinate. Anchors are derived,
carry a hash of the surfaces they were computed against, and can be thrown away
and recomputed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Union

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
"""Further from cortex than the policy allows -- no cortical column can
honestly claim it. By default this means more than
:attr:`PlacementPolicy.max_surface_distance_mm` from the nearest point on
*either* bounding surface, pial or white matter."""

UNKNOWN_ANATOMY = "unknown_anatomy"
"""Excluded by the anatomy rule rather than by geometry."""

NON_CORTICAL = "non_cortical"
"""Labelled as white matter or a subcortical structure.

Not a geometric outcome -- the anchor is perfectly good -- but a statement that
the *label* says there is no cortex here to be on. Real montages carry these in
quantity: a depth electrode's contacts are routinely labelled
``Left-Cerebral-White-Matter``, ``Left-Hippocampus`` or ``Left-Putamen``, and
the surface position such a contact projects to is a rough locality rather than
a location. The geometry cannot tell -- a contact in white matter still has
cortex a millimetre away on some sulcal bank -- so this is the one thing only
the anatomy column knows."""

NO_COORDINATE = "no_coordinate"
"""No usable coordinate: the row had a non-finite x, y or z.

Montages carry placeholder rows for unconnected amplifier channels, with NaN
coordinates and names like ``NaN1``. They are kept rather than dropped so the
row indices still line up with a data array recorded on the same channels."""

PLACEMENTS = (ON_SURFACE, PROJECTED, TOO_FAR, UNKNOWN_ANATOMY,
              NON_CORTICAL, NO_COORDINATE)

#: How :meth:`ElectrodeAnchors.evaluate` treats an electrode's distance from the
#: surface it is anchored to.
OFFSET_NONE = "none"
"""Land the electrode on the target surface -- the original behaviour, and still
the default. Everything that is not on cortex is flattened onto it."""

OFFSET_FRAME = "frame"
"""Keep the electrode off the surface, at the position its residual describes.

The residual is stored as three signed millimetres in the orthonormal frame of
its anchor triangle (see :func:`frame_components`), and that frame exists on
every surface of the subject because the triangle is named by vertex. So the
residual can be re-expressed on the inflated or flat surface without ever
interpolating anything across a sulcus."""

OFFSETS = (OFFSET_NONE, OFFSET_FRAME)

#: How the residual is scaled when the surface it rides on stretches.
SCALE_AUTO = "auto"
"""``similarity`` where the frame is shared by a device, ``anisotropic`` where
each contact carries its own. The right default for a mixed montage: a depth
electrode keeps its shape, a grid keeps its standoff in millimetres."""

SCALE_SIMILARITY = "similarity"
"""Scale all three axes by the local linear expansion. A similarity transform,
so the device's shape and entry angle are preserved exactly and its size follows
the cortex around it -- neighbours stay evenly spaced, and that spacing grows or
shrinks with inflation."""

SCALE_ANISOTROPIC = "anisotropic"
"""Scale the two tangential axes and leave the normal axis in literal
millimetres. A shear, and only sound per contact: applied to a shared device
frame it tilts a shaft off its trajectory and makes its pitch uneven."""

SCALE_RIGID = "rigid"
"""Leave all three axes in literal millimetres. For when the offset must mean
millimetres regardless of what the surface did."""

SCALE_MODES = (SCALE_AUTO, SCALE_SIMILARITY, SCALE_ANISOTROPIC, SCALE_RIGID)

#: Whether the contacts of one device share a frame.
ANCHOR_PER_CONTACT = "per_contact"
"""Every contact carries its own anchor. Right for a grid or strip, which drapes
over folds and whose contacts each genuinely sit on their own column."""

ANCHOR_PER_DEVICE = "per_device"
"""One anchor for the whole device; every contact's residual is expressed in that
frame. Right for a shaft, whose contacts do *not* each belong to the cortex
nearest them -- a depth electrode threads folds, so per-contact anchoring makes
consecutive contacts hop between banks."""

ANCHOR_AUTO = "auto"
""":data:`ANCHOR_PER_DEVICE` for ``seeg`` and ``depth``, :data:`ANCHOR_PER_CONTACT`
for everything else, keyed on the group type."""

ANCHOR_MODES = (ANCHOR_AUTO, ANCHOR_PER_CONTACT, ANCHOR_PER_DEVICE)

#: Group types whose contacts lie along a single rigid shank, and which
#: :data:`ANCHOR_AUTO` therefore anchors once. Matches ``_connect.LINEAR_TYPES``.
SHARED_FRAME_TYPES = ("seeg", "depth")

STRAIGHT_TOLERANCE_MM = 2.0
"""How far a device's contacts may sit from their own best-fit line and still be
read as a rigid shank, when the montage does not say what kind of device it is.

A real montage often does not. Of the three clinical montages in this
filestore, two carry no ``group_type`` at all -- and every one of TCH06's
twenty-one groups is a depth electrode. Without a fallback those montages get
per-contact anchoring, silently, which is the case
:func:`regroup_anchors` exists to prevent.

The geometry separates the two cleanly, because a depth electrode is a rigid
needle and nothing else in the vocabulary is. Measured over those montages:

===============================  ==================
device                           max off-axis
===============================  ==================
27 depth shafts (TCH06, S0033)   0.00 - 0.45 mm
3 depth shafts (S0019)           0.65 - 0.95 mm
64-contact grid (S0019 ``LG``)   49.04 mm
===============================  ==================

Two orders of magnitude, so the threshold is not delicate. It sits above the
0.95 mm of the least straight real shaft -- localisation from post-implant
imaging is not exact -- and far below anything that drapes.

A strip is linear but *not* straight: it follows the convexity it lies on, so
it fails this test, correctly. Were a short strip on flat cortex to pass, the
cost is small -- a strip hugs the surface either way -- while a shaft that
fails to group scatters over tens of millimetres. The asymmetry is why the
tolerance is generous rather than tight.

An explicit ``group_type`` is always believed; this decides only when there is
none.
"""


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

    One number decides it by default: how far from cortex a contact may be and
    still be worth drawing, as :attr:`max_surface_distance_mm`. The three
    remaining geometric bounds are off unless set, and each asks a narrower
    question than that one does.

    The rule is geometric first. The design document's original formulation --
    drop an electrode whose anatomical label is ``Unknown`` -- keys on a field
    that is *optional* in the data specification, so with unlabelled or
    atypical anatomy it discards contacts that are merely undocumented. Here
    the anatomy rule is one configurable term (off by default) on top of a
    distance test, and everything it excludes is reported rather than removed.

    Parameters
    ----------
    max_surface_distance_mm : float
        How far an electrode may sit from the nearest point on a bounding
        cortical surface -- pial *or* white matter, whichever is closer -- and
        still be projected. This is the rule that says whether a projection
        means anything: a contact this close to cortex has a column to be drawn
        on, and one further away does not.

        Four millimetres by default. Measured on a real subject, a subdural
        grid resting on the pia stays under 1.6 mm and a depth electrode driven
        56 mm inward stays under 2.3 mm -- cortex folds, so a shaft is never
        far from some sulcal bank. What four millimetres excludes is a contact
        in the deep white-matter core, in a ventricle, or outside the head.

        Both directions are bounded, so an sEEG contact genuinely far from any
        grey matter is excluded rather than drawn on a column it has no claim
        to. Raise it, or set it to ``np.inf``, to project everything.

        This is not a registration check, and the same folding that makes it
        generous is why. A *contiguous* grid lifted 15 mm off the convexity it
        rested on fails every contact, but a *scattered* montage shifted the
        same 15 mm keeps more than half of them, because each stray point finds
        some bank to land near. Use :func:`check_alignment` to ask whether the
        coordinates are in the surfaces' space; this asks only whether an
        individual contact has cortex near enough to be drawn on.
    max_offset_mm : float or None
        How far an electrode may sit from the cortical column it is assigned
        to, measured perpendicular to that column. None -- the default -- for
        no bound: the surface-distance rule above already excludes anything
        this would catch, and it does so by a measure that survives folding.
        Kept because it is the honest way to ask "is this contact over its own
        column?", which is a different question from "is it near cortex?".
    max_above_pia_mm : float or None
        How far *outside* the pial surface an electrode may sit and still be
        placed, or None -- the default -- for no bound beyond the
        surface-distance rule. Subdural grid and strip contacts are
        legitimately a millimetre or two above the pia.
    max_below_wm_mm : float or None
        The same bound below the white-matter boundary, or None -- the default
        -- for no bound beyond the surface-distance rule. Unlike that rule this
        one is measured along the electrode's own column, so it asks "how deep
        is this contact?" rather than "how far from cortex is it?".
    drop_unknown_anatomy : bool
        Apply the anatomy rule at all. Off by default.
    unknown_labels : sequence of str
        Anatomical labels counted as unknown, compared case-insensitively.
    flag_non_cortical : bool
        Mark electrodes whose anatomical label names white matter or a
        subcortical structure as :data:`NON_CORTICAL`. On by default, because
        the alternative is drawing them as though they were on cortex. They are
        still *placed*: marking is not dropping, and ``select(placeable=True)``
        keeps them, since where a hippocampal contact projects to is often
        exactly what a reader wants to see.
    non_cortical_patterns : sequence of str
        Substrings that make a label non-cortical, matched case-insensitively.
        The defaults cover FreeSurfer's aseg vocabulary -- ``Left-Hippocampus``,
        ``Left-Cerebral-White-Matter`` and the rest -- which is what appears in
        an anatomy column alongside cortical parcel names. Destrieux and
        Desikan-Killiany cortical labels (``superiortemporal``,
        ``ctx_lh_G_temporal_inf``) deliberately do not match.
    """

    max_surface_distance_mm: float = 4.0
    max_offset_mm: Optional[float] = None
    max_above_pia_mm: Optional[float] = None
    max_below_wm_mm: Optional[float] = None
    drop_unknown_anatomy: bool = False
    unknown_labels: Sequence[str] = ("", "unknown", "none", "n/a", "nan")
    flag_non_cortical: bool = True
    non_cortical_patterns: Sequence[str] = (
        "white-matter", "white_matter", "wm",
        "left-", "right-",          # FreeSurfer aseg subcortical labels
        "hippocampus", "amygdala", "putamen", "thalamus", "caudate",
        "pallidum", "accumbens", "ventricle", "ventraldc", "brain-stem",
        "csf", "vessel", "choroid",
    )


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
    dist_pia_mm : (n,) array of float
        Distance to the nearest point on the pial surface.
    dist_wm_mm : (n,) array of float
        Distance to the nearest point on the white-matter surface. NaN when the
        subject has none, in which case :attr:`surface_distance_mm` falls back
        to the pial distance alone.
    placement : (n,) array of str
        One of the module-level placement constants.
    surface_hash : str
        Identifies the surfaces these anchors were computed against, so a
        cached anchor set can be recognised as stale.
    frame : (n, 3) array of float, optional
        The electrode's residual from its anchor point, in signed millimetres
        along its anchor triangle's orthonormal basis -- see
        :func:`frame_components`. This is the direction ``offset_mm`` and
        ``depth`` throw away, and it is what lets :meth:`evaluate` put an
        electrode *off* the surface rather than on it.

        ``None`` on anchors computed before this existed, in which case
        :meth:`evaluate` supports only :data:`OFFSET_NONE`.
    frame_scale_mm : (n,) array of float, optional
        Size of each electrode's own anchor triangle on the surface the frame
        was measured against, as ``sqrt(2 * area)``. The denominator of the
        local scale factor.
    anchor_index : (n,) array of int, optional
        Which electrode's triangle supplies the frame for each electrode.
        ``arange(n)`` -- each its own -- unless :func:`regroup_anchors` has
        given a device a shared frame. Electrodes sharing a value are one rigid
        device, which is also how :meth:`evaluate` knows to scale them together.
    """

    hemi: npt.NDArray[np.str_]
    verts: npt.NDArray[np.intp]
    weights: npt.NDArray[np.floating]
    depth: npt.NDArray[np.floating]
    depth_mm: npt.NDArray[np.floating]
    thickness_mm: npt.NDArray[np.floating]
    offset_mm: npt.NDArray[np.floating]
    dist_pia_mm: npt.NDArray[np.floating]
    dist_wm_mm: npt.NDArray[np.floating]
    placement: npt.NDArray[np.str_]
    surface_hash: str = ""
    frame: Optional[npt.NDArray[np.floating]] = None
    frame_scale_mm: Optional[npt.NDArray[np.floating]] = None
    anchor_index: Optional[npt.NDArray[np.intp]] = None

    def __len__(self) -> int:
        return len(self.verts)

    def __getitem__(self, index: Union[int, slice, npt.NDArray]) -> "ElectrodeAnchors":
        idx = np.atleast_1d(np.asarray(index)) if isinstance(index, int) else index
        sub = lambda a: None if a is None else a[idx]  # noqa: E731
        # `anchor_index` names rows of the *full* set, so a subset's stored
        # indices are meaningless unless they are renumbered -- and a device
        # that was only partly selected has no anchor left to point at. Rebuild
        # it where the whole device survived and fall back to self-anchoring
        # where it did not, which is the honest answer: with the frame-defining
        # contact gone, the remaining contacts are no longer one rigid device.
        anchor_index = None
        if self.anchor_index is not None:
            keep = np.zeros(len(self), dtype=bool)
            keep[idx] = True
            renumber = np.full(len(self), -1, dtype=np.intp)
            renumber[keep] = np.arange(int(keep.sum()), dtype=np.intp)
            anchor_index = renumber[self.anchor_index[idx]]
            orphan = anchor_index < 0
            anchor_index[orphan] = np.nonzero(orphan)[0]
        return ElectrodeAnchors(
            hemi=self.hemi[idx],
            verts=self.verts[idx],
            weights=self.weights[idx],
            depth=self.depth[idx],
            depth_mm=self.depth_mm[idx],
            thickness_mm=self.thickness_mm[idx],
            offset_mm=self.offset_mm[idx],
            dist_pia_mm=self.dist_pia_mm[idx],
            dist_wm_mm=self.dist_wm_mm[idx],
            placement=self.placement[idx],
            surface_hash=self.surface_hash,
            frame=sub(self.frame),
            frame_scale_mm=sub(self.frame_scale_mm),
            anchor_index=anchor_index,
        )

    @property
    def hosts(self) -> npt.NDArray[np.intp]:
        """:attr:`anchor_index`, defaulted to each electrode anchoring itself."""
        if self.anchor_index is None:
            return np.arange(len(self), dtype=np.intp)
        return np.asarray(self.anchor_index, dtype=np.intp)

    @property
    def surface_distance_mm(self) -> npt.NDArray[np.floating]:
        """Distance to the nearer bounding surface, pial or white matter.

        The quantity :attr:`PlacementPolicy.max_surface_distance_mm` bounds.
        ``fmin`` rather than ``minimum``, so a subject with no white-matter
        surface degrades to the pial distance instead of returning NaN for
        every electrode.
        """
        return np.fmin(self.dist_pia_mm, self.dist_wm_mm)

    @property
    def placeable(self) -> npt.NDArray[np.bool_]:
        """Electrodes that have an honest surface position.

        :data:`NON_CORTICAL` counts as placeable: the anchor is sound, and where
        a hippocampal or white-matter contact projects to is usually exactly
        what a reader wants shown -- flagged, not hidden. Only
        :data:`TOO_FAR`, :data:`UNKNOWN_ANATOMY` and :data:`NO_COORDINATE` are
        excluded.
        """
        return np.isin(self.placement, [ON_SURFACE, PROJECTED, NON_CORTICAL])

    def evaluate(
        self,
        surfaces: Mapping[str, npt.NDArray[np.floating]],
        *,
        offset: str = OFFSET_NONE,
        scale_mode: str = SCALE_AUTO,
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
        offset : {"none", "frame"}
            :data:`OFFSET_NONE` lands the electrode *on* the surface, purely
            barycentrically. This is the original behaviour and remains the
            default, so nothing that does not ask for the new one changes.

            :data:`OFFSET_FRAME` puts it where it actually is, by re-expressing
            the stored :attr:`frame` residual in the target triangle's own
            basis. Evaluated against the surface the frame was measured on
            (the pia), this reproduces the input coordinate exactly.
        scale_mode : {"auto", "similarity", "anisotropic", "rigid"}
            How the residual is scaled when the surface stretches. Ignored for
            :data:`OFFSET_NONE`. See the module constants; :data:`SCALE_AUTO`
            picks per device and is what a mixed montage wants.

        Returns
        -------
        (n, 3) array
            One position per electrode. Electrodes the policy excluded still
            get a position; filter on :attr:`placeable` if you do not want them.

        Notes
        -----
        Under :data:`OFFSET_NONE` depth is not represented: it has no geometric
        meaning once every contact is pinned to the sheet, and it is left to
        decide whether an electrode is drawn rather than where. Under
        :data:`OFFSET_FRAME` it *is* represented, as real displacement off the
        surface, and the distinction between the two is the point of the
        argument.
        """
        if offset not in OFFSETS:
            raise ValueError("offset must be one of %r, got %r" % (OFFSETS, offset))
        if scale_mode not in SCALE_MODES:
            raise ValueError(
                "scale_mode must be one of %r, got %r" % (SCALE_MODES, scale_mode)
            )

        out = np.full((len(self), 3), np.nan)
        for hemi in HEMIS:
            if hemi not in surfaces:
                continue
            sel = self.hemi == hemi
            if not sel.any():
                continue
            tri = np.asarray(surfaces[hemi])[self.verts[sel]]     # (m, 3, 3)
            out[sel] = np.einsum("mij,mi->mj", tri, self.weights[sel])

        if offset == OFFSET_NONE:
            return out
        if self.frame is None or self.frame_scale_mm is None:
            raise ValueError(
                "offset=%r needs a frame, and these anchors have none. They were "
                "computed before frames existed, or by a caller that did not pass "
                "the coordinates. Re-run ElectrodeSet.anchor()." % (OFFSET_FRAME,)
            )

        hosts = self.hosts
        for hemi in HEMIS:
            if hemi not in surfaces:
                continue
            sel = np.nonzero(self.hemi == hemi)[0]
            if not len(sel):
                continue
            pts = np.asarray(surfaces[hemi], dtype=np.float64)

            # Local linear scale where each electrode itself sits, then one
            # scale per device: the median over its contacts. The median rather
            # than the frame-defining contact's own value, because that contact
            # is systematically unrepresentative -- a shaft is anchored near its
            # entry, on a gyral crown, and crowns inflate quite differently from
            # the tissue a shaft threads. Measured on S1 over three synthetic
            # 15-contact shafts, using the anchor's own scale rather than the
            # device median stretched two of the three by more than 60%.
            own_size = _triangle_basis(pts[self.verts[sel]])[3]
            with np.errstate(invalid="ignore", divide="ignore"):
                own_scale = own_size / self.frame_scale_mm[sel]

            scale = np.array(own_scale, dtype=np.float64)
            shared = np.zeros(len(sel), dtype=bool)
            for host in np.unique(hosts[sel]):
                member = hosts[sel] == host
                if member.sum() > 1:
                    shared |= member
                    with np.errstate(invalid="ignore"):
                        scale[member] = np.nanmedian(own_scale[member])

            host_tri = pts[self.verts[hosts[sel]]]
            t1, t2, normal, _ = _triangle_basis(host_tri)
            origin = np.einsum("mij,mi->mj", host_tri, self.weights[hosts[sel]])

            if scale_mode == SCALE_RIGID:
                tangential, perpendicular = np.ones_like(scale), np.ones_like(scale)
            elif scale_mode == SCALE_SIMILARITY:
                tangential, perpendicular = scale, scale
            elif scale_mode == SCALE_ANISOTROPIC:
                tangential, perpendicular = scale, np.ones_like(scale)
            else:  # SCALE_AUTO
                tangential = scale
                perpendicular = np.where(shared, scale, 1.0)

            f = self.frame[sel]
            out[sel] = (
                origin
                + (tangential * f[:, 0])[:, None] * t1
                + (tangential * f[:, 1])[:, None] * t2
                + (perpendicular * f[:, 2])[:, None] * normal
            )
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

    Candidate triangles come from the mid-surface, but the winner among them is
    chosen by distance to the *cortical column* -- the short segment from the
    pial point to the white-matter point at the same barycentric location --
    rather than by distance to the mid-surface itself. That distinction is worth
    the two extra lines: selecting on mid-surface distance biases depth toward
    the middle of the ribbon, because a contact sitting on the pia over a curved
    patch finds a nearer mid-surface point on a *neighbouring* column. Measured
    on S1 over 160 contacts at known depths, mid-surface selection gives a mean
    depth error of 0.073 and a worst case of 0.38; column selection gives 0.006
    and 0.17. Depth is the axis the viewer's slider rides on, so that error
    matters more than its size suggests.
    """
    from scipy.spatial import cKDTree

    pia = np.asarray(pair.pia, dtype=np.float64)
    polys = np.asarray(pair.polys)
    wm = None if pair.wm is None else np.asarray(pair.wm, dtype=np.float64)

    # The pial and white-matter surfaces bound the ribbon, so their midpoint is
    # the least biased place to look for candidates; searching on the pia alone
    # pulls superficial contacts onto the near bank of a sulcus.
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
    depth = np.full(n, np.nan)
    thickness = np.full(n, np.nan)
    lateral = np.full(n, np.nan)
    to_pia = np.full(n, np.nan)
    to_wm = np.full(n, np.nan)

    def nearest_on(surface: npt.NDArray[np.floating], tri: npt.NDArray[np.integer],
                   point: npt.NDArray[np.floating]) -> float:
        """Distance from ``point`` to the closest of ``surface``'s triangles.

        Over the candidate faces only, which is the mid-surface neighbourhood
        the search already gathered -- so this is the nearest point on the
        columns *near* the electrode rather than a global query over the whole
        hemisphere. The two differ only for an electrode already far outside
        any policy, where the answer is "too far" either way.
        """
        w = _closest_point_weights(surface[tri], point)
        foot = np.einsum("fij,fi->fj", surface[tri], w)
        return float(np.linalg.norm(foot - point, axis=1).min())

    for i in range(n):
        cand = np.unique(
            np.concatenate([indices[indptr[v]:indptr[v + 1]] for v in nearest[i]])
        )
        tri = polys[cand]
        w = _closest_point_weights(mid[tri], coords[i])

        if wm is None:
            foot = np.einsum("fij,fi->fj", mid[tri], w)
            dist = np.linalg.norm(foot - coords[i], axis=1)
            best = int(np.argmin(dist))
            face[i], weights[i], distance[i] = cand[best], w[best], dist[best]
            lateral[i] = dist[best]
            # `mid` *is* the pial surface here, so this is the distance to the
            # only surface there is; `to_wm` stays NaN and drops out of the
            # policy's fmin.
            to_pia[i] = dist[best]
            continue

        pial_point = np.einsum("fij,fi->fj", pia[tri], w)
        column = np.einsum("fij,fi->fj", wm[tri], w) - pial_point
        length2 = np.einsum("fi,fi->f", column, column)
        rel = coords[i] - pial_point
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.einsum("fi,fi->f", rel, column) / length2
        t = np.where(length2 > 0, t, 0.0)

        # Distance to the slab: perpendicular within the ribbon, and to the
        # nearer face of it outside.
        to_segment = np.linalg.norm(rel - np.clip(t, 0.0, 1.0)[:, None] * column, axis=1)
        best = int(np.argmin(to_segment))

        face[i] = cand[best]
        weights[i] = w[best]
        distance[i] = to_segment[best]
        depth[i] = t[best] if length2[best] > 0 else np.nan
        thickness[i] = np.sqrt(length2[best])
        lateral[i] = np.linalg.norm(rel[best] - t[best] * column[best])

        # Measured against the bounding surfaces themselves rather than derived
        # from `depth` and `lateral`, which would only be the distance to *this*
        # column's pial point. Cortex folds: a contact deep in white matter is
        # routinely a millimetre from a neighbouring sulcal bank while being
        # centimetres from its own column's pia, and it is the former that says
        # whether projecting it means anything.
        to_pia[i] = nearest_on(pia, tri, coords[i])
        to_wm[i] = nearest_on(wm, tri, coords[i])

    verts = polys[face]
    return dict(
        verts=verts,
        weights=weights,
        distance=distance,
        depth=depth,
        depth_mm=depth * thickness,
        thickness_mm=thickness,
        offset_mm=np.where(np.isfinite(lateral), lateral, distance),
        dist_pia_mm=to_pia,
        dist_wm_mm=to_wm,
    )


def _triangle_basis(
    tri: npt.NDArray[np.floating],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """An orthonormal frame per triangle, plus the edge it was built from.

    Parameters
    ----------
    tri : (m, 3, 3) array
        ``m`` triangles, each three points.

    Returns
    -------
    t1, t2, normal : (m, 3) arrays
        A right-handed orthonormal basis. ``t1`` runs along the ``v0 -> v1``
        edge and ``normal`` is the triangle's own normal.
    scale_mm : (m,) array
        ``sqrt(2 * area)``, a length-like measure of the triangle's size. Kept
        because it is the only thing that makes the frame comparable between two
        surfaces: the *ratio* of this quantity on the target to its value on the
        source is the local linear scale factor the residual rides on.

        From the area rather than from the ``v0 -> v1`` edge, even though that
        edge is what fixes ``t1``. A single edge's length is a noisy estimate of
        an isotropic quantity, and *which* edge is ``v0 -> v1`` is an artifact of
        how the triangle happened to be written down.

    Notes
    -----
    Built from the triangle's own edge rather than from an arbitrary reference
    direction, because that is what makes the basis *portable*. The same three
    vertex indices name a triangle on every surface of the subject, so a frame
    defined this way rotates and stretches with the surface, and re-expressing a
    residual in it never interpolates between two points that are near in space
    but far through the tissue.

    A degenerate triangle -- zero area, or a zero-length first edge -- yields a
    NaN frame rather than a fabricated one. The flat surface has genuinely
    degenerate triangles, since ``freesurfer._move_disconnect_points_to_zero``
    collapses unreferenced vertices to the origin.
    """
    tri = np.asarray(tri, dtype=np.float64)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]

    normal = np.cross(e1, e2)
    nlen = np.linalg.norm(normal, axis=1)
    e1len = np.linalg.norm(e1, axis=1)
    ok = (nlen > 0) & (e1len > 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        normal = normal / np.where(ok, nlen, 1.0)[:, None]
        t1 = e1 / np.where(ok, e1len, 1.0)[:, None]
        # e1 lies in the triangle's plane, so it is already perpendicular to the
        # normal and this subtraction is a no-op to floating point. Done anyway
        # so a nearly-degenerate triangle degrades smoothly rather than tilting
        # the frame out of the surface.
        t1 = t1 - np.einsum("mi,mi->m", t1, normal)[:, None] * normal
        t1len = np.linalg.norm(t1, axis=1)
        ok = ok & (t1len > 0)
        t1 = t1 / np.where(ok, t1len, 1.0)[:, None]

    t2 = np.cross(normal, t1)

    bad = ~ok
    t1[bad] = np.nan
    t2[bad] = np.nan
    normal[bad] = np.nan
    return t1, t2, normal, np.where(ok, np.sqrt(nlen), np.nan)


def frame_components(
    coords: npt.NDArray[np.floating],
    anchors: "ElectrodeAnchors",
    surfaces: Mapping[str, npt.NDArray[np.floating]],
    anchor_index: Optional[npt.NDArray[np.integer]] = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Each electrode's residual, in the orthonormal frame of its anchor triangle.

    This is the whole of what :data:`OFFSET_FRAME` needs to store, and it is
    three numbers: how far the electrode is from its anchor point along each
    axis of a basis the anchor triangle defines. Together with the barycentric
    anchor it is a *complete* description of the electrode's position -- unlike
    ``depth`` and ``offset_mm``, which record magnitudes and throw the direction
    away.

    Parameters
    ----------
    coords : (n, 3) array
        The electrode coordinates, in the same frame as ``surfaces``. For a
        subject whose surfaces were converted by ``mris_convert --to-scanner``
        this means TkRegRAS *plus* ``surface_space_offset`` -- the same shifted
        copy the anchoring itself was done against, never the raw column.
    anchors : ElectrodeAnchors
        Supplies ``verts``, ``weights`` and ``hemi``.
    surfaces : mapping
        ``{"lh": pts, "rh": pts}`` for the surface the residual is measured
        against. Anchoring measures depth from the pia, so this should be the
        pial surface if the two are to agree.
    anchor_index : (n,) array of int, optional
        Which electrode's triangle defines the frame for each electrode.
        Defaults to each its own. Anything else means a shared frame -- see
        :func:`regroup_anchors`.

    Returns
    -------
    frame : (n, 3) array
        Signed millimetres along ``(t1, t2, normal)`` of the anchor triangle.
    scale_mm : (n,) array
        Size of each electrode's *own* triangle on this surface, as
        ``sqrt(2 * area)``. Its own, not its anchor's, because this is what the
        local scale factor is measured from at evaluation time, and a device's
        scale is the median over its contacts' own neighbourhoods.
    """
    coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
    n = len(anchors)
    if len(coords) != n:
        raise ValueError(
            "coords has %d rows but there are %d anchors" % (len(coords), n)
        )
    if anchor_index is None:
        anchor_index = np.arange(n, dtype=np.intp)
    anchor_index = np.asarray(anchor_index, dtype=np.intp)

    frame = np.full((n, 3), np.nan)
    scale_mm = np.full(n, np.nan)

    for hemi in HEMIS:
        if hemi not in surfaces:
            continue
        sel = np.nonzero(anchors.hemi == hemi)[0]
        if not len(sel):
            continue
        pts = np.asarray(surfaces[hemi], dtype=np.float64)

        # The scale factor is a property of where the electrode itself sits, so
        # it comes from its own triangle even when the frame comes from another.
        scale_mm[sel] = _triangle_basis(pts[anchors.verts[sel]])[3]

        host = anchor_index[sel]
        tri = pts[anchors.verts[host]]
        t1, t2, normal, _ = _triangle_basis(tri)
        origin = np.einsum("mij,mi->mj", tri, anchors.weights[host])
        rel = coords[sel] - origin
        frame[sel, 0] = np.einsum("mi,mi->m", rel, t1)
        frame[sel, 1] = np.einsum("mi,mi->m", rel, t2)
        frame[sel, 2] = np.einsum("mi,mi->m", rel, normal)

    return frame, scale_mm


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

    # Placeholder rows for unconnected amplifier channels carry NaN coordinates,
    # and a NaN reaching cKDTree.query poisons the result rather than failing
    # loudly. Anchor only the real ones and fill the rest in afterwards, keeping
    # the rows, so indices still line up with data recorded on those channels.
    finite = np.isfinite(coords).all(axis=1)
    real = coords[finite]

    per_hemi = {h: _anchor_one_hemisphere(real, hemis[h], neighbours) for h in hemis}

    # Whichever hemisphere's cortical slab the electrode is closer to.
    names = sorted(per_hemi)
    stacked = np.stack([per_hemi[h]["distance"] for h in names], axis=1)
    winner = np.argmin(stacked, axis=1)

    m, n = len(real), len(coords)
    pick = lambda key: np.stack(  # noqa: E731
        [per_hemi[h][key] for h in names], axis=1
    )[np.arange(m), winner]

    def scatter(values: npt.NDArray, fill: Any) -> npt.NDArray:
        """Put per-real-electrode results back at their original row indices."""
        out = np.full((n,) + values.shape[1:], fill, dtype=values.dtype)
        out[finite] = values
        return out

    anchors = ElectrodeAnchors(
        hemi=scatter(np.array([names[w] for w in winner], dtype="<U2"), ""),
        verts=scatter(pick("verts").astype(np.intp), 0),
        weights=scatter(pick("weights"), np.nan),
        depth=scatter(pick("depth"), np.nan),
        depth_mm=scatter(pick("depth_mm"), np.nan),
        thickness_mm=scatter(pick("thickness_mm"), np.nan),
        offset_mm=scatter(pick("offset_mm"), np.nan),
        dist_pia_mm=scatter(pick("dist_pia_mm"), np.nan),
        dist_wm_mm=scatter(pick("dist_wm_mm"), np.nan),
        placement=np.full(n, ON_SURFACE, dtype="<U16"),
        surface_hash=surface_hash(hemis),
    )
    anchors.placement = classify_placement(anchors, policy, anatomy)
    anchors.placement[~finite] = NO_COORDINATE

    # The residual, measured against the pia so it agrees with `depth`, whose
    # zero is the pial surface. Cheap -- one triangle basis per electrode, no
    # search -- and it is what makes `evaluate(offset="frame")` possible at all.
    anchors.frame, anchors.frame_scale_mm = frame_components(
        coords, anchors, {h: hemis[h].pia for h in hemis}
    )
    return anchors


def is_straight(
    points: npt.NDArray[np.floating], tolerance_mm: float = STRAIGHT_TOLERANCE_MM
) -> bool:
    """Whether these contacts lie along one straight line, within a tolerance.

    How a device with no recorded ``group_type`` is recognised as a rigid shank.
    See :data:`STRAIGHT_TOLERANCE_MM` for the measurement the threshold comes
    from.

    Fewer than three finite points is not evidence of anything -- two points are
    always collinear -- so it answers False rather than letting a pair of
    contacts claim to be a device.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return False
    centred = points - points.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    residual = centred - np.outer(centred @ axis, axis)
    return bool(np.linalg.norm(residual, axis=1).max() <= tolerance_mm)


def regroup_anchors(
    anchors: ElectrodeAnchors,
    coords: npt.NDArray[np.floating],
    surfaces: Mapping[str, npt.NDArray[np.floating]],
    groups: Optional[Sequence[str]] = None,
    group_types: Optional[Sequence[str]] = None,
    mode: str = ANCHOR_AUTO,
    shared_frame_types: Sequence[str] = SHARED_FRAME_TYPES,
    straight_tolerance_mm: float = STRAIGHT_TOLERANCE_MM,
) -> ElectrodeAnchors:
    """Give each device's contacts one shared frame, where that is the right model.

    Per-contact anchoring assumes every contact belongs to the cortex nearest
    it. For a grid that is true. For a depth electrode it is false in a way that
    destroys the montage: a shaft threads folds, so consecutive contacts anchor
    to different sulcal banks and the device arrives as a cloud rather than a
    track. Measured on S1, a 15-contact shaft at a uniform 4 mm pitch evaluates
    on the inflated surface with consecutive spacings from 0.0 to 29.4 mm -- two
    contacts on the identical vertex, and one 4 mm step becoming 29.4 mm.

    Anchoring the device once and expressing every contact's residual in that
    one frame makes it a rigid body again: spacing is uniform by construction
    and the entry angle is exact.

    Separate from :func:`anchor_to_surfaces` for the same reason
    :func:`classify_placement` is -- so changing the grouping does not re-run
    the geometry.

    Parameters
    ----------
    anchors : ElectrodeAnchors
        Anchors that already carry a frame.
    coords : (n, 3) array
        The same coordinates the anchors were computed from, in the same frame
        -- shifted by ``surface_space_offset`` if the anchoring was.
    surfaces : mapping
        The surface the frames are measured against, i.e. the pia.
    groups : (n,) sequence of str, optional
        Device name per electrode. Without it nothing is shared and the anchors
        come back unchanged.
    group_types : (n,) sequence of str, optional
        ``grid`` / ``strip`` / ``seeg`` / ``depth``, consulted by
        :data:`ANCHOR_AUTO`. Where a group records none, :data:`ANCHOR_AUTO`
        falls back to :func:`is_straight` -- real montages routinely omit this
        column, and two of the three in this filestore do.
    mode : {"auto", "per_contact", "per_device"}
    shared_frame_types : sequence of str
        Which ``group_types`` :data:`ANCHOR_AUTO` treats as one rigid device.
    straight_tolerance_mm : float
        Passed to :func:`is_straight` for the unlabelled case.

    Returns
    -------
    ElectrodeAnchors
        A new set; the input is not modified.

    Notes
    -----
    Which contact of a device supplies the frame is decided by the smallest
    :attr:`offset_mm` -- the contact best explained by the column it sits over,
    and so the one whose triangle is least likely to be a bank it merely passed
    near. Deliberately not the shallowest contact: ``depth`` is not trustworthy
    for exactly the contacts this function exists to fix, since a deep contact
    anchored to a distant bank reports a depth relative to *that* bank. On the
    S1 shaft above, ``depth`` runs to -6.1 -- nominally outside the pia -- for
    contacts several centimetres inside the brain.

    A device is never allowed to span hemispheres: contacts are grouped by
    ``(group, hemi)``, so a mislabelled or genuinely bilateral group splits
    rather than anchoring one side to the other.
    """
    if mode not in ANCHOR_MODES:
        raise ValueError("mode must be one of %r, got %r" % (ANCHOR_MODES, mode))
    if mode == ANCHOR_PER_CONTACT or groups is None:
        return anchors

    n = len(anchors)
    groups = np.asarray(groups, dtype=object)
    if len(groups) != n:
        raise ValueError("groups has %d entries but there are %d anchors" % (len(groups), n))
    if group_types is None:
        types = np.array([""] * n, dtype=object)
    else:
        types = np.asarray(group_types, dtype=object)
        if len(types) != n:
            raise ValueError(
                "group_types has %d entries but there are %d anchors" % (len(types), n)
            )

    shared = {str(t).lower() for t in shared_frame_types}
    hosts = np.arange(n, dtype=np.intp)

    # A contact with no coordinate has no anchor worth sharing and no residual
    # to express; leave it self-anchored so it stays NaN rather than being
    # placed by its neighbours' frame.
    usable = anchors.placement != NO_COORDINATE

    for key in {(g, h) for g, h in zip(groups, anchors.hemi)}:
        member = np.nonzero(
            (groups == key[0]) & (anchors.hemi == key[1]) & usable
        )[0]
        if len(member) < 2:
            continue
        if mode == ANCHOR_AUTO:
            kinds = {str(t).lower() for t in types[member] if str(t).strip()}
            if kinds:
                # An explicit label is believed, either way.
                if not kinds & shared:
                    continue
            elif not is_straight(coords[member], straight_tolerance_mm):
                # No label. Fall back to the geometry, because a montage that
                # records no device type is common and is not a reason to give
                # its depth electrodes the treatment meant for grids.
                continue
        offsets = np.where(
            np.isfinite(anchors.offset_mm[member]), anchors.offset_mm[member], np.inf
        )
        hosts[member] = member[int(np.argmin(offsets))]

    frame, scale = frame_components(coords, anchors, surfaces, anchor_index=hosts)
    return replace(anchors, frame=frame, frame_scale_mm=scale, anchor_index=hosts)


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

    # The default rule, and normally the only one that fires: how far from
    # cortex the contact is, whichever bounding surface is nearer. Every test
    # here reads an unmeasured quantity as passing rather than failing --
    # `nan_to_num` on a distance that was never computed -- so an anchor from an
    # older file is kept and reported, not silently dropped.
    surface_distance = np.nan_to_num(anchors.surface_distance_mm)
    too_far = surface_distance > policy.max_surface_distance_mm

    if policy.max_offset_mm is not None:
        too_far |= np.nan_to_num(anchors.offset_mm) > policy.max_offset_mm
    if policy.max_above_pia_mm is not None:
        above = -np.minimum(anchors.depth_mm, 0.0)      # mm outside the pia
        too_far |= np.nan_to_num(above) > policy.max_above_pia_mm
    if policy.max_below_wm_mm is not None:
        below = np.maximum(anchors.depth_mm - anchors.thickness_mm, 0.0)
        too_far |= np.nan_to_num(below) > policy.max_below_wm_mm
    placement[too_far] = TOO_FAR

    if policy.drop_unknown_anatomy or policy.flag_non_cortical:
        if anatomy is None:
            if policy.drop_unknown_anatomy:
                raise ValueError(
                    "policy.drop_unknown_anatomy is set but no anatomy labels "
                    "were given"
                )
        else:
            labels = np.array([str(a).strip().lower() for a in anatomy])
            if len(labels) != n:
                raise ValueError(
                    "anatomy has %d entries but there are %d electrodes"
                    % (len(labels), n)
                )
            if policy.flag_non_cortical:
                patterns = [q.lower() for q in policy.non_cortical_patterns]
                non_cortical = np.array(
                    [any(q in label for q in patterns) for label in labels]
                )
                placement[non_cortical & ~too_far] = NON_CORTICAL
            if policy.drop_unknown_anatomy:
                unknown = np.isin(labels, [u.lower() for u in policy.unknown_labels])
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
    n_skipped: int = 0
    """Rows with a non-finite coordinate, which were left out of every figure
    above. Real montages carry these in quantity -- placeholder rows for
    unconnected amplifier channels -- and one of them is enough to turn an
    unguarded median into NaN, so they are excluded and counted rather than
    propagated."""
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
        if self.n_skipped:
            lines.append(
                "  %d row%s skipped for having no finite coordinate"
                % (self.n_skipped, "" if self.n_skipped == 1 else "s")
            )
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

    # Placeholder rows for unconnected amplifier channels carry NaN
    # coordinates. `anchor_to_surfaces` keeps them, correctly -- their row
    # indices still line up with data recorded on those channels -- but their
    # positions come back NaN, and a single one of those turns every statistic
    # below into NaN. A report that says nothing at all is worse than one that
    # says what it measured and how much it left out.
    finite = np.isfinite(coords).all(axis=1)
    if not finite.any():
        raise ValueError(
            "no electrode has a finite coordinate, so there is nothing to "
            "check alignment against"
        )
    measured = coords[finite]

    anchors = anchor_to_surfaces(
        measured, hemis,
        policy=PlacementPolicy(max_surface_distance_mm=np.inf),
    )
    residuals = measured - anchors.evaluate({h: hemis[h].pia for h in hemis})
    offsets = np.linalg.norm(residuals, axis=1)
    return AlignmentReport(
        median_offset_mm=float(np.median(offsets)),
        max_offset_mm=float(np.max(offsets)),
        systematic_shift_mm=residuals.mean(0),
        n_electrodes=int(finite.sum()),
        n_skipped=int((~finite).sum()),
        suspicious=bool(np.median(offsets) > threshold_mm),
    )
