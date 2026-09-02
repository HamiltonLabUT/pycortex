"""Handing an electrode set to the webgl viewer.

The viewer is sent anchors, not positions. A position would be wrong the moment
the surface moved, and the whole point of an anchor is that the browser can
re-derive a position for whatever inflation and depth the sliders are currently
at -- which it does with the same ``get_position`` the picker and the ROI labels
already use.

What crosses the wire is therefore small: three vertex indices, three weights, a
depth and some metadata per contact. A three-hundred-contact montage is a few
tens of kilobytes of JSON, so it rides inside ``viewopts`` rather than needing a
payload format of its own.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import numpy.typing as npt

from ._anchor import HEMIS
from ._connect import group_edges

if TYPE_CHECKING:
    from ._set import ElectrodeSet

#: pycortex names hemispheres ``lh``/``rh``; the viewer's ``posdata`` and pivots
#: are keyed ``left``/``right``. Translated here, once, rather than in JavaScript.
_VIEWER_HEMI = {"lh": "left", "rh": "right"}


def ctm_vertex_index(ctmfile: str) -> npt.NDArray[np.integer]:
    """Map a pycortex vertex index onto the CTM's vertex order.

    A subject's CTM does not store vertices in pycortex order. ``mg2``, the
    compression the viewer packs with, sorts them spatially, and ``brainctm``
    saves the resulting permutation beside the ``.ctm`` as an ``.npz``. Per-vertex
    *data* is permuted through it in python before being sent
    (:meth:`cortex.webgl.data.Package.reorder`), so an electrode's vertex indices
    have to make the same trip or they name different vertices at the far end.

    The array is merged over both hemispheres, in pycortex's left-then-right
    order, and the permutation keeps the two apart -- every left vertex lands in
    the left block -- so a per-hemisphere index can be offset in and back out
    again.

    Note this is only *half* the journey. Three.js applies a second, local
    shuffle when it loads the buffers, to keep each draw chunk inside a 16-bit
    index; that one is ``indexMap``, and the browser applies it. Getting only one
    of the two right leaves markers scattered over the hemisphere, on the surface
    and plausible-looking, which is how this was missed until a within-grid
    distance was actually measured.
    """
    npz = os.path.splitext(ctmfile)[0] + ".npz"
    with np.load(npz) as index:
        return np.asarray(index["inverse"]).copy()


def to_viewer_json(
    eset: "ElectrodeSet",
    ctm_index: Optional[npt.NDArray[np.integer]] = None,
    left_nverts: Optional[int] = None,
    placeable_only: bool = True,
    radius: float = 1.5,
    color: str = "#ffcc33",
    connections: bool = True,
    line_color: Optional[str] = None,
    offsets: bool = True,
) -> dict[str, Any]:
    """Serialise an electrode set for the browser.

    Parameters
    ----------
    eset : ElectrodeSet
        Must be anchored; :meth:`~cortex.electrodes.ElectrodeSet.anchor` first.
    ctm_index : (nverts,) array, optional
        The subject's pycortex-to-CTM vertex permutation, from
        :func:`ctm_vertex_index`. Vertex indices are passed through untouched
        without it, which is right only for a caller that has already converted
        them.
    left_nverts : int, optional
        Vertices in the left hemisphere, needed to index the merged
        ``ctm_index``. Looked up from the subject when not given.
    placeable_only : bool
        Send only the contacts with an honest surface position. On by default,
        because a contact the policy rejected has no position for the viewer to
        draw: an unplaced one would sit whereever its meaningless anchor pointed
        and look exactly like a real one.
    radius : float
        Fallback marker radius in millimetres, the same units as the surface.
        Each contact overrides it with half its own recorded diameter where the
        montage has one, so markers are drawn at the size the electrodes are.
    color : str
        A CSS colour for every marker. One colour is all this carries -- colour
        by data value belongs with the ``Electrode`` views and their colormap.
    connections : bool
        Join contacts that are neighbours on the same device with a line, so a
        montage reads as devices rather than as a cloud of dots. Which pairs are
        neighbours is decided by :func:`~cortex.electrodes._connect.group_edges`;
        this only says whether to work them out and send them.
    line_color : str, optional
        A CSS colour for those lines. The marker ``color`` by default, which
        keeps a device reading as one object.
    offsets : bool
        Draw each contact at its real distance off the surface rather than on
        it, by carrying its stored frame residual onto whatever the inflation
        slider is showing. On by default, and the viewer exposes a toggle.

        It matters most for depth electrodes and hardly at all for a subdural
        grid, whose contacts sit a fraction of a millimetre from the column they
        anchor to. Measured on S1, a shaft at a uniform 4 mm pitch *projected*
        onto the inflated surface has consecutive spacings from 1.7 to 40.5 mm;
        carried through one frame they are all equal.

    Returns
    -------
    dict
        Ready for ``json.dumps`` and the viewer's ``viewopts.electrodes``.
    """
    if eset.anchors is None:
        raise ValueError(
            "electrodes must be anchored before they can be sent to the viewer; "
            "call eset.anchor() first"
        )

    anchors = eset.anchors
    keep = anchors.placeable if placeable_only else np.ones(len(eset), dtype=bool)
    host = anchors.hosts

    # The viewer draws `coords` directly against the CTM, which is built from
    # the subject's stored surfaces -- so the coordinates have to be in *their*
    # space, not the montage's. On a subject imported with
    # `mris_convert --to-scanner` those differ by c_ras, and a contact sent
    # unconverted is drawn a few millimetres off, on the neighbouring gyrus.
    offset = np.zeros(3)
    if eset.subject is not None:
        from ._set import surface_space_offset

        offset = surface_space_offset(eset.subject)
    coords = eset.coords + offset

    if ctm_index is not None and left_nverts is None:
        from ..database import db

        if eset.subject is None:
            raise ValueError("left_nverts is needed when the set has no subject")
        left_nverts = len(db.get_surf(eset.subject, "pia", "lh")[0])

    contacts = []
    drawn = []
    for i in np.flatnonzero(keep):
        hemi = str(anchors.hemi[i])
        if hemi not in HEMIS:
            continue
        drawn.append(i)
        contacts.append({
            # Its position in the montage. The browser needs it because this
            # function *drops* contacts -- unplaceable ones, and any whose
            # hemisphere did not resolve -- while a view's per-contact arrays are
            # montage-length. Without it the browser can only count, and counting
            # is wrong the moment anything is dropped.
            "index": int(i),
            "name": str(eset.names[i]),
            "group": str(eset.group[i]),
            "group_type": str(eset.group_type[i]).lower(),
            "hemi": _VIEWER_HEMI[hemi],
            # The measured position in the surfaces' own space, sent alongside
            # the anchor. The viewer shows this one on the anatomical surface --
            # where a coordinate is still meaningful -- and crosses over to the
            # anchor as soon as the surface starts to deform.
            "coords": [float(x) for x in coords[i]],
            "verts": _ctm_verts(anchors.verts[i], hemi, ctm_index, left_nverts),
            "weights": [float(w) for w in anchors.weights[i]],
            # The residual, and the triangle it is measured against. Sent inline
            # rather than as a reference to another contact, because the frame's
            # host may itself have been dropped by `placeable_only` -- a shaft
            # whose best-anchored contact was excluded still has fourteen others
            # that need its frame.
            "frame": _frame(anchors, i),
            "frame_verts": _ctm_verts(
                anchors.verts[host[i]], hemi, ctm_index, left_nverts
            ),
            "frame_weights": [float(w) for w in anchors.weights[host[i]]],
            "frame_scale_mm": _finite(
                None if anchors.frame_scale_mm is None else anchors.frame_scale_mm[i]
            ),
            # Which contacts move as one rigid device. The browser needs the
            # grouping rather than a precomputed scale, because the surface it
            # is scaling against is a morph target that changes with the slider,
            # so the scale has to be recomputed on every mix event.
            "device": int(host[i]),
            "depth": _finite(anchors.depth[i]),
            # Millimetres as well as the normalised depth, because the viewer's
            # depth window is specified in millimetres and cortical thickness
            # varies from about 1 to 4.5 mm across the surface -- a window in
            # normalised units would mean something different in every gyrus.
            "depth_mm": _finite(anchors.depth_mm[i]),
            "thickness_mm": _finite(anchors.thickness_mm[i]),
            "anatomy": str(eset.anatomy[i]),
            "status": str(eset.status[i]),
            "size": _finite(eset.size[i]),
            "placement": str(anchors.placement[i]),
        })

    # Edges index into `contacts`, not into `eset`, and are worked out from the
    # contacts that survived `keep` rather than from all of them. A contact the
    # placement policy rejected is not drawn, so it has no position for a line
    # to reach -- and chaining straight past the gap keeps a shank with one
    # unplaceable contact a single unbroken probe.
    drawn = np.asarray(drawn, dtype=int)
    # The same surface-space coordinates the markers are drawn at. Which pairs
    # are neighbours is decided from a device's own contacts, so it is invariant
    # to the shift and either array gives the same edges -- but one function
    # holding two coordinate frames is how the next mistake gets made.
    edges = [] if not connections else group_edges(
        coords[drawn], eset.group[drawn], eset.group_type[drawn],
        names=eset.names[drawn],
    )

    return {
        "electrodes": contacts,
        "edges": [[int(a), int(b)] for a, b in edges],
        "connections": bool(connections),
        # How long a montage-indexed array should be, so the browser can notice
        # when it is handed one that does not match.
        "nelec": int(len(eset)),
        "radius": float(radius),
        "offsets": bool(offsets),
        "color": color,
        "line_color": color if line_color is None else line_color,
        "subject": eset.subject,
    }


def _ctm_verts(
    verts: npt.NDArray[np.integer],
    hemi: str,
    ctm_index: Optional[npt.NDArray[np.integer]],
    left_nverts: Optional[int],
) -> list[int]:
    """Three vertex indices in the CTM's order, ready for the browser."""
    if ctm_index is None:
        return [int(v) for v in verts]
    offset = 0 if hemi == "lh" else int(left_nverts or 0)
    return [int(ctm_index[int(v) + offset] - offset) for v in verts]


def _finite(value: Any) -> Optional[float]:
    """A float, or None where JSON cannot carry NaN."""
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else value


def _frame(anchors: "ElectrodeAnchors", i: int) -> Optional[list[float]]:
    """One contact's frame residual, or None if it has none or it is not finite.

    All-or-nothing per contact: a frame with one NaN component is not a partial
    answer, it is a degenerate anchor triangle, and half a displacement would
    put the marker somewhere specific and wrong.
    """
    if anchors.frame is None:
        return None
    row = anchors.frame[i]
    if not np.isfinite(row).all():
        return None
    return [float(x) for x in row]
