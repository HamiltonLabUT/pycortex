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

    if ctm_index is not None and left_nverts is None:
        from ..database import db

        if eset.subject is None:
            raise ValueError("left_nverts is needed when the set has no subject")
        left_nverts = len(db.get_surf(eset.subject, "pia", "lh")[0])

    contacts = []
    for i in np.flatnonzero(keep):
        hemi = str(anchors.hemi[i])
        if hemi not in HEMIS:
            continue
        contacts.append({
            "name": str(eset.names[i]),
            "group": str(eset.group[i]),
            "group_type": str(eset.group_type[i]).lower(),
            "hemi": _VIEWER_HEMI[hemi],
            # The measured TkRegRAS position, sent alongside the anchor. The
            # viewer shows this one on the anatomical surface -- where a
            # coordinate is still meaningful -- and crosses over to the anchor
            # as soon as the surface starts to deform.
            "coords": [float(x) for x in eset.coords[i]],
            "verts": _ctm_verts(anchors.verts[i], hemi, ctm_index, left_nverts),
            "weights": [float(w) for w in anchors.weights[i]],
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

    return {
        "electrodes": contacts,
        "radius": float(radius),
        "color": color,
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
    value = float(value)
    return None if not np.isfinite(value) else value
