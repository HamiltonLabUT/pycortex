"""Intracranial electrodes on a pycortex surface.

The unit here is an *electrode* -- one recording contact, with a name, a
position in TkRegRAS, and metadata -- rather than a voxel or a vertex. An
electrode has no vertex identity of its own, so keeping it in the right place
while the surface inflates and flattens means parameterising it against the
surface rather than storing a position: see :mod:`cortex.electrodes._anchor`.

Two objects, deliberately separate:

- :class:`ElectrodeSet` -- geometry and metadata for a subject's electrodes.
  No data values. This is what a figure or the viewer draws.
- ``Electrode`` and friends (added in a later phase) -- data *over* an electrode
  set, colormapped and serialisable, the way :class:`cortex.Vertex` is data over
  a surface.

Typical use::

    import cortex.electrodes as ce

    eset = ce.load_electrodes("sub-01_electrodes.tsv", subject="S1")
    print(eset.groups)                      # check the inferred grouping
    print(ce.check_alignment(eset.coords, ce.load_surface_pairs("S1")).summary())

    anchors = eset.anchor()                 # geometry, once
    print(anchors.summary())                # what the placement policy did

    grids = eset.select(group_type=["grid", "strip"], placeable=True)
    xyz = grids.positions("flat")           # follows the surface, flat or not

A montage of any size reads better as a field than as a field of dots, and
:mod:`cortex.electrodes._blobs` turns one into the other: each contact's value
spread over the cortex within a few millimetres of it, measured along the
surface rather than through it. The result is an ordinary
:class:`cortex.Vertex`, so everything that already draws one draws this::

    elec = cortex.Electrode(values, "S1", "native")
    cortex.quickshow(elec.to_vertex(sigma=3))       # what the contacts say
    cortex.quickshow(elec.coverage(sigma=3))        # where they can say it
"""

from __future__ import annotations

from ._anchor import (
    NON_CORTICAL,
    NO_COORDINATE,
    ON_SURFACE,
    PLACEMENTS,
    PROJECTED,
    TOO_FAR,
    UNKNOWN_ANATOMY,
    AlignmentReport,
    ElectrodeAnchors,
    PlacementPolicy,
    SurfacePair,
    anchor_to_surfaces,
    check_alignment,
    classify_placement,
    surface_hash,
)
from ._io import (
    check_hemispheres,
    from_dict,
    load_electrodes,
    load_electrodes_json,
    read_elecs_mat,
    read_electrodes_tsv,
    save_electrodes_json,
    to_dict,
    write_electrodes_tsv,
)
from ._blobs import (
    DEFAULT_SIGMA,
    RADIUS_SIGMAS,
    surface_weights,
    total_weight,
    volume_weights,
    weighted_mean,
)
from ._connect import GRID_SLACK, LINEAR_TYPES, group_edges
from ._webgl import to_viewer_json
from ._set import (
    GROUP_TYPES,
    STATUSES,
    ElectrodeInfo,
    ElectrodeSet,
    infer_group,
    infer_index,
    load_surface_pairs,
    surface_space_offset,
)

__all__ = [
    # the set
    "ElectrodeSet",
    "ElectrodeInfo",
    "GROUP_TYPES",
    "STATUSES",
    "infer_group",
    "infer_index",
    # anchoring
    "ElectrodeAnchors",
    "PlacementPolicy",
    "SurfacePair",
    "anchor_to_surfaces",
    "classify_placement",
    "load_surface_pairs",
    "surface_space_offset",
    "surface_hash",
    "ON_SURFACE",
    "PROJECTED",
    "NON_CORTICAL",
    "NO_COORDINATE",
    "TOO_FAR",
    "UNKNOWN_ANATOMY",
    "PLACEMENTS",
    # spreading a contact's value over the cortex around it
    "surface_weights",
    "volume_weights",
    "weighted_mean",
    "total_weight",
    "DEFAULT_SIGMA",
    "RADIUS_SIGMAS",
    # sanity checks
    "AlignmentReport",
    "check_alignment",
    "check_hemispheres",
    # which contacts are neighbours on their device
    "group_edges",
    "LINEAR_TYPES",
    "GRID_SLACK",
    # io
    "to_viewer_json",
    "load_electrodes",
    "read_electrodes_tsv",
    "read_elecs_mat",
    "write_electrodes_tsv",
    "load_electrodes_json",
    "save_electrodes_json",
    "to_dict",
    "from_dict",
]
