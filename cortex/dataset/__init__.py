"""Classes representing brain data -- volumetric, per-vertex, or per-electrode --
for visualization.

The nine public view classes form a 3x3 grid: three spaces (volumetric,
surface, electrode) crossed with three channel layouts (scalar + 1D colormap,
two channels + 2D colormap, three channels + alpha).

===========  ==================  ====================  ====================
space        scalar              2D                    RGB
===========  ==================  ====================  ====================
volumetric   :class:`Volume`     :class:`Volume2D`     :class:`VolumeRGB`
surface      :class:`Vertex`     :class:`Vertex2D`     :class:`VertexRGB`
electrode    :class:`Electrode`  :class:`Electrode2D`  :class:`ElectrodeRGB`
===========  ==================  ====================  ====================

The channel layout is the inheritance axis: :class:`Dataview` is the common
root, and :class:`ScalarView`, :class:`Dataview2D` and :class:`DataviewRGB` sit
under it. See ``INHERITANCE.md`` for the map and ``TYPING_ALTERNATIVES.md`` for
the restructuring options that were considered.
"""

from __future__ import annotations

# `views` first, and kept out of isort's reach, because the constraint is live:
# `views` closes its circular dependency on `viewRGB`/`view2D` with deferred
# imports at the very bottom of its own module, so if `view2D` is imported before
# `views` has finished, it pulls in `viewRGB`, which re-enters `views`, whose
# bottom then finds `viewRGB` half-built. Hoisting the `.view2D` line above this
# one fails with `ImportError: cannot import name 'Colors' from partially
# initialized module`, which is what that ordering buys.
# `test_submodule_can_be_imported_first` separately pins that each submodule still
# works as the process's entry point.
from .views import (  # isort: skip
    Dataview,
    DataviewJSON,
    HasSubject,
    Packable,
    RenderableView,
    ScalarView,
    ElectrodeView,
    SurfaceView,
    as_renderable,
    Vertex,
    Volume,
    VolumetricView,
    _from_hdf_data,
)
from ._space import (
    BrainSpace,
    SurfaceSpace,
    VolumeSpace,
    register_space,
    registered_spaces,
)
from ._electrode_space import ElectrodeSpace
from .electrode_views import Electrode, Electrode2D, ElectrodeRGB
from ._webgl import ElectrodeValues, MosaicTexture, VertexAttributes, WebGLPayload
from .braindata import BrainData, VertexData, VolumeData
from .dataset import Dataset, DatasetLike, normalize
from .view2D import Dataview2D, Vertex2D, Volume2D
from .viewRGB import Colors, DataviewRGB, VertexRGB, VolumeRGB

__all__ = [
    # the nine public view classes
    "Volume",
    "Vertex",
    "Electrode",
    "Volume2D",
    "Vertex2D",
    "Electrode2D",
    "VolumeRGB",
    "VertexRGB",
    "ElectrodeRGB",
    # containers and helpers
    "Dataset",
    "DatasetLike",
    "Colors",
    "normalize",
    "DataviewJSON",
    # abstract bases, by both their current and their historical names
    "Dataview",
    "Packable",
    "ScalarView",
    "Dataview2D",
    "DataviewRGB",
    "BrainData",
    "VolumeData",
    "VertexData",
    # spaces -- the open axis
    "BrainSpace",
    "VolumeSpace",
    "SurfaceSpace",
    "ElectrodeSpace",
    "register_space",
    "registered_spaces",
    # the webgl wire encodings a space can pack its arrays into
    "WebGLPayload",
    "MosaicTexture",
    "VertexAttributes",
    "ElectrodeValues",
    # the spatial axis of the grid, and helpers for narrowing it
    "VolumetricView",
    "SurfaceView",
    "ElectrodeView",
    "RenderableView",
    "HasSubject",
    "as_renderable",
]
