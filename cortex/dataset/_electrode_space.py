"""The electrode space: data indexed by intracranial contact.

The third :class:`~cortex.dataset._space.BrainSpace`, and the first added since
the package was restructured around that being the open axis. What it adds to
the two built-ins is a distinction neither of them has to make.

A volume or a surface has exactly one subject: whose scan, whose cortex. An
electrode has *two*. The contacts were implanted in one person, but their
coordinates were written down in some anatomy, and a lab writes them down more
than once -- in the patient's own scan, and again after registering to a
template like ``fsaverage``. Both files describe the same physical electrodes;
they are not the same numbers, and they are not read against the same surfaces.

So this space carries:

``subject``
    Whose surfaces the coordinates live in, and therefore what a renderer draws.
    This has to be what :attr:`~cortex.dataset.views.Dataview.subject` reports,
    because that is the only question every renderer asks of a view --
    ``get_flatmask(braindata.subject)``, ``db.get_overlay(braindata.subject)``,
    and the line in ``webgl/data.py`` deciding which CTMs to pack. Report the
    implanted subject there and the viewer loads the wrong brain.

``montage_subjects``
    Whose electrodes they are. A tuple, always -- length one in the ordinary
    case, longer for a montage combined across subjects by
    :meth:`cortex.Electrode.concat`.

``montage``
    Which localisation this is: ``"native"``, or the name of the subject whose
    anatomy they were registered to. The rule connecting it to ``subject`` is
    one line, and lives in :meth:`cortex.database.Database.surface_subject`:
    the montage *is* the surface subject, except ``"native"``, which means the
    owner.

Nothing here transforms coordinates. A montage is read from the file the lab
already produced; pycortex does not perform the registration and will not
pretend to.
"""

from __future__ import annotations

import json
import sys
import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import h5py
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..electrodes import ElectrodeSet
    from .views import DataviewJSON, ScalarView

from ..database import db
from ._space import BrainSpace, MaskSpec, SpaceViews, register_space
from ._webgl import ElectrodeValues, WebGLPayload

NATIVE = "native"
"""The montage that means "the subject's own scan"."""

#: Above this many bytes, the serialised set is left out of the HDF file and the
#: montage is re-read from the filestore on load. h5py writes an attribute into
#: the object header, which the default file format caps at 64 KB -- and a large
#: clinical montage with anchors lands close enough to that to matter.
_MAX_EMBEDDED_SET_BYTES = 60_000


def resolve_montage(
    subject: str, montage: str = NATIVE
) -> tuple[str, str]:
    """``(surface_subject, montage)`` for an owner and a montage name.

    Accepts the owner's own name as a montage, so ``to_sub(own_subject)`` is the
    native montage rather than a missing file.
    """
    montage = NATIVE if montage == subject else montage
    return db.surface_subject(subject, montage), montage


@register_space
class ElectrodeSpace(BrainSpace):
    """One value per intracranial contact, over a named montage.

    Parameters
    ----------
    subject : str
        The **surface** subject: whose anatomy these coordinates are in. Note
        that :class:`cortex.Electrode`'s corresponding argument is the *owner*;
        the view translates. That asymmetry is deliberate -- a user says "S1's
        electrodes, in fsaverage space" and a renderer asks "which brain do I
        load", and those are different subjects.
    montage : str
        ``"native"``, or the name of the subject the contacts were registered to.
    montage_subjects : sequence of str, optional
        Whose electrodes these are. Defaults to ``(subject,)``, which is right
        for a native montage and is the only case where guessing is safe.
    electrodes : ElectrodeSet, optional
        A pre-built set, used instead of reading the filestore. The escape hatch
        for a set that has not been saved, for a montage combined across
        subjects, and for tests -- which is also why it is not one of
        :attr:`spec_keys`: what identifies this space is the montage, and the
        set is what the montage resolves to.
    """

    hdf_key = "electrodes"

    #: What :meth:`~cortex.dataset._space.BrainSpace.from_spec` requires before it
    #: will build this space from raw arrays. Both are needed: the montage says
    #: which localisation, and the owners say whose -- and for a template montage
    #: neither is recoverable from ``subject``, which names the template.
    spec_keys = ("montage", "montage_subjects")

    def __init__(
        self,
        subject: Union[str, bytes],
        montage: str = NATIVE,
        montage_subjects: Optional[Sequence[str]] = None,
        electrodes: Optional["ElectrodeSet"] = None,
    ) -> None:
        super().__init__(subject)
        self.montage = str(montage)
        self.montage_subjects: tuple[str, ...] = (
            (self.subject,)
            if montage_subjects is None
            else tuple(str(s) for s in montage_subjects)
        )
        self._electrodes = electrodes

    @property
    def electrodes(self) -> "ElectrodeSet":
        """The contacts this space is over, read from the filestore if need be.

        Loaded lazily and cached, so building a space costs nothing until
        something asks how many contacts there are -- which
        :meth:`coerce` does immediately for any view holding real data, but not
        for a space built only to be compared against.
        """
        if self._electrodes is None:
            self._electrodes = self._load()
        return self._electrodes

    def _load(self) -> "ElectrodeSet":
        if len(self.montage_subjects) != 1:
            raise ValueError(
                "a montage combined across %d subjects (%s) has no single file to "
                "read; it exists only as the set it was built from, so it must be "
                "passed in as electrodes="
                % (len(self.montage_subjects), ", ".join(self.montage_subjects))
            )
        return db.get_montage(self.montage_subjects[0], self.montage)

    @property
    def n_electrodes(self) -> int:
        return len(self.electrodes)

    @property
    def xfmname(self) -> None:
        """An electrode space has no transform.

        Contacts are positioned by a barycentric anchor onto three surface
        vertices, not by sampling a volume through a matrix, so there is nothing
        for a :class:`~cortex.xfm.Transform` to name. Returning ``None`` puts
        this space in the same bucket as a surface for HDF slot 7, which is why
        :meth:`from_hdf` has to discriminate on something it writes itself.
        """
        return None

    # ------------------------------------------------------------------
    # binding an array
    # ------------------------------------------------------------------
    def coerce(self, data: Optional[npt.NDArray]) -> npt.NDArray:
        """Check that ``data`` has one value per contact.

        No padding, unlike :class:`~cortex.dataset._space.SurfaceSpace`, which
        accepts a single hemisphere's worth of vertices and fills in the other.
        There is no comparable convention here: a montage is a flat list of
        contacts in one order, and an array of the wrong length is a mismatch
        between the data and the montage rather than a partial answer.
        """
        n = self.n_electrodes
        if data is None:
            return np.zeros((n,))
        given = data.shape[-1]
        if given != n:
            raise ValueError(
                "data has %d values but the %r montage for %s has %d contacts"
                % (given, self.montage, ", ".join(self.montage_subjects), n)
            )
        return data

    def is_movie(self, data: npt.NDArray) -> bool:
        # (t, n) versus (n,)
        return data.ndim > 1

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return (self.n_electrodes,)

    def align(
        self, first: "ScalarView", second: "ScalarView"
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The two arrays, once it is established they mean the same contacts.

        Overridden because the inherited default hands both back unconditionally,
        and here that is not safe. Two electrode views agree on ``subject`` --
        which is all
        :func:`~cortex.dataset.views._resolve_channels` checks -- as soon as they
        are drawn on the same template, even when one holds S1's contacts and the
        other S2's. Equal lengths would then let a 2D view colormap one subject's
        values against another's.
        """
        other = second.space
        if not isinstance(other, ElectrodeSpace):
            raise TypeError(
                "cannot align electrode data against %s" % type(other).__name__
            )
        if (other.montage, other.montage_subjects) != (
            self.montage, self.montage_subjects,
        ):
            raise ValueError(
                "these views are over different montages (%s of %s, and %s of %s), "
                "so position i is not the same contact in both"
                % (
                    self.montage, ", ".join(self.montage_subjects),
                    other.montage, ", ".join(other.montage_subjects),
                )
            )
        return first.data, second.data

    def wrap(self, data: npt.NDArray, **kwargs: Any) -> Any:
        from .electrode_views import Electrode

        # The set is handed over rather than left to be re-read: `wrap` is what
        # `.raw` and `copy()` go through, so a lookup here would hit the
        # filestore once per arithmetic operation -- and for a combined montage
        # there is no file to hit.
        return Electrode(
            data, self.montage_subjects[0], self.montage,
            montage_subjects=self.montage_subjects, electrodes=self.electrodes,
            **kwargs,
        )

    def wrap_rgb(
        self,
        red: npt.NDArray,
        green: npt.NDArray,
        blue: npt.NDArray,
        alpha: Optional[npt.NDArray] = None,
        **kwargs: Any,
    ) -> Any:
        from .electrode_views import ElectrodeRGB

        return ElectrodeRGB(
            red, green, blue, self.montage_subjects[0], self.montage, alpha,
            montage_subjects=self.montage_subjects, electrodes=self.electrodes,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_json(self) -> "DataviewJSON":
        from .views import DataviewJSON as _JSON

        return _JSON()

    def describe_layout(self, data: npt.NDArray) -> "DataviewJSON":
        """How many contacts the browser should expect, and how many frames.

        ``nelec`` is also the key ``dataset.js`` dispatches on to choose the
        electrode payload class, so it has to be present on every electrode
        record and absent from every other -- the JS tests for ``undefined``.
        """
        from .views import DataviewJSON as _JSON

        frames = data.shape[0] if self.is_movie(data) else 1
        return _JSON(nelec=self.n_electrodes, frames=frames)

    def pack_for_webgl(self, data: npt.NDArray, *, raw: bool) -> WebGLPayload:
        """One value per contact per frame, shipped as a plain ``.npy``.

        Neither of the two older encodings fits: a mosaic tiles a 3-D grid, and
        the per-vertex one permutes into the CTM's vertex order and premultiplies
        alpha. Both of those would corrupt an array indexed by contact.
        """
        return ElectrodeValues(data, raw=raw)

    def write_hdf_attrs(
        self, h5: Union[h5py.File, h5py.Group], node: h5py.Dataset
    ) -> None:
        """Record the montage, and the contacts themselves where they will fit.

        The set is embedded rather than merely named because
        :meth:`from_hdf` is handed the node's attributes and nothing else -- not
        the file -- so it cannot go and read a group the way ``Dataset`` reads
        the surfaces and ROIs it packs. Embedding is therefore the only way a
        saved view reloads on a machine whose filestore lacks the montage.

        h5py writes an attribute into the object header, which the default file
        format caps at 64 KB, and a few hundred contacts with anchors reach that.
        Past the cap the montage name is written alone and the filestore becomes
        the source on load, which is a weaker guarantee stated out loud rather
        than an error at save time -- the data is what the user asked to keep.
        """
        node.attrs["montage"] = self.montage
        node.attrs["montage_subjects"] = list(self.montage_subjects)

        from ..electrodes import to_dict

        blob = json.dumps(to_dict(self.electrodes))
        if len(blob.encode("utf-8")) > _MAX_EMBEDDED_SET_BYTES:
            warnings.warn(
                "the %r montage for %s is too large to store in this file (%d "
                "contacts); it will be re-read from the filestore when this view "
                "is loaded, so the file is not self-contained"
                % (self.montage, ", ".join(self.montage_subjects), self.n_electrodes)
            )
            return
        node.attrs["electrode_set"] = blob

    @classmethod
    def from_hdf(
        cls,
        attrs: dict[str, Any],
        *,
        subject: str,
        xfmname: Optional[str],
        mask: MaskSpec,
    ) -> Optional["Self"]:
        """Claim a node that names a montage.

        ``montage`` is the discriminator rather than ``hdf_key``: it is written
        by :meth:`write_hdf_attrs` in both the embedded and the too-large case,
        and no other space writes it. Testing it here is what lets this space be
        consulted ahead of :class:`~cortex.dataset._space.SurfaceSpace`, which
        would otherwise claim every node without a transform -- including these.
        """
        if "montage" not in attrs:
            return None

        montage = _as_str(attrs["montage"])
        owners = tuple(_as_str(s) for s in attrs.get("montage_subjects", [subject]))

        electrodes = None
        if "electrode_set" in attrs:
            from ..electrodes import from_dict

            electrodes = from_dict(json.loads(_as_str(attrs["electrode_set"])))

        return cls(
            subject, montage=montage, montage_subjects=owners, electrodes=electrodes
        )

    @classmethod
    def views(cls) -> SpaceViews:
        from .electrode_views import Electrode, Electrode2D, ElectrodeRGB

        return SpaceViews(scalar=Electrode, twod=Electrode2D, rgb=ElectrodeRGB)

    def __repr__(self) -> str:
        return "<ElectrodeSpace(%s, %s of %s)>" % (
            self.subject, self.montage, ", ".join(self.montage_subjects),
        )


def _as_str(value: Any) -> str:
    """One HDF attribute as text.

    h5py hands back ``bytes`` for a string written by an older library and
    ``np.str_`` for one written by this one, and neither is a ``str`` to
    ``json.loads`` or to a dict key.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
