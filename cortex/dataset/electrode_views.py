"""The three electrode views: data over a montage, in the shape of the grid.

:class:`Electrode`, :class:`Electrode2D` and :class:`ElectrodeRGB` are to an
:class:`~cortex.electrodes.ElectrodeSet` what :class:`~cortex.dataset.views.Vertex`
and its siblings are to a surface. Everything that makes them views --
colormapping, ``.raw``, ``uniques``, alpha, the arithmetic operators, HDF -- is
inherited from the three abstract columns; what is written here is the montage
argument and the two operations that relate one montage to another,
:meth:`Electrode.to_sub` and :meth:`Electrode.concat`.

Those two live on the views rather than on
:class:`~cortex.dataset._electrode_space.ElectrodeSpace` because they relate a
*pair* of spaces, which is the same reason ``Volume.map`` and ``Vertex.map``
are not on their spaces either.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union, cast

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
import numpy.typing as npt

from ._electrode_space import NATIVE, ElectrodeSpace, resolve_montage
from ._hdf import _hash
from .view2D import Dataview2D, _resolve_2d_channels
from .viewRGB import Color, Colors, DataviewRGB, _resolve_rgb_channels
from .views import ElectrodeView, ScalarView

if TYPE_CHECKING:
    from ..electrodes import ElectrodeSet


MARKER_SHAPES = ("sphere", "cube", "diamond")
"""The shapes a marker may take.

The single source of truth for the vocabulary, shared by the viewer's
``module.makeGeometry`` and by ``quickflat``'s ``MARKER_BY_SHAPE``. Adding a
shape means adding it in all three; the webgl tests walk this tuple and drive
the viewer with each name, which is what stops them drifting apart.
"""


class _ElectrodeMarkers:
    """Per-contact marker size and shape, carried on an electrode view.

    Both are stored in ``attrs`` and exposed as properties, the way
    :attr:`~cortex.dataset.views.Dataview.priority` is. That is not an
    implementation shortcut: ``attrs`` is the one part of the view record
    designed to be open, and it already round-trips through :meth:`copy`, HDF
    slot 6, and :meth:`~cortex.dataset.views.Dataview.to_json` into the browser.
    Storing them anywhere else would mean re-implementing all three.

    They override what the montage says. An :class:`~cortex.electrodes.ElectrodeSet`
    records each contact's physical diameter, which is the right default and is
    the same for every dataset plotted on that montage; a view's vector is for
    when the *data* should drive the marker -- bigger where the effect is
    stronger, a different shape for the contacts under discussion. Absent
    vectors leave the montage in charge.
    """

    #: Set by the concrete view classes; declared for mypy.
    attrs: dict

    @property
    def n_electrodes(self) -> int:  # pragma: no cover - provided by the view
        raise NotImplementedError

    @property
    def marker_size(self) -> Optional[npt.NDArray[np.floating]]:
        """Per-contact marker diameter in millimetres, or None.

        NaN at a position means "no override here" -- that contact falls back to
        the montage's own diameter, matching
        :attr:`~cortex.electrodes.ElectrodeSet.size`, where NaN likewise means
        absent.
        """
        stored = self.attrs.get("marker_size")
        if stored is None:
            return None
        return np.array([np.nan if v is None else v for v in stored], dtype=np.float64)

    @property
    def marker_shape(self) -> Optional[npt.NDArray[np.str_]]:
        """Per-contact marker shape, one of :data:`MARKER_SHAPES`, or None."""
        stored = self.attrs.get("marker_shape")
        return None if stored is None else np.array(stored, dtype=str)

    def _install_markers(
        self,
        marker_size: Optional[npt.ArrayLike],
        marker_shape: Optional[Union[str, Sequence[str], npt.NDArray]],
    ) -> None:
        """Validate and store, or leave whatever ``attrs`` already carried.

        The precedence here is the inverse of ``priority``'s ``setdefault``, and
        deliberately so: ``priority``'s default of 1 is indistinguishable from an
        explicit 1, so metadata carried in ``attrs`` has to win. ``None`` here
        means unambiguously "not given", which is what lets a view rebuilt from
        HDF keep the vectors its ``attrs`` arrived with.
        """
        if marker_size is not None:
            self.attrs["marker_size"] = self._check_size(marker_size)
        elif "marker_size" in self.attrs:
            self._recheck("marker_size", self._check_size)

        if marker_shape is not None:
            self.attrs["marker_shape"] = self._check_shape(marker_shape)
        elif "marker_shape" in self.attrs:
            self._recheck("marker_shape", self._check_shape)

    def _recheck(self, key: str, check) -> None:
        """Re-validate a key that arrived through ``attrs`` rather than a parameter.

        Warns and drops rather than raising. ``Dataset.from_file`` swallows
        per-view exceptions, so raising here would make the whole view disappear
        instead of the one bad key -- which is a worse trade when the file is
        someone else's and already written.
        """
        import warnings

        try:
            self.attrs[key] = check(self.attrs[key])
        except (TypeError, ValueError) as exc:
            warnings.warn("Dropping %s: %s" % (key, exc))
            del self.attrs[key]

    def _check_size(self, value: npt.ArrayLike) -> list:
        n = self.n_electrodes
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim > 1:
            raise ValueError(
                "marker_size must be one value per contact, got shape %r. Markers "
                "do not animate: a movie's frames all draw at the same size."
                % (arr.shape,)
            )
        if arr.shape != (n,):
            raise ValueError(
                "marker_size has %d entries but the montage has %d contacts"
                % (arr.size, n)
            )
        if np.any(arr[np.isfinite(arr)] < 0):
            raise ValueError("marker_size must not be negative")
        if np.isinf(arr).any():
            raise ValueError("marker_size must be finite, or NaN to fall back")
        # None rather than NaN: JSON has no NaN, and `_dumps` would emit a bare
        # NaN token that survives being inlined into the page and then dies the
        # moment anything calls JSON.parse.
        return [None if np.isnan(v) else float(v) for v in arr]

    def _check_shape(self, value: Union[str, Sequence[str], npt.NDArray]) -> list:
        n = self.n_electrodes
        # Checked before any sequence handling, so a one-contact montage is not
        # ambiguous between "a name" and "a sequence of one name".
        names = [value] * n if isinstance(value, str) else [str(v) for v in value]
        if len(names) != n:
            raise ValueError(
                "marker_shape has %d entries but the montage has %d contacts"
                % (len(names), n)
            )
        unknown = sorted({name for name in names if name not in MARKER_SHAPES})
        if unknown:
            raise ValueError(
                "unknown marker shape%s %s; expected one of %s"
                % ("s" if len(unknown) > 1 else "", ", ".join(repr(u) for u in unknown),
                   ", ".join(repr(s) for s in MARKER_SHAPES))
            )
        return names


class Electrode(ScalarView, ElectrodeView, _ElectrodeMarkers):
    """One value per intracranial contact, with a colormap.

    The electrode counterpart of :class:`~cortex.dataset.views.Volume` and
    :class:`~cortex.dataset.views.Vertex`, and used the same way::

        vol  = cortex.Volume(data, "S1", "fullhead")
        elec = cortex.Electrode(data, "S1", "native")

    Parameters
    ----------
    data : ndarray or None
        One value per contact, ``(n_contacts,)``, or ``(t, n_contacts)`` for a
        movie. ``None`` gives an all-zero array over the montage.
    subject : str
        Whose electrodes these are. Note that this is **not** necessarily what
        :attr:`subject` reports back: for a template montage the view's subject
        is the template, because that is whose surfaces get drawn. See
        ``montage`` below.
    montage : str
        Which localisation of this subject's contacts to use: ``"native"`` for
        the subject's own scan, or the name of the subject they were registered
        to -- ``"fsaverage"``, ``"cvs_avg35_inMNI152"``. Read from
        ``<filestore>/<subject>/electrodes/electrodes_<montage>.json``; see
        :meth:`cortex.database.Database.list_montages` for what a subject has.
    cmap : str, optional
        Colormap name. Defaults to the configured default.
    vmin, vmax : float, optional
        Colour bounds. Default to the 1st and 99th percentiles of the data, so
        the view always leaves construction with numeric bounds.
    description : str, optional
        Shown in the viewer's dataset list.
    state : optional
        Arbitrary state carried through to the browser.
    priority : int, optional
        Display order in the viewer.
    attrs : mapping, optional
        Metadata. The only route for a key that is not a parameter -- an
        unrecognised keyword is a ``TypeError``, not a silently stored attribute.
    montage_subjects : sequence of str, optional
        Whose electrodes these are, when that is more than one subject. Only
        :meth:`concat` has cause to pass it; it defaults to ``(subject,)``.
    electrodes : ElectrodeSet, optional
        Use this set instead of reading the filestore. For a montage that has
        not been saved, and for one combined across subjects.

    Notes
    -----
    ``subject`` reports whose *surfaces* these coordinates live in, which for a
    template montage is the template rather than the argument above::

        >>> cortex.Electrode(d, "S1", "fsaverage").subject
        'fsaverage'
        >>> cortex.Electrode(d, "S1", "fsaverage").montage_subjects
        ('S1',)

    That is not a quirk of this class but the only thing that can work: every
    renderer in pycortex reads ``dataview.subject`` to decide which brain to
    load, so reporting the implanted subject there would draw fsaverage
    coordinates on that patient's cortex.
    """

    def __init__(
        self,
        data: Union[npt.NDArray, str, None],
        subject: Union[str, bytes],
        montage: str = NATIVE,
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,
        montage_subjects: Optional[Sequence[str]] = None,
        electrodes: Optional["ElectrodeSet"] = None,
        marker_size: Optional[npt.ArrayLike] = None,
        marker_shape: Optional[Union[str, Sequence[str], npt.NDArray]] = None,
    ) -> None:
        owner = subject if isinstance(subject, str) else subject.decode("utf-8")
        surface, montage = resolve_montage(owner, montage)
        super().__init__(
            data,
            ElectrodeSpace(
                surface,
                montage=montage,
                montage_subjects=(owner,) if montage_subjects is None else montage_subjects,
                electrodes=electrodes,
            ),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            priority=priority,
            attrs=attrs,
        )

        # After super(), because validation needs the space to know how many
        # contacts the montage has.
        self._install_markers(marker_size, marker_shape)

    _space: ElectrodeSpace

    @property
    def space(self) -> ElectrodeSpace:
        return self._space

    @property
    def raw(self) -> ElectrodeRGB:
        return cast(ElectrodeRGB, self._build_raw())

    @property
    def name(self) -> str:
        """Content hash of the values **and the montage they are over**.

        :attr:`~cortex.dataset.views.ScalarView.name` hashes the array alone,
        which is right whenever a view's array determines what it means. Here it
        does not: the same numbers over S1's native montage and over S1's
        fsaverage montage are different data, sitting on different brains.

        This matters because ``name`` is the HDF node name, and a node carries
        the space's attributes -- so two spaces sharing a node is two montages
        sharing one set of montage attributes. ``_write_data_hdf`` skips a node
        whose data already hashes to its name, so the second view of a colliding
        pair would silently inherit the first one's montage and reload onto the
        wrong subject's cortex.

        A volumetric view has the same collision available to it and survives
        it, because HDF slot 7 of the *view* record repeats its transform name
        per view. An electrode space has no transform, so slot 7 is null and the
        data node is the only place the montage is written.
        """
        montage = "%s|%s|" % (self.montage, ",".join(self.montage_subjects))
        return "__%s" % _hash(
            np.frombuffer(
                montage.encode("utf-8") + np.ascontiguousarray(self.data).tobytes(),
                dtype=np.uint8,
            )
        )[:16]

    # -- the montage vocabulary -----------------------------------------

    @property
    def montage(self) -> str:
        """Which localisation this is: ``"native"`` or a template subject."""
        return self.space.montage

    @property
    def montage_subjects(self) -> tuple[str, ...]:
        """Whose electrodes these are. A tuple; longer than one after :meth:`concat`."""
        return self.space.montage_subjects

    @property
    def electrodes(self) -> "ElectrodeSet":
        """The contacts themselves: names, coordinates, metadata, anchors."""
        return self.space.electrodes

    @property
    def n_electrodes(self) -> int:
        return self.space.n_electrodes

    @property
    def names(self) -> npt.NDArray[np.str_]:
        """The contact names, in the order the data is indexed by."""
        return self.electrodes.names

    # -- relating two montages ------------------------------------------

    def to_sub(self, subject: str) -> "Electrode":
        """The same data, on ``subject``'s surfaces.

        Reads the montage the lab already localised into that subject's space;
        it does **not** transform the coordinates. Registering a montage to a
        template is an offline step with its own tooling, and inventing an
        answer here would put contacts somewhere plausible and wrong.

        Passing the implanted subject's own name gives back the native montage.

        Parameters
        ----------
        subject : str
            The subject whose anatomy to read the contacts in.

        Returns
        -------
        Electrode
            Over the same values, carrying this view's colormap, bounds and
            metadata, but a different montage -- and so a different
            :attr:`subject`.

        Examples
        --------
        >>> s1 = cortex.Electrode(data, "S1", "native")
        >>> fs = s1.to_sub("fsaverage")
        >>> fs.subject, fs.montage_subjects
        ('fsaverage', ('S1',))
        """
        return Electrode(
            self.data,
            self.montage_subjects[0],
            subject,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            description=self.description,
            state=self.state,
            priority=self.priority,
            attrs=self.attrs,
            montage_subjects=self.montage_subjects if len(self.montage_subjects) > 1 else None,
            # Passed explicitly rather than riding in `attrs`: `to_sub` reads a
            # *different* montage, which may have a different number of contacts.
            # The parameter path raises on a mismatch; the attrs path would warn
            # and silently drop the vectors, and someone moving a montage onto a
            # template wants to hear about it.
            marker_size=self.marker_size,
            marker_shape=self.marker_shape,
        )

    @classmethod
    def concat(
        cls,
        sub: str,
        data: Mapping[str, Union[npt.NDArray, "Electrode"]],
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        description: str = "",
        state: Any = None,
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,
        marker_size: Optional[npt.ArrayLike] = None,
        marker_shape: Optional[Union[str, Sequence[str], npt.NDArray]] = None,
    ) -> "Electrode":
        """Several subjects' electrodes on one common surface, as a single view.

        Parameters
        ----------
        sub : str
            The subject whose surfaces to draw on -- a template every
            contributing subject has a montage for.
        data : mapping
            Subject name to that subject's values, either a raw array or an
            :class:`Electrode` (whose ``data`` is taken). Iterated in the
            mapping's own order, which is the order the combined array is
            indexed in.
        cmap, vmin, vmax, description, state, priority, attrs
            As for the constructor. One colormap and one pair of bounds across
            every subject, which is the point of combining them.

        Returns
        -------
        Electrode
            Over the concatenated contacts, with ``montage_subjects`` naming
            every contributor.

        Notes
        -----
        Contact names are namespaced ``"S1/LSTG1"``, because
        :class:`~cortex.electrodes.ElectrodeSet` requires them to be unique and
        two subjects will both have an ``LSTG1``. The bare name survives as the
        contact's ``group``-mates would expect, and ``owner`` records the
        subject, so ``combined.electrodes.select(owner="S1")`` recovers one
        subject's contacts.

        Examples
        --------
        >>> both = cortex.Electrode.concat(
        ...     sub="fsaverage", data={"S1": s1_values, "S2": s2_values})
        """
        from ..database import db
        from ..electrodes import ElectrodeSet

        if not data:
            raise ValueError("concat needs at least one subject's data")
        if sub in data:
            raise ValueError(
                "%r is both the surface subject and a contributor; a subject's "
                "own montage of itself is 'native', which cannot be combined "
                "with anyone else's" % sub
            )

        sets, arrays = [], []
        for owner, values in data.items():
            eset = db.get_montage(owner, sub)
            array = np.asarray(values.data if isinstance(values, Electrode) else values)
            if array.shape[-1] != len(eset):
                raise ValueError(
                    "%s has %d values but its %r montage has %d contacts"
                    % (owner, array.shape[-1], sub, len(eset))
                )
            sets.append(eset)
            arrays.append(array)

        frames = {a.shape[0] for a in arrays if a.ndim > 1}
        if len(frames) > 1:
            raise ValueError(
                "movies of different lengths cannot be combined: %s"
                % ", ".join(str(f) for f in sorted(frames))
            )

        owners = tuple(data)
        combined = ElectrodeSet(
            names=[
                "%s/%s" % (o, n) for o, e in zip(owners, sets) for n in e.names
            ],
            coords=np.vstack([e.coords for e in sets]),
            subject=sub,
            # Groups are namespaced too, or two subjects' LSTG strips would read
            # as one device and draw with a single marker shape.
            group=[
                "%s/%s" % (o, g) for o, e in zip(owners, sets) for g in e.group
            ],
            size=[v for e in sets for v in e.size],
            anatomy=[v for e in sets for v in e.anatomy],
            status=[v for e in sets for v in e.status],
            group_type=[v for e in sets for v in e.group_type],
            owner=[v for e in sets for v in e.owner],
        )
        # Markers concatenate from the contributors the way the montage's own
        # `size` does above, unless the caller names them. All-or-nothing: a
        # partial set would silently mean "and the rest get montage defaults",
        # which is a quiet wrong answer of exactly the kind this feature exists
        # to prevent.
        views = [v for v in data.values() if isinstance(v, Electrode)]
        if marker_size is None:
            marker_size = _concat_markers(views, len(data), "marker_size")
        if marker_shape is None:
            marker_shape = _concat_markers(views, len(data), "marker_shape")

        return cls(
            np.concatenate(arrays, axis=-1),
            owners[0],
            sub,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            description=description,
            state=state,
            priority=priority,
            attrs=attrs,
            montage_subjects=owners,
            electrodes=combined,
            marker_size=marker_size,
            marker_shape=marker_shape,
        )

    # -- construction helpers -------------------------------------------

    @classmethod
    def empty(
        cls, subject: str, montage: str = NATIVE, value: float = 0, **kwargs: Any
    ) -> Self:
        """A view over this montage with every contact set to ``value``."""
        return cls(cls._sample(_shape_of(subject, montage), value),
                   subject, montage, **kwargs)

    @classmethod
    def random(cls, subject: str, montage: str = NATIVE, **kwargs: Any) -> Self:
        """A view over this montage filled with standard normal noise."""
        return cls(cls._sample(_shape_of(subject, montage), None),
                   subject, montage, **kwargs)

    def __repr__(self) -> str:
        return "<electrode data for (%s, %s of %s)>" % (
            self.subject, self.montage, ", ".join(self.montage_subjects),
        )


class Electrode2D(Dataview2D[Electrode], ElectrodeView, _ElectrodeMarkers):
    """Two electrode channels over one montage, jointly colormapped.

    The case this exists for is comparing two models over the same contacts --
    two encoding-model correlations, a response before and after -- where a 2D
    colormap says more than two figures side by side.

    Parameters
    ----------
    dim1, dim2 : ndarray or Electrode
        The two channels, over the same montage. Either both arrays or both
        views; mixing them is a ``TypeError``.
    subject : str, optional
        Whose electrodes. Required when the channels are raw arrays, since there
        is then nothing else to build the montage from.
    montage : str, optional
        Which localisation. Likewise required with raw arrays.
    description : str, optional
    cmap : str, optional
        A 2D colormap name.
    vmin, vmax : float, optional
        Bounds for ``dim1``.
    vmin2, vmax2 : float, optional
        Bounds for ``dim2``.
    alpha : ndarray, optional
        Overrides the alpha the 2D colormap produces.
    state, priority, attrs
        As for :class:`Electrode`.
    """

    def __init__(
        self,
        dim1: Union[npt.NDArray, Electrode],
        dim2: Union[npt.NDArray, Electrode],
        subject: Optional[str] = None,
        montage: Optional[str] = None,
        description: str = "",
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        vmin2: Optional[float] = None,
        vmax2: Optional[float] = None,
        alpha: Optional[npt.NDArray] = None,
        state: Any = None,
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,
        montage_subjects: Optional[Sequence[str]] = None,
        electrodes: Optional["ElectrodeSet"] = None,
        marker_size: Optional[npt.ArrayLike] = None,
        marker_shape: Optional[Union[str, Sequence[str], npt.NDArray]] = None,
    ) -> None:
        surface, spec = _montage_spec(subject, montage, montage_subjects)
        chan1, chan2 = _resolve_2d_channels(
            dim1,
            dim2,
            channel_cls=Electrode,
            space_cls=ElectrodeSpace,
            subject=surface,
            spec=spec,
            ranges=((vmin, vmax), (vmin2, vmax2)),
        )
        super().__init__(
            chan1,
            chan2,
            description=description,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vmin2=vmin2,
            vmax2=vmax2,
            alpha=alpha,
            state=state,
            priority=priority,
            attrs=attrs,
        )

        # After super(), because validation needs the space to know how many
        # contacts the montage has.
        self._install_markers(marker_size, marker_shape)

    @property
    def space(self) -> ElectrodeSpace:
        return self.dim1.space

    @property
    def raw(self) -> ElectrodeRGB:
        return cast(ElectrodeRGB, super().raw)

    @property
    def montage(self) -> str:
        return self.space.montage

    @property
    def montage_subjects(self) -> tuple[str, ...]:
        return self.space.montage_subjects

    @property
    def electrodes(self) -> "ElectrodeSet":
        return self.space.electrodes

    @property
    def n_electrodes(self) -> int:
        return self.space.n_electrodes

    def to_sub(self, subject: str) -> "Electrode2D":
        """The same two channels, on ``subject``'s surfaces. See :meth:`Electrode.to_sub`."""
        return Electrode2D(
            self.dim1.to_sub(subject),
            self.dim2.to_sub(subject),
            description=self.description,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            vmin2=self.vmin2,
            vmax2=self.vmax2,
            alpha=self._alpha,
            state=self.state,
            priority=self.priority,
            # Explicit for the same reason as Electrode.to_sub: a different
            # montage may be a different length, and that should raise.
            marker_size=self.marker_size,
            marker_shape=self.marker_shape,
            attrs=self.attrs,
        )

    def __repr__(self) -> str:
        return "<2D electrode data for (%s, %s)>" % (
            self.dim1.subject, self.dim1.montage,
        )


class ElectrodeRGB(DataviewRGB[Electrode], ElectrodeView, _ElectrodeMarkers):
    """Three electrode channels as red, green and blue, plus alpha.

    Parameters
    ----------
    channel1, channel2, channel3 : ndarray or Electrode
        The three colour channels over the same montage. Either all arrays or
        all views.
    subject : str, optional
        Whose electrodes. Required with raw arrays.
    montage : str, optional
        Which localisation. Required with raw arrays.
    alpha : ndarray or Electrode, optional
        Per-contact opacity in ``[0, 1]``. Defaults to fully opaque, with NaNs
        in any channel masked out.
    description : str, optional
    state : optional
    channel1color, channel2color, channel3color : Color, optional
        The colour each channel contributes. Default red, green and blue; other
        choices mix in HSV.
    max_color_value : float, optional
        Caps the value channel of the HSV mix.
    max_color_saturation : float, optional
        Caps the saturation channel.
    vmin, vmax : float or 3-tuple, optional
        Bounds for scaling the channels, shared or one per channel.
    autorange : {"individual", "shared"}, optional
        Whether unbounded channels are scaled separately or against a common
        range.
    priority, attrs
        As for :class:`Electrode`.
    """

    def __init__(
        self,
        channel1: Union[npt.NDArray, Electrode],
        channel2: Union[npt.NDArray, Electrode],
        channel3: Union[npt.NDArray, Electrode],
        subject: Optional[str] = None,
        montage: Optional[str] = None,
        alpha: Optional[Union[npt.NDArray, Electrode]] = None,
        description: str = "",
        state: Any = None,
        channel1color: Color[int] = Colors.Red,
        channel2color: Color[int] = Colors.Green,
        channel3color: Color[int] = Colors.Blue,
        max_color_value: Optional[float] = None,
        max_color_saturation: float = 1.0,
        vmin: Optional[Union[float, tuple[float, float, float]]] = None,
        vmax: Optional[Union[float, tuple[float, float, float]]] = None,
        autorange: Literal["shared", "individual"] = "individual",
        priority: int = 1,
        attrs: Optional[Mapping[str, Any]] = None,
        montage_subjects: Optional[Sequence[str]] = None,
        electrodes: Optional["ElectrodeSet"] = None,
        marker_size: Optional[npt.ArrayLike] = None,
        marker_shape: Optional[Union[str, Sequence[str], npt.NDArray]] = None,
    ) -> None:
        surface, spec = _montage_spec(subject, montage, montage_subjects)
        red, green, blue, resolved_alpha = _resolve_rgb_channels(
            (channel1, channel2, channel3),
            channel_cls=Electrode,
            space_cls=ElectrodeSpace,
            subject=surface,
            spec=spec,
            colors=(channel1color, channel2color, channel3color),
            max_color_value=max_color_value,
            max_color_saturation=max_color_saturation,
            vmin=vmin,
            vmax=vmax,
            autorange=autorange,
            alpha=alpha,
        )
        super().__init__(
            red,
            green,
            blue,
            alpha=resolved_alpha,
            subject=surface,
            description=description,
            state=state,
            priority=priority,
            attrs=attrs,
        )

        # After super(), because validation needs the space to know how many
        # contacts the montage has.
        self._install_markers(marker_size, marker_shape)

    @property
    def space(self) -> ElectrodeSpace:
        return self.red.space

    @property
    def montage(self) -> str:
        return self.space.montage

    @property
    def montage_subjects(self) -> tuple[str, ...]:
        return self.space.montage_subjects

    @property
    def electrodes(self) -> "ElectrodeSet":
        return self.space.electrodes

    @property
    def n_electrodes(self) -> int:
        return self.space.n_electrodes

    def to_sub(self, subject: str) -> "ElectrodeRGB":
        """The same three channels, on ``subject``'s surfaces. See :meth:`Electrode.to_sub`."""
        alpha = self._alpha
        return ElectrodeRGB(
            self.red.to_sub(subject),
            self.green.to_sub(subject),
            self.blue.to_sub(subject),
            # A raw array carries no montage, so it needs no moving -- and an
            # absent alpha must stay absent rather than become a synthesised
            # opaque channel, which is what reading `self.alpha` would give.
            alpha=(
                alpha.to_sub(subject)
                if isinstance(alpha, Electrode)
                else cast(Optional[npt.NDArray], alpha)
            ),
            description=self.description,
            state=self.state,
            priority=self.priority,
            # Explicit for the same reason as Electrode.to_sub: a different
            # montage may be a different length, and that should raise.
            marker_size=self.marker_size,
            marker_shape=self.marker_shape,
            attrs=self.attrs,
        )

    def __repr__(self) -> str:
        return "<RGB electrode data for (%s, %s)>" % (self.subject, self.montage)


def _concat_markers(views: Sequence["Electrode"], n_inputs: int, attr: str):
    """One marker vector from several contributing views, or None.

    Returns None when nobody carries one, the concatenation when everybody does,
    and raises in between -- a partial set would quietly mean "and the rest fall
    back to their montages", which reads as a deliberate choice and is not one.
    """
    if len(views) != n_inputs:
        return None                       # some inputs were plain arrays
    carried = [getattr(v, attr) for v in views]
    if all(c is None for c in carried):
        return None
    if any(c is None for c in carried):
        missing = [v.montage_subjects[0] for v, c in zip(views, carried) if c is None]
        raise ValueError(
            "%s is set on some views but not on %s; set it on all of them or "
            "none" % (attr, ", ".join(missing))
        )
    return np.concatenate(carried)


def _shape_of(subject: str, montage: str) -> tuple[int, ...]:
    """The shape a fresh array over this montage should have."""
    surface, montage = resolve_montage(subject, montage)
    return ElectrodeSpace(
        surface, montage=montage, montage_subjects=(subject,)
    ).template_shape


def _montage_spec(
    subject: Optional[str],
    montage: Optional[str],
    montage_subjects: Optional[Sequence[str]],
) -> tuple[Optional[str], dict[str, Any]]:
    """``(surface_subject, spec)`` for a composite view's montage arguments.

    Both composite constructors take the *owner* as ``subject``, matching
    :class:`Electrode`, but must hand
    :func:`~cortex.dataset.views._resolve_channels` the **surface** subject:
    that function compares what it is given against ``channel.subject``, which
    for a template montage is the template. Passing the owner would reject a
    perfectly good pair of channels with "subject in Electrode objects
    ('fsaverage') is different than specified subject ('S1')".
    """
    if subject is None or montage is None:
        # Not enough to build a space from. That is legal -- it is the ordinary
        # call when channels are passed as views -- and `_resolve_channels`
        # raises with the right message if it turns out they were arrays.
        return subject, {"montage": montage, "montage_subjects": montage_subjects}
    surface, montage = resolve_montage(subject, montage)
    owners = (subject,) if montage_subjects is None else tuple(montage_subjects)
    return surface, {"montage": montage, "montage_subjects": owners}
