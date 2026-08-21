"""The electrode set: geometry and metadata, with no data values attached.

An :class:`ElectrodeSet` is to electrodes what the SVG overlay is to ROIs -- a
per-subject description of *where things are and what they are called*, holding
no measurements. Values live in the ``Electrode`` view classes, over an
electrode space, the way values over a surface live in ``Vertex``.

Keeping them apart is what lets a metadata-only workflow (draw the grid, label
it, check the anatomy) and a data workflow (colour every contact by its
encoding-model correlation) coexist without either one carrying the other's
baggage -- and it is where receptive fields and evoked responses can hang, since
those are per-channel auxiliary arrays rather than anything a colormap consumes.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from ._anchor import (
    HEMIS,
    ElectrodeAnchors,
    PlacementPolicy,
    SurfacePair,
    anchor_to_surfaces,
    classify_placement,
)

GROUP_TYPES = ("grid", "strip", "seeg", "depth")
"""The physical device an electrode belongs to. Free-form in practice, so this
is the vocabulary the defaults understand rather than a constraint."""

STATUSES = ("good", "bad")

MISSING = ""
"""How an absent optional string is stored. Absent numbers are NaN. Neither is
None, so every field stays a real array and indexing works uniformly."""

_TRAILING_NUMBER = re.compile(r"^(?P<group>.*?)[\s_.\-]*(?P<index>\d+)\s*$")


def infer_group(name: str) -> str:
    """The electrode group implied by a channel name.

    ``"LSTG1"``, ``"LSTG_2"`` and ``"LSTG-03"`` all give ``"LSTG"``: a shared
    prefix followed by a number means a shared physical device. A name with no
    trailing number is its own group.

    Naming conventions differ between labs, which is why an explicit ``group``
    column always wins over this and why :meth:`ElectrodeSet.groups` is worth
    printing once on import -- an unexpected grouping is much easier to see as a
    list of group names than as a misdrawn figure.
    """
    text = str(name).strip()
    match = _TRAILING_NUMBER.match(text)
    # An all-digits name has no prefix to be a group; it is its own group
    # rather than a member of the empty-named one.
    return match.group("group") or text if match else text


def infer_index(name: str) -> Optional[int]:
    """The contact number implied by a channel name, or None."""
    match = _TRAILING_NUMBER.match(str(name).strip())
    return int(match.group("index")) if match else None


class ElectrodeInfo(NamedTuple):
    """One electrode, as handed to a tooltip or a metadata panel."""

    name: str
    coords: npt.NDArray[np.floating]
    group: str
    group_type: str
    anatomy: str
    status: str
    size: float
    hemi: str = MISSING
    depth: float = float("nan")
    placement: str = MISSING
    #: Which subject this contact was implanted in. Only interesting once a set
    #: holds more than one, which is what a tooltip over a combined montage needs.
    owner: str = MISSING


def _str_array(
    values: Optional[Sequence[Any]], n: int, what: str
) -> npt.NDArray[np.str_]:
    if values is None:
        return np.array([MISSING] * n, dtype=str)
    out = np.array(["" if v is None else str(v) for v in values], dtype=str)
    if len(out) != n:
        raise ValueError("%s has %d entries but there are %d electrodes" % (what, len(out), n))
    return out


def _float_array(
    values: Optional[Sequence[Any]], n: int, what: str
) -> npt.NDArray[np.floating]:
    if values is None:
        return np.full(n, np.nan)
    out = np.asarray(values, dtype=np.float64).ravel()
    if len(out) != n:
        raise ValueError("%s has %d entries but there are %d electrodes" % (what, len(out), n))
    return out


class ElectrodeSet:
    """A subject's electrodes: names, coordinates and metadata.

    Parameters
    ----------
    names : sequence of str
        Channel names, e.g. ``["LSTG1", "LSTG2", ...]``. Required, and required
        to be unique -- everything downstream identifies an electrode by name.
    coords : (n, 3) array
        Electrode coordinates in TkRegRAS, the space FreeSurfer's surfaces are
        already in and therefore the space pycortex holds them in. Required.
    subject : str, optional
        pycortex subject these belong to. Needed to anchor them.
    group : sequence of str, optional
        Physical device each electrode belongs to. Inferred from the channel
        names when absent; see :func:`infer_group`.
    size : sequence of float, optional
        Contact diameter in millimetres.
    anatomy : sequence of str, optional
        Anatomical label, from whichever atlas the lab uses.
    status : sequence of str, optional
        ``"good"`` or ``"bad"``.
    group_type : sequence of str, optional
        ``"grid"``, ``"strip"``, ``"seeg"`` or ``"depth"``.
    stated_hemisphere : sequence of str, optional
        The hemisphere the *file* claimed, kept only so that
        :func:`~cortex.electrodes.check_hemispheres` can report where it
        disagrees with the geometry. Nothing reads it otherwise: which
        hemisphere an electrode is in is decided by which surface it is
        nearer to, and a stated value neither overrides that nor is
        overridden by it in silence.
    owner : sequence of str, optional
        Which subject each contact was implanted in. Distinct from
        :attr:`subject`, which says whose *surfaces* these coordinates live in
        and therefore what they anchor to: a set read from S1's
        ``electrodes_fsaverage.json`` has owner ``"S1"`` and subject
        ``"fsaverage"``. The two differ only for a template montage, and a set
        holding several owners only arises from
        :meth:`cortex.Electrode.concat`.

    Notes
    -----
    Every field is a parallel array of length ``n``; absent optional strings are
    ``""`` and absent numbers are NaN, so there is no ``None`` to guard against
    and slicing the set slices every field together.

    ``coords`` is the truth and is never modified. Anchors are derived from it
    and can be recomputed at any time -- which is what makes the surface
    projection lossless in the only sense available: nothing is thrown away,
    even though a projection itself cannot be inverted.
    """

    def __init__(
        self,
        names: Sequence[str],
        coords: npt.ArrayLike,
        subject: Optional[str] = None,
        group: Optional[Sequence[str]] = None,
        size: Optional[Sequence[float]] = None,
        anatomy: Optional[Sequence[str]] = None,
        status: Optional[Sequence[str]] = None,
        group_type: Optional[Sequence[str]] = None,
        stated_hemisphere: Optional[Sequence[str]] = None,
        owner: Optional[Sequence[str]] = None,
        anchors: Optional[ElectrodeAnchors] = None,
    ) -> None:
        self.names = np.array([str(n) for n in names], dtype=str)
        n = len(self.names)
        unique, counts = np.unique(self.names, return_counts=True)
        if len(unique) != n:
            raise ValueError(
                "electrode names must be unique; repeated: %s"
                % ", ".join(unique[counts > 1])
            )

        # An empty set is legal, and has to be: `select()` returning nothing is
        # an ordinary answer -- no electrode in that group, none at that depth --
        # and raising there would make every caller pre-check before filtering.
        self.coords = (
            np.zeros((0, 3))
            if n == 0
            else np.atleast_2d(np.asarray(coords, dtype=np.float64))
        )
        if self.coords.shape != (n, 3):
            raise ValueError(
                "coords must be (%d, 3), got %r" % (n, self.coords.shape)
            )

        self.subject = subject
        self.group = (
            np.array([infer_group(name) for name in self.names], dtype=str)
            if group is None
            else _str_array(group, n, "group")
        )
        self.size = _float_array(size, n, "size")
        self.anatomy = _str_array(anatomy, n, "anatomy")
        self.status = _str_array(status, n, "status")
        self.group_type = _str_array(group_type, n, "group_type")
        # Defaults to `subject` rather than MISSING, because for a native
        # montage -- which is every set P0 and P1 built -- the two really are
        # the same subject, and defaulting to blank would make `select(owner=)`
        # answer "none of them" for the common case.
        self.owner = _str_array(
            [subject or MISSING] * n if owner is None else owner, n, "owner"
        )
        self.stated_hemisphere = (
            None
            if stated_hemisphere is None
            else _str_array(
                [str(h).strip().lower()[:1] for h in stated_hemisphere],
                n, "stated_hemisphere",
            )
        )

        if anchors is not None and len(anchors) != n:
            raise ValueError(
                "anchors has %d entries but there are %d electrodes"
                % (len(anchors), n)
            )
        self.anchors = anchors

    # -- basics ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        subject = "" if self.subject is None else " for %s" % self.subject
        anchored = "" if self.anchors is None else ", anchored"
        return "<%d electrodes in %d groups%s%s>" % (
            len(self), len(self.groups), subject, anchored,
        )

    def __getitem__(
        self, index: Union[int, str, slice, Sequence[Any], npt.NDArray]
    ) -> Union[ElectrodeInfo, "ElectrodeSet"]:
        """An :class:`ElectrodeInfo` for one electrode, a subset for anything else.

        Integers and names select one electrode because that is what a tooltip
        or a click handler wants; slices, masks and name lists select a subset,
        because that is what a figure wants.
        """
        if isinstance(index, str):
            matches = np.flatnonzero(self.names == index)
            if not len(matches):
                raise KeyError("no electrode named %r" % index)
            index = int(matches[0])
        if isinstance(index, (int, np.integer)):
            i = int(index)
            return ElectrodeInfo(
                name=str(self.names[i]),
                coords=self.coords[i],
                group=str(self.group[i]),
                group_type=str(self.group_type[i]),
                anatomy=str(self.anatomy[i]),
                status=str(self.status[i]),
                size=float(self.size[i]),
                hemi=MISSING if self.anchors is None else str(self.anchors.hemi[i]),
                depth=float("nan") if self.anchors is None else float(self.anchors.depth[i]),
                placement=MISSING if self.anchors is None else str(self.anchors.placement[i]),
                owner=str(self.owner[i]),
            )

        idx: Any = index
        if isinstance(idx, (list, tuple, np.ndarray)):
            arr = np.asarray(idx)
            if arr.dtype.kind in "US":
                idx = np.array([int(np.flatnonzero(self.names == v)[0]) for v in arr])
        return ElectrodeSet(
            names=self.names[idx],
            coords=self.coords[idx],
            subject=self.subject,
            group=self.group[idx],
            size=self.size[idx],
            anatomy=self.anatomy[idx],
            status=self.status[idx],
            group_type=self.group_type[idx],
            stated_hemisphere=(
                None if self.stated_hemisphere is None else self.stated_hemisphere[idx]
            ),
            owner=self.owner[idx],
            anchors=None if self.anchors is None else self.anchors[idx],
        )

    def __eq__(self, other: Any) -> bool:
        """Whether two sets describe the same electrodes in the same space.

        Content equality, not identity, because two loads of one file must
        compare equal: :func:`~cortex.dataset.views._resolve_channels` validates
        a composite view's space by testing ``existing != value`` on whatever
        identifies it, and an identity comparison there would reject a perfectly
        good pair of channels.

        Deliberately a **plain bool** rather than an elementwise array. That same
        comparison is used in an ``if``, and a numpy array there raises "truth
        value of an array is ambiguous".

        Metadata is not compared: names, coordinates and the space they are in
        are what make two sets the same set. Anatomy labels and statuses are
        annotations on it, and a set that has since been anchored or relabelled
        is still the same electrodes.
        """
        if not isinstance(other, ElectrodeSet):
            return NotImplemented
        return bool(
            self.subject == other.subject
            and np.array_equal(self.names, other.names)
            and np.array_equal(self.coords, other.coords, equal_nan=True)
        )

    # Kept identity-based on purpose, rather than dropped or made to match
    # __eq__. A set is mutable -- `db.get_electrodes` reassigns `subject` and
    # `anchor()` reassigns `anchors` -- so a content hash would change under a
    # dict; and defining __eq__ without this would make the class unhashable,
    # which is a silent break for anything outside this repo that puts one in a
    # set. Nothing here hashes an ElectrodeSet.
    __hash__ = object.__hash__

    def __iter__(self) -> Iterator[ElectrodeInfo]:
        for i in range(len(self)):
            info = self[i]
            assert isinstance(info, ElectrodeInfo)
            yield info

    @property
    def groups(self) -> list[str]:
        """The distinct electrode groups, in first-appearance order."""
        if not len(self):
            return []
        _, first = np.unique(self.group, return_index=True)
        return [str(g) for g in self.group[np.sort(first)]]

    @property
    def depth(self) -> npt.NDArray[np.floating]:
        """Normalised pia-to-white-matter depth. All NaN until anchored."""
        if self.anchors is None:
            return np.full(len(self), np.nan)
        return self.anchors.depth

    # -- selection ------------------------------------------------------

    def select(
        self,
        where: Optional[npt.NDArray[np.bool_]] = None,
        placeable: bool = False,
        **criteria: Union[str, Sequence[str]],
    ) -> "ElectrodeSet":
        """A subset of the electrodes.

        Parameters
        ----------
        where : (n,) bool array, optional
            An explicit mask, combined with everything else by ``and``.
        placeable : bool
            Keep only electrodes the placement policy accepted. Requires
            anchors.
        **criteria
            Field name to accepted value or values, e.g.
            ``select(group="LSTG")`` or
            ``select(group_type=["grid", "strip"], status="good")``. Any string
            field works, including ``hemi`` and ``placement`` once anchored.

        Returns
        -------
        ElectrodeSet
            A new set; the original is untouched.
        """
        mask = np.ones(len(self), dtype=bool) if where is None else np.asarray(where, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError("where must be a (%d,) mask" % len(self))

        for field, wanted in criteria.items():
            values = self._field(field)
            allowed = [wanted] if isinstance(wanted, str) else list(wanted)
            mask &= np.isin(values, allowed)

        if placeable:
            if self.anchors is None:
                raise ValueError("select(placeable=True) needs anchored electrodes")
            mask &= self.anchors.placeable

        return self[mask]  # type: ignore[return-value]

    def _field(self, name: str) -> npt.NDArray:
        if hasattr(self, name) and name not in (
            "anchors", "coords", "subject", "stated_hemisphere",
        ):
            return getattr(self, name)
        if self.anchors is not None and hasattr(self.anchors, name):
            return getattr(self.anchors, name)
        raise KeyError(
            "no field %r on this electrode set%s"
            % (name, "" if self.anchors is not None else " (it is not anchored)")
        )

    # -- anchoring ------------------------------------------------------

    def anchor(
        self,
        subject: Optional[str] = None,
        policy: Optional[PlacementPolicy] = None,
        surfaces: Optional[Mapping[str, SurfacePair]] = None,
        inplace: bool = True,
    ) -> ElectrodeAnchors:
        """Compute this set's surface anchors, loading the surfaces if needed.

        Parameters
        ----------
        subject : str, optional
            Overrides ``self.subject`` for this call.
        policy : PlacementPolicy, optional
        surfaces : mapping, optional
            Pre-loaded ``{"lh": SurfacePair, "rh": SurfacePair}``, which skips
            the database entirely -- useful for testing and for surfaces
            pycortex does not hold.
        inplace : bool
            Store the result on the set as well as returning it.

        Returns
        -------
        ElectrodeAnchors
        """
        if surfaces is None:
            subject = subject or self.subject
            if subject is None:
                raise ValueError(
                    "anchoring needs a subject, either on the set or as an argument"
                )
            surfaces = load_surface_pairs(subject)
        anchors = anchor_to_surfaces(
            self.coords, surfaces, policy=policy, anatomy=self.anatomy
        )
        if inplace:
            self.anchors = anchors
        return anchors

    def reclassify(self, policy: PlacementPolicy) -> npt.NDArray[np.str_]:
        """Re-apply a placement policy without redoing the geometry."""
        if self.anchors is None:
            raise ValueError("reclassify() needs anchored electrodes")
        self.anchors.placement = classify_placement(self.anchors, policy, self.anatomy)
        return self.anchors.placement

    def positions(self, surface_type: str = "flat", subject: Optional[str] = None,
                  nudge: bool = True) -> npt.NDArray[np.floating]:
        """Where these electrodes sit on some surface of the subject.

        Parameters
        ----------
        surface_type : str
            Any surface pycortex holds: ``"fiducial"``, ``"inflated"``,
            ``"flat"``, ``"pia"``, ``"wm"``.
        subject : str, optional
        nudge : bool
            Pass through to :meth:`cortex.database.Database.get_surf`, which
            shifts the hemispheres apart for every surface except the fiducial.
            On for flatmaps, where the two hemispheres are laid out side by
            side. Anchoring itself always happens un-nudged.

        Returns
        -------
        (n, 3) array
        """
        from ..database import db

        if self.anchors is None:
            raise ValueError("positions() needs anchored electrodes; call anchor() first")
        subject = subject or self.subject
        if subject is None:
            raise ValueError("positions() needs a subject")

        left, right = db.get_surf(subject, surface_type, "both", nudge=nudge)
        return self.anchors.evaluate({"lh": left[0], "rh": right[0]})


def load_surface_pairs(subject: str) -> dict[str, SurfacePair]:
    """The pial and white-matter surfaces a subject's anchors are built on.

    Loaded un-nudged and in their own coordinates: ``get_surf(..., nudge=True)``
    shifts non-fiducial hemispheres apart in x for display, which would silently
    move every anchor. Nudging is a rendering concern and belongs nowhere near
    the geometry.

    Falls back to the fiducial surface alone when a subject has no separate pial
    and white-matter surfaces. Anchors still work; depth comes back NaN, which
    matches the viewer -- ``brainctm`` omits the ``wm`` attribute for such a
    subject and its depth slider does nothing.
    """
    from ..database import db

    pairs: dict[str, SurfacePair] = {}
    try:
        pia = dict(zip(HEMIS, db.get_surf(subject, "pia", "both", nudge=False)))
        wm = dict(zip(HEMIS, db.get_surf(subject, "wm", "both", nudge=False)))
    except (IOError, KeyError):
        fiducial = dict(zip(HEMIS, db.get_surf(subject, "fiducial", "both", nudge=False)))
        return {h: SurfacePair(pia=fiducial[h][0], polys=fiducial[h][1]) for h in HEMIS}

    for hemi in HEMIS:
        pairs[hemi] = SurfacePair(pia=pia[hemi][0], polys=pia[hemi][1], wm=wm[hemi][0])
    return pairs
