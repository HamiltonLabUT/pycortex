"""Reading and writing electrode sets.

Two formats, for two jobs.

**BIDS ``*_electrodes.tsv``** is the interchange format. Its columns --
``name``, ``x``, ``y``, ``z``, ``size``, ``group``, ``hemisphere``, ``type``,
``status`` -- map almost one to one onto the fields an
:class:`~cortex.electrodes.ElectrodeSet` holds, so files another lab already has
load without a bespoke parser, and files pycortex writes are readable by
everything else in the iEEG ecosystem. Missing values are ``n/a``, per BIDS.

**MATLAB ``*_elecs_all.mat``** is what ``img_pipe`` and the labs downstream of
it write -- ``TDT_elecs_all.mat``, ``clinical_elecs_all.mat`` -- holding
``elecmatrix`` (n by 3, surface RAS), ``eleclabels`` (short name, long name,
device type) and usually ``anatomy`` (those three plus an anatomical label).

**JSON** is the filestore format, at
``<filestore>/<subject>/electrodes/<name>.json``. It carries the same fields
plus the derived anchors and the hash of the surfaces they were computed
against, so a stale cache is recognisable rather than merely wrong.

Both are read with the standard library. Electrode tables are hundreds of rows,
not millions, and a pandas dependency for that would be a poor trade.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

from ._anchor import ElectrodeAnchors
from ._set import MISSING, ElectrodeSet

NA = "n/a"
"""BIDS' missing-value token, in both directions."""

#: Column aliases seen in the wild, mapped onto our field names. The keys are
#: lower-cased on read, so ``Name``, ``NAME`` and ``name`` all arrive here.
_ALIASES = {
    "name": "name", "label": "name", "channel": "name", "electrode": "name",
    "x": "x", "y": "y", "z": "z",
    "size": "size", "diameter": "size",
    "group": "group",
    "type": "group_type", "group_type": "group_type", "electrode_type": "group_type",
    "status": "status",
    "anatomy": "anatomy", "anat": "anatomy", "region": "anatomy",
    "label_destrieux": "anatomy", "label_dk": "anatomy", "aparc": "anatomy",
    "hemisphere": "hemisphere", "hemi": "hemisphere",
}

_TSV_COLUMNS = ("name", "x", "y", "z", "size", "group", "group_type", "status", "anatomy")


def _clean(value: Optional[str]) -> str:
    """A cell as a string, with BIDS' and everyone else's nulls flattened."""
    if value is None:
        return MISSING
    text = value.strip()
    return MISSING if text.lower() in (NA, "na", "nan", "none", "null", "") else text


def _as_float(value: Optional[str]) -> float:
    text = _clean(value)
    if text == MISSING:
        return float("nan")
    return float(text)


def read_electrodes_tsv(
    fname: Union[str, os.PathLike], subject: Optional[str] = None
) -> ElectrodeSet:
    """Read a BIDS-style ``*_electrodes.tsv`` (or ``.csv``).

    Parameters
    ----------
    fname : path
        Tab-separated by default; comma-separated if the name ends ``.csv``.
    subject : str, optional
        pycortex subject to attach the set to.

    Returns
    -------
    ElectrodeSet

    Notes
    -----
    A ``hemisphere`` column is read but not trusted: which hemisphere an
    electrode belongs to is decided by the anchoring, which simply takes
    whichever surface is nearer. A stated hemisphere that disagrees is
    surfaced by :func:`check_hemispheres` rather than silently overriding
    the geometry or being silently overridden by it.
    """
    delimiter = "," if str(fname).lower().endswith(".csv") else "\t"
    with open(fname, "r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError("%s has no rows" % fname)

    columns: dict[str, list[str]] = {}
    for raw_name in rows[0]:
        field = _ALIASES.get(str(raw_name).strip().lower())
        if field is not None and field not in columns:
            columns[field] = [row.get(raw_name, "") for row in rows]

    for required in ("name", "x", "y", "z"):
        if required not in columns:
            raise ValueError(
                "%s is missing a %r column (saw %s)"
                % (fname, required, ", ".join(sorted(rows[0])))
            )

    coords = np.array(
        [[_as_float(v) for v in columns[axis]] for axis in ("x", "y", "z")]
    ).T

    optional: dict[str, Any] = {
        field: [_clean(v) for v in columns[field]]
        for field in ("group", "anatomy", "status", "group_type")
        if field in columns
    }
    if "size" in columns:
        optional["size"] = [_as_float(v) for v in columns["size"]]

    return ElectrodeSet(
        names=[_clean(v) for v in columns["name"]],
        coords=coords,
        subject=subject,
        stated_hemisphere=columns.get("hemisphere"),
        **optional,
    )


def write_electrodes_tsv(eset: ElectrodeSet, fname: Union[str, os.PathLike]) -> None:
    """Write an electrode set as a BIDS-style ``*_electrodes.tsv``.

    Metadata and coordinates only: anchors are derived from a particular pair of
    surfaces and have no meaning in an interchange file. Use
    :func:`save_electrodes_json` to keep them.
    """
    delimiter = "," if str(fname).lower().endswith(".csv") else "\t"
    with open(fname, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow(_TSV_COLUMNS)
        for i in range(len(eset)):
            writer.writerow([
                eset.names[i],
                "%.6f" % eset.coords[i, 0],
                "%.6f" % eset.coords[i, 1],
                "%.6f" % eset.coords[i, 2],
                NA if np.isnan(eset.size[i]) else "%.4f" % eset.size[i],
                eset.group[i] or NA,
                eset.group_type[i] or NA,
                eset.status[i] or NA,
                eset.anatomy[i] or NA,
            ])


def to_dict(eset: ElectrodeSet) -> dict[str, Any]:
    """An electrode set as plain JSON-able types, anchors included."""
    out: dict[str, Any] = {
        "format": "pycortex-electrodes-1",
        "subject": eset.subject,
        "names": [str(n) for n in eset.names],
        "coords": eset.coords.tolist(),
        "group": [str(g) for g in eset.group],
        "size": [None if np.isnan(s) else float(s) for s in eset.size],
        "anatomy": [str(a) for a in eset.anatomy],
        "status": [str(s) for s in eset.status],
        "group_type": [str(t) for t in eset.group_type],
    }
    if eset.stated_hemisphere is not None:
        out["stated_hemisphere"] = [str(h) for h in eset.stated_hemisphere]
    if eset.anchors is not None:
        anchors = eset.anchors
        out["anchors"] = {
            "hemi": [str(h) for h in anchors.hemi],
            "verts": anchors.verts.tolist(),
            "weights": anchors.weights.tolist(),
            "depth": _nan_to_none(anchors.depth),
            "depth_mm": _nan_to_none(anchors.depth_mm),
            "thickness_mm": _nan_to_none(anchors.thickness_mm),
            "offset_mm": _nan_to_none(anchors.offset_mm),
            "placement": [str(p) for p in anchors.placement],
            "surface_hash": anchors.surface_hash,
        }
    return out


def from_dict(payload: dict[str, Any]) -> ElectrodeSet:
    """Rebuild an electrode set from :func:`to_dict`'s output."""
    anchors = None
    if payload.get("anchors"):
        raw = payload["anchors"]
        anchors = ElectrodeAnchors(
            hemi=np.array(raw["hemi"], dtype=str),
            verts=np.array(raw["verts"], dtype=np.intp),
            weights=np.array(raw["weights"], dtype=np.float64),
            depth=_none_to_nan(raw["depth"]),
            depth_mm=_none_to_nan(raw["depth_mm"]),
            thickness_mm=_none_to_nan(raw["thickness_mm"]),
            offset_mm=_none_to_nan(raw["offset_mm"]),
            placement=np.array(raw["placement"], dtype=str),
            surface_hash=raw.get("surface_hash", ""),
        )
    return ElectrodeSet(
        names=payload["names"],
        coords=np.array(payload["coords"], dtype=np.float64),
        subject=payload.get("subject"),
        group=payload.get("group"),
        size=[np.nan if s is None else s for s in payload["size"]] if payload.get("size") else None,
        anatomy=payload.get("anatomy"),
        status=payload.get("status"),
        group_type=payload.get("group_type"),
        stated_hemisphere=payload.get("stated_hemisphere"),
        anchors=anchors,
    )


def save_electrodes_json(eset: ElectrodeSet, fname: Union[str, os.PathLike]) -> None:
    """Write an electrode set, anchors and all, as JSON."""
    directory = os.path.dirname(os.path.abspath(str(fname)))
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(fname, "w", encoding="utf-8") as handle:
        json.dump(to_dict(eset), handle, indent=1)


def load_electrodes_json(fname: Union[str, os.PathLike]) -> ElectrodeSet:
    """Read an electrode set written by :func:`save_electrodes_json`."""
    with open(fname, "r", encoding="utf-8") as handle:
        return from_dict(json.load(handle))


def load_electrodes(
    fname: Union[str, os.PathLike], subject: Optional[str] = None
) -> ElectrodeSet:
    """Read an electrode set, picking the reader by extension."""
    lowered = str(fname).lower()
    if lowered.endswith(".json"):
        eset = load_electrodes_json(fname)
        if subject is not None:
            eset.subject = subject
        return eset
    if lowered.endswith(".mat"):
        return read_elecs_mat(fname, subject=subject)
    return read_electrodes_tsv(fname, subject=subject)


def check_hemispheres(eset: ElectrodeSet) -> list[str]:
    """Names whose stated hemisphere disagrees with where they anchored.

    Empty when the file stated none, or when everything agrees. A non-empty
    result is worth looking at before anything else: a whole group on the wrong
    side is the signature of a left-right flip somewhere upstream, and it is
    invisible on a flatmap, where both hemispheres are drawn side by side and a
    mirrored grid still looks like a grid.
    """
    stated = eset.stated_hemisphere
    if stated is None or eset.anchors is None:
        return []
    wrong = []
    for i, letter in enumerate(stated):
        if letter and letter in "lr" and letter != str(eset.anchors.hemi[i])[0]:
            wrong.append(str(eset.names[i]))
    return wrong


def _nan_to_none(values: Iterable[float]) -> list[Optional[float]]:
    return [None if np.isnan(v) else float(v) for v in values]


def _none_to_nan(values: Sequence[Optional[float]]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def read_elecs_mat(
    fname: Union[str, os.PathLike],
    subject: Optional[str] = None,
    drop_missing: bool = True,
) -> "ElectrodeSet":
    """Read an ``img_pipe``-style ``*_elecs_all.mat``.

    The format ``img_pipe`` writes and that several iEEG labs have standardised
    on. Three arrays, of which only the first is required:

    ``elecmatrix``
        ``(n, 3)`` coordinates in FreeSurfer surface RAS -- the same space
        pycortex holds its surfaces in, so no transform is needed.
    ``eleclabels``
        ``(n, 3)`` cell array: short channel name, long channel name, device
        type (``grid``, ``strip``, ``depth``).
    ``anatomy``
        ``(n, 4)`` cell array: those three plus an anatomical label.

    Parameters
    ----------
    fname : path
    subject : str, optional
        pycortex subject to attach the set to.
    drop_missing : bool
        Drop rows whose coordinates are not finite. On by default, because such
        rows are placeholders for unconnected amplifier channels rather than
        electrodes.

        Turn it off to keep them, so row indices still line up with a data array
        recorded on all the original channels; they then anchor to
        :data:`~cortex.electrodes.NO_COORDINATE` and draw nowhere. Note the
        catch: those rows are conventionally named ``NaN1``, ``NaN2`` and so on,
        and a montage with two disconnected blocks restarts that numbering, so
        the names collide and :class:`~cortex.electrodes.ElectrodeSet` refuses
        the set. Uniqueness is worth keeping -- names are how everything
        downstream identifies a contact -- so a file like that must be read with
        ``drop_missing=True``, which is why it is the default.

        Keying on the coordinate rather than on the name is deliberate: a
        non-finite coordinate means "no electrode here" in any lab's
        convention, where ``NaN1`` is one lab's spelling of it.

    Returns
    -------
    ElectrodeSet

    Notes
    -----
    The anatomy column mixes vocabularies, and that is not a defect to
    normalise away: one real montage carries Desikan-Killiany parcels
    (``superiortemporal``), Destrieux parcels (``ctx_lh_G_temporal_inf``),
    FreeSurfer aseg structures (``Left-Hippocampus``,
    ``Left-Cerebral-White-Matter``), the literal string ``Unknown``, and blanks,
    all in the same file. They are stored verbatim;
    :class:`~cortex.electrodes.PlacementPolicy` is what interprets them, and it
    is the aseg entries that tell you which contacts have no cortex to be on.
    """
    import scipy.io

    from ._set import ElectrodeSet

    mat = scipy.io.loadmat(str(fname))
    if "elecmatrix" not in mat:
        raise ValueError(
            "%s has no 'elecmatrix'; keys are %s"
            % (fname, ", ".join(k for k in mat if not k.startswith("__")))
        )

    coords = np.asarray(mat["elecmatrix"], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError("elecmatrix must be (n, 3), got %r" % (coords.shape,))
    coords = coords[:, :3]
    n = len(coords)

    table = _cell_to_str(mat.get("anatomy"))
    if table is None or len(table) != n:
        table = _cell_to_str(mat.get("eleclabels"))
    if table is not None and len(table) != n:
        table = None

    def column(index: int) -> Optional[list[str]]:
        if table is None or table.shape[1] <= index:
            return None
        return [_clean(v) for v in table[:, index]]

    names = column(0) or ["e%d" % (i + 1) for i in range(n)]
    group_type = column(2)
    anatomy = column(3)

    keep = np.asarray(
        np.isfinite(coords).all(axis=1) if drop_missing else np.ones(n),
        dtype=bool,
    )

    def subset(values: Optional[Sequence[str]]) -> Optional[list[str]]:
        if values is None:
            return None
        return [v for v, k in zip(values, keep.tolist()) if k]

    kept_names = subset(names)
    assert kept_names is not None                    # `names` is never None
    return ElectrodeSet(
        names=kept_names,
        coords=coords[keep],
        subject=subject,
        group_type=subset(group_type),
        anatomy=subset(anatomy),
    )


def _cell_to_str(cell: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """A MATLAB cell array of char as a 2-D array of str.

    ``loadmat`` hands back an object array whose entries are themselves ``(1, 1)``
    arrays of ``numpy.str_``, with empty cells arriving as zero-size arrays.
    """
    if cell is None:
        return None
    cell = np.atleast_2d(np.asarray(cell, dtype=object))
    out = np.empty(cell.shape, dtype=object)
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            value = cell[i, j]
            if value is None or (hasattr(value, "size") and value.size == 0):
                out[i, j] = MISSING
            else:
                out[i, j] = str(np.squeeze(value))
    return out
