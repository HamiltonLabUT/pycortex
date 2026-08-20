"""Reading and writing electrode sets.

Two formats, for two jobs.

**BIDS ``*_electrodes.tsv``** is the interchange format. Its columns --
``name``, ``x``, ``y``, ``z``, ``size``, ``group``, ``hemisphere``, ``type``,
``status`` -- map almost one to one onto the fields an
:class:`~cortex.electrodes.ElectrodeSet` holds, so files another lab already has
load without a bespoke parser, and files pycortex writes are readable by
everything else in the iEEG ecosystem. Missing values are ``n/a``, per BIDS.

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
    if str(fname).lower().endswith(".json"):
        eset = load_electrodes_json(fname)
        if subject is not None:
            eset.subject = subject
        return eset
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
