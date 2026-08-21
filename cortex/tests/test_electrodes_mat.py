"""Reading ``img_pipe``-style ``*_elecs_all.mat`` montages.

The fixtures here are synthetic, but their shape is taken from a real clinical
montage: a mixture of grids, strips and depths; an anatomy column carrying four
different vocabularies at once; and placeholder rows for unconnected amplifier
channels, which have non-finite coordinates and duplicated names.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex.electrodes import (
    NO_COORDINATE,
    NON_CORTICAL,
    ON_SURFACE,
    PlacementPolicy,
    anchor_to_surfaces,
    load_electrodes,
    read_elecs_mat,
)

from .test_electrode_anchor import plane_pair

# One real montage carried all four of these in a single anatomy column,
# alongside blanks: Desikan-Killiany parcels, Destrieux parcels, FreeSurfer
# aseg structures, and the literal string "Unknown".
ANATOMY = [
    "superiortemporal",                 # Desikan-Killiany
    "ctx_lh_G_temporal_inf",            # Destrieux
    "Left-Hippocampus",                 # aseg, subcortical
    "Left-Cerebral-White-Matter",       # aseg, white matter
]


def write_mat(path, coords, names, kinds, anatomy=None, include_anatomy=True):
    """A minimal MATLAB montage in the layout ``img_pipe`` writes."""
    import scipy.io

    def cell(values):
        out = np.empty((len(values), 1), dtype=object)
        for i, value in enumerate(values):
            out[i, 0] = np.array([value])
        return out

    labels = np.hstack([cell(names), cell(["Long%s" % n for n in names]), cell(kinds)])
    payload = {"elecmatrix": np.asarray(coords, dtype=float), "eleclabels": labels}
    if include_anatomy:
        payload["anatomy"] = np.hstack([labels, cell(anatomy or [""] * len(names))])
    scipy.io.savemat(str(path), payload)
    return path


@pytest.fixture
def montage(tmp_path):
    """Four real contacts, then two placeholders with no coordinates."""
    coords = np.array([
        [-20.0, 0.0, 0.0], [-20.0, 2.0, 0.0], [-20.0, 4.0, 0.0], [-20.0, 6.0, 0.0],
        [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
    ])
    return write_mat(
        tmp_path / "TDT_elecs_all.mat", coords,
        names=["TG1", "TG2", "AD1", "AD2", "NaN1", "NaN2"],
        kinds=["grid", "grid", "depth", "depth", "NaN", "NaN"],
        anatomy=ANATOMY + ["", ""],
    )


@pytest.fixture
def hemis():
    return {"lh": plane_pair(-30.0, -10.0), "rh": plane_pair(10.0, 30.0)}


# -- reading ---------------------------------------------------------------

def test_the_three_arrays_become_a_set(montage):
    eset = read_elecs_mat(montage, subject="TEST")
    assert list(eset.names) == ["TG1", "TG2", "AD1", "AD2"]
    assert list(eset.group_type) == ["grid", "grid", "depth", "depth"]
    assert list(eset.anatomy) == ANATOMY
    assert eset.subject == "TEST"


def test_groups_are_inferred_from_the_channel_names(montage):
    assert read_elecs_mat(montage).groups == ["TG", "AD"]


def test_placeholder_rows_are_dropped_by_default(montage):
    assert len(read_elecs_mat(montage)) == 4


def test_placeholder_rows_can_be_kept(montage):
    eset = read_elecs_mat(montage, drop_missing=False)
    assert len(eset) == 6
    assert not np.isfinite(eset.coords[4:]).any()


def test_a_file_without_an_anatomy_array_still_reads(tmp_path):
    path = write_mat(tmp_path / "e.mat", np.zeros((2, 3)), ["A1", "A2"],
                     ["grid", "grid"], include_anatomy=False)
    eset = read_elecs_mat(path)
    assert list(eset.names) == ["A1", "A2"]
    assert list(eset.group_type) == ["grid", "grid"]
    assert list(eset.anatomy) == ["", ""]           # no anatomy column to read


def test_a_file_without_elecmatrix_says_so(tmp_path):
    import scipy.io
    path = tmp_path / "e.mat"
    scipy.io.savemat(str(path), {"something_else": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="no 'elecmatrix'"):
        read_elecs_mat(path)


def test_load_electrodes_dispatches_on_the_extension(montage):
    assert len(load_electrodes(montage)) == 4


def test_duplicate_placeholder_names_are_refused_when_kept(tmp_path):
    """The documented limitation of drop_missing=False.

    A montage with two disconnected blocks restarts the placeholder numbering,
    so those names collide. Uniqueness is worth keeping -- names are how
    everything downstream identifies a contact -- so such a file must be read
    with the default, and the error names the offenders.
    """
    coords = np.vstack([np.zeros((1, 3)), np.full((2, 3), np.nan)])
    path = write_mat(tmp_path / "e.mat", coords, ["TG1", "NaN1", "NaN1"],
                     ["grid", "NaN", "NaN"])
    with pytest.raises(ValueError, match="unique"):
        read_elecs_mat(path, drop_missing=False)
    assert len(read_elecs_mat(path)) == 1


# -- what the anchoring makes of them --------------------------------------

def test_a_row_with_no_coordinate_anchors_to_nothing(montage, hemis):
    eset = read_elecs_mat(montage, drop_missing=False, subject="TEST")
    anchors = eset.anchor(surfaces=hemis)

    assert list(anchors.placement[4:]) == [NO_COORDINATE, NO_COORDINATE]
    assert not anchors.placeable[4:].any()
    # ...and the real ones are unaffected by their presence.
    assert np.isfinite(anchors.depth[:4]).all()
    assert anchors.placeable[:4].all()


def test_non_finite_coordinates_do_not_poison_the_others(hemis):
    """A NaN reaching cKDTree.query corrupts the whole result rather than
    failing, so the non-finite rows have to be held back from the search."""
    good = np.array([[-20.0, 0.0, -1.5]])
    mixed = np.vstack([good, [[np.nan] * 3], good])
    anchors = anchor_to_surfaces(mixed, hemis)
    assert np.allclose(anchors.depth[[0, 2]], 0.5)
    assert anchors.placement[1] == NO_COORDINATE


# -- the anatomy vocabulary ------------------------------------------------

@pytest.mark.parametrize(
    "label, expected",
    [
        ("superiortemporal", ON_SURFACE),           # Desikan-Killiany
        ("ctx_lh_G_temporal_inf", ON_SURFACE),      # Destrieux
        ("ctx_lh_S_circular_insula_sup", ON_SURFACE),
        ("insula", ON_SURFACE),
        # Near-misses. Parahippocampal gyrus is cortex; the pattern that must
        # catch "Left-Hippocampus" is one character away from swallowing it, so
        # it is "hippocampus" and not "hippocamp". Do not broaden it.
        ("parahippocampal", ON_SURFACE),
        ("ctx_lh_G_oc-temp_med-Parahip", ON_SURFACE),
        ("Left-Hippocampus", NON_CORTICAL),
        ("Left-Putamen", NON_CORTICAL),
        ("Left-VentralDC", NON_CORTICAL),
        ("Left-Cerebral-White-Matter", NON_CORTICAL),
        ("Right-Amygdala", NON_CORTICAL),
    ],
)
def test_aseg_labels_are_non_cortical_and_parcels_are_not(hemis, label, expected):
    """The one thing only the anatomy column knows.

    A contact in white matter or in a subcortical structure still has cortex a
    millimetre away on some sulcal bank, so the geometry cannot tell it apart
    from a contact genuinely in grey matter. The label can.
    """
    anchors = anchor_to_surfaces(
        np.array([[-20.0, 0.0, -1.5]]), hemis, anatomy=np.array([label])
    )
    assert anchors.placement[0] == expected


def test_non_cortical_contacts_are_still_placeable(hemis):
    """Marked, not hidden: where a hippocampal contact projects to is usually
    exactly what a reader wants to see."""
    anchors = anchor_to_surfaces(
        np.array([[-20.0, 0.0, -1.5]]), hemis, anatomy=np.array(["Left-Hippocampus"])
    )
    assert anchors.placeable[0]
    assert np.isfinite(anchors.depth[0])


def test_the_non_cortical_rule_can_be_turned_off(hemis):
    anchors = anchor_to_surfaces(
        np.array([[-20.0, 0.0, -1.5]]), hemis,
        policy=PlacementPolicy(flag_non_cortical=False),
        anatomy=np.array(["Left-Hippocampus"]),
    )
    assert anchors.placement[0] == ON_SURFACE


def test_the_non_cortical_rule_is_quiet_without_labels(hemis):
    """It is on by default, so it must not demand an anatomy column."""
    anchors = anchor_to_surfaces(np.array([[-20.0, 0.0, -1.5]]), hemis)
    assert anchors.placement[0] == ON_SURFACE
