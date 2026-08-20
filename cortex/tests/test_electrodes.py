"""The electrode set: construction, grouping, selection and file formats."""

from __future__ import annotations

import numpy as np
import pytest

from cortex.electrodes import (
    ON_SURFACE,
    ElectrodeInfo,
    ElectrodeSet,
    PlacementPolicy,
    check_hemispheres,
    from_dict,
    infer_group,
    infer_index,
    load_electrodes,
    load_electrodes_json,
    read_electrodes_tsv,
    save_electrodes_json,
    to_dict,
    write_electrodes_tsv,
)

from .test_electrode_anchor import plane_pair


@pytest.fixture
def hemis():
    return {"lh": plane_pair(-30.0, -10.0), "rh": plane_pair(10.0, 30.0)}


@pytest.fixture
def eset():
    """Two groups, one per hemisphere, sitting just above the test sheets."""
    return ElectrodeSet(
        names=["LSTG1", "LSTG2", "LSTG3", "RPCING1", "RPCING2"],
        coords=np.array([
            [-20.0, -4.0, 1.0],
            [-20.0, 0.0, 1.0],
            [-20.0, 4.0, 1.0],
            [20.0, 0.0, 1.0],
            [20.0, 4.0, 1.0],
        ]),
        subject="TEST",
        group_type=["grid", "grid", "grid", "seeg", "seeg"],
        status=["good", "good", "bad", "good", "good"],
        anatomy=["STG", "STG", "Unknown", "PCC", "PCC"],
        size=[2.3, 2.3, 2.3, 0.8, 0.8],
    )


# -- group inference --------------------------------------------------------

@pytest.mark.parametrize(
    "name, group, index",
    [
        ("LSTG1", "LSTG", 1),
        ("LSTG12", "LSTG", 12),
        ("RPCING01", "RPCING", 1),
        ("LSTG_2", "LSTG", 2),
        ("LSTG-3", "LSTG", 3),
        ("LSTG.4", "LSTG", 4),
        ("  LSTG 5 ", "LSTG", 5),
        ("REF", "REF", None),        # no number at all
        ("7", "7", 7),               # all digits: its own group, not the empty one
    ],
)
def test_group_and_index_are_read_off_the_channel_name(name, group, index):
    assert infer_group(name) == group
    assert infer_index(name) == index


def test_an_explicit_group_column_wins_over_inference():
    eset = ElectrodeSet(
        names=["A1", "A2"], coords=np.zeros((2, 3)), group=["strip_one", "strip_two"]
    )
    assert list(eset.group) == ["strip_one", "strip_two"]


# -- construction -----------------------------------------------------------

def test_optional_fields_default_to_missing_rather_than_none():
    eset = ElectrodeSet(names=["A1"], coords=np.zeros((1, 3)))
    assert eset.anatomy[0] == ""
    assert eset.status[0] == ""
    assert np.isnan(eset.size[0])
    assert eset.stated_hemisphere is None


def test_duplicate_names_are_refused():
    with pytest.raises(ValueError, match="unique"):
        ElectrodeSet(names=["A1", "A1"], coords=np.zeros((2, 3)))


def test_coords_must_be_n_by_three():
    with pytest.raises(ValueError, match=r"\(2, 3\)"):
        ElectrodeSet(names=["A1", "A2"], coords=np.zeros((2, 2)))


def test_a_mismatched_metadata_column_is_refused():
    with pytest.raises(ValueError, match="status"):
        ElectrodeSet(names=["A1", "A2"], coords=np.zeros((2, 3)), status=["good"])


def test_an_empty_set_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        ElectrodeSet(names=[], coords=np.zeros((0, 3)))


# -- access -----------------------------------------------------------------

def test_an_integer_or_a_name_gives_one_electrode(eset):
    by_index, by_name = eset[0], eset["LSTG1"]
    assert isinstance(by_index, ElectrodeInfo)
    assert by_index.name == by_name.name == "LSTG1"
    assert by_index.group == "LSTG"
    assert by_index.group_type == "grid"


def test_an_unknown_name_raises(eset):
    with pytest.raises(KeyError, match="NOPE"):
        eset["NOPE"]


def test_a_mask_or_slice_gives_a_subset(eset):
    subset = eset[np.array([True, True, False, False, False])]
    assert isinstance(subset, ElectrodeSet)
    assert list(subset.names) == ["LSTG1", "LSTG2"]
    assert len(eset[:2]) == 2
    assert list(eset[["RPCING1", "LSTG3"]].names) == ["RPCING1", "LSTG3"]


def test_iterating_yields_one_record_per_electrode(eset):
    assert [e.name for e in eset] == list(eset.names)


def test_groups_keep_first_appearance_order(eset):
    assert eset.groups == ["LSTG", "RPCING"]


def test_repr_says_what_it_holds(eset):
    assert "5 electrodes" in repr(eset)
    assert "2 groups" in repr(eset)
    assert "TEST" in repr(eset)


# -- selection --------------------------------------------------------------

def test_select_on_one_field(eset):
    assert len(eset.select(group="LSTG")) == 3


def test_select_accepts_several_values(eset):
    assert len(eset.select(group_type=["grid", "seeg"])) == 5
    assert len(eset.select(group_type=["seeg"])) == 2


def test_select_combines_criteria_with_and(eset):
    assert list(eset.select(group="LSTG", status="good").names) == ["LSTG1", "LSTG2"]


def test_select_takes_an_explicit_mask(eset):
    mask = eset.coords[:, 1] > 0
    assert list(eset.select(where=mask).names) == ["LSTG3", "RPCING2"]


def test_select_on_an_unknown_field_says_so(eset):
    with pytest.raises(KeyError, match="not anchored"):
        eset.select(hemi="lh")


def test_select_leaves_the_original_alone(eset):
    before = len(eset)
    eset.select(group="LSTG")
    assert len(eset) == before


def test_placeable_selection_needs_anchors(eset):
    with pytest.raises(ValueError, match="anchored"):
        eset.select(placeable=True)


# -- anchoring through the set ---------------------------------------------

def test_anchor_stores_on_the_set_and_enables_the_derived_fields(eset, hemis):
    assert eset.anchors is None
    anchors = eset.anchor(surfaces=hemis)
    assert eset.anchors is anchors
    assert list(anchors.hemi) == ["lh", "lh", "lh", "rh", "rh"]
    assert np.allclose(eset.depth, -1.0 / 3.0)          # 1 mm above a 3 mm ribbon
    assert list(eset.select(hemi="lh").names) == ["LSTG1", "LSTG2", "LSTG3"]


def test_an_anchored_record_carries_hemisphere_and_depth(eset, hemis):
    eset.anchor(surfaces=hemis)
    record = eset["RPCING1"]
    assert record.hemi == "rh"
    assert np.isclose(record.depth, -1.0 / 3.0)


def test_subsetting_carries_the_anchors_along(eset, hemis):
    eset.anchor(surfaces=hemis)
    subset = eset.select(group="RPCING")
    assert subset.anchors is not None
    assert list(subset.anchors.hemi) == ["rh", "rh"]


def test_anchoring_without_a_subject_or_surfaces_says_so():
    bare = ElectrodeSet(names=["A1"], coords=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="subject"):
        bare.anchor()


def test_reclassify_reuses_the_geometry(eset, hemis):
    eset.anchor(surfaces=hemis)
    verts = eset.anchors.verts.copy()
    placement = eset.reclassify(PlacementPolicy(drop_unknown_anatomy=True))
    assert placement[2] == "unknown_anatomy"        # the one labelled "Unknown"
    assert placement[0] == "projected"
    assert np.array_equal(eset.anchors.verts, verts)


def test_positions_needs_anchors(eset):
    with pytest.raises(ValueError, match="anchor"):
        eset.positions("flat")


# -- tsv --------------------------------------------------------------------

def test_tsv_round_trip(eset, tmp_path):
    path = tmp_path / "sub-01_electrodes.tsv"
    write_electrodes_tsv(eset, path)
    back = read_electrodes_tsv(path, subject="TEST")

    assert list(back.names) == list(eset.names)
    assert np.allclose(back.coords, eset.coords)
    assert list(back.group_type) == list(eset.group_type)
    assert list(back.anatomy) == list(eset.anatomy)
    assert np.allclose(back.size, eset.size)
    assert back.subject == "TEST"


def test_csv_is_read_by_extension(eset, tmp_path):
    path = tmp_path / "electrodes.csv"
    write_electrodes_tsv(eset, path)
    assert path.read_text().splitlines()[0].count(",") == 8
    assert list(read_electrodes_tsv(path).names) == list(eset.names)


def test_column_aliases_are_accepted(tmp_path):
    path = tmp_path / "e.tsv"
    path.write_text("label\tx\ty\tz\themisphere\telectrode_type\n"
                    "LSTG1\t-20\t0\t1\tL\tgrid\n")
    eset = read_electrodes_tsv(path)
    assert list(eset.names) == ["LSTG1"]
    assert list(eset.group_type) == ["grid"]
    assert list(eset.stated_hemisphere) == ["l"]


def test_bids_missing_values_become_missing(tmp_path):
    path = tmp_path / "e.tsv"
    path.write_text("name\tx\ty\tz\tsize\tgroup\n"
                    "LSTG1\t-20\t0\t1\tn/a\tn/a\n")
    eset = read_electrodes_tsv(path)
    assert np.isnan(eset.size[0])
    assert eset.group[0] == ""


def test_a_missing_coordinate_column_names_itself(tmp_path):
    path = tmp_path / "e.tsv"
    path.write_text("name\tx\ty\nLSTG1\t-20\t0\n")
    with pytest.raises(ValueError, match="'z'"):
        read_electrodes_tsv(path)


def test_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "e.tsv"
    path.write_text("name\tx\ty\tz\n")
    with pytest.raises(ValueError, match="no rows"):
        read_electrodes_tsv(path)


# -- json -------------------------------------------------------------------

def test_json_round_trip_keeps_the_anchors(eset, hemis, tmp_path):
    eset.anchor(surfaces=hemis)
    path = tmp_path / "sub" / "electrodes" / "clinical.json"
    save_electrodes_json(eset, path)
    back = load_electrodes_json(path)

    assert back.anchors is not None
    assert np.array_equal(back.anchors.verts, eset.anchors.verts)
    assert np.allclose(back.anchors.weights, eset.anchors.weights)
    assert np.allclose(back.anchors.depth, eset.anchors.depth)
    assert list(back.anchors.placement) == list(eset.anchors.placement)
    assert back.anchors.surface_hash == eset.anchors.surface_hash
    assert back.subject == "TEST"


def test_json_round_trip_without_anchors(eset, tmp_path):
    path = tmp_path / "e.json"
    save_electrodes_json(eset, path)
    assert load_electrodes_json(path).anchors is None


def test_nans_survive_the_json_round_trip(tmp_path):
    eset = ElectrodeSet(names=["A1"], coords=np.zeros((1, 3)))
    assert np.isnan(from_dict(to_dict(eset)).size[0])


def test_load_electrodes_picks_the_reader_by_extension(eset, tmp_path):
    tsv, js = tmp_path / "e.tsv", tmp_path / "e.json"
    write_electrodes_tsv(eset, tsv)
    save_electrodes_json(eset, js)
    assert len(load_electrodes(tsv)) == len(load_electrodes(js)) == len(eset)
    assert load_electrodes(js, subject="OTHER").subject == "OTHER"


# -- the hemisphere cross-check --------------------------------------------

def test_a_stated_hemisphere_that_disagrees_is_reported(eset, hemis):
    eset.stated_hemisphere = np.array(["l", "l", "l", "l", "r"])  # RPCING1 is wrong
    eset.anchor(surfaces=hemis)
    assert check_hemispheres(eset) == ["RPCING1"]


def test_nothing_is_reported_when_the_file_stated_nothing(eset, hemis):
    eset.anchor(surfaces=hemis)
    assert check_hemispheres(eset) == []


def test_nothing_is_reported_before_anchoring(eset):
    eset.stated_hemisphere = np.array(["r", "r", "r", "r", "r"])
    assert check_hemispheres(eset) == []


def test_the_stated_hemisphere_does_not_override_the_geometry(eset, hemis):
    eset.stated_hemisphere = np.array(["r"] * 5)
    eset.anchor(surfaces=hemis)
    assert list(eset.anchors.hemi) == ["lh", "lh", "lh", "rh", "rh"]
    assert eset.anchors.placement[0] == "projected"
