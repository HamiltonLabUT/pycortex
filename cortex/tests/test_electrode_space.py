"""The electrode space and its three views.

Modelled on ``test_new_space.py``'s
``test_the_documented_skeleton_for_a_new_space_actually_works``, which is what
``cortex/dataset/ADDING_A_SPACE.md`` tells a new space to copy -- so most of this
file asks the same questions of ``Electrode`` that that one asks of a synthetic
space, plus the ones only a montage raises.

Nothing here needs real surfaces: the space's whole geometric question is "how
many contacts", so a scratch filestore holding coordinates and no anatomy is
enough. ``test_electrodes_subject.py`` covers anchoring against real surfaces.
"""

import os
import warnings

import cortex
import numpy as np
import pytest

from cortex.electrodes import ElectrodeSet

NAMES = ["LSTG%d" % i for i in range(1, 6)]
N = len(NAMES)


@pytest.fixture
def montages(tmp_path, monkeypatch):
    """A filestore with three bare subjects and montages relating them.

    S1 and S2 each have contacts localised in their own scan and again in
    ``fsaverage``. The coordinates differ between montages so that a view
    reloading the wrong one is visible rather than a coin flip.
    """
    for subject in ("S1", "S2", "fsaverage"):
        for directory in ("transforms", "anatomicals", "cache", "surfaces",
                          "surface-info", "views"):
            os.makedirs(tmp_path / subject / directory)

    db = cortex.database.db
    monkeypatch.setattr(db, "filestore", str(tmp_path))
    db.reload_subjects()

    rng = np.random.RandomState(0)
    db.save_montage("S1", ElectrodeSet(NAMES, rng.randn(N, 3), subject="S1"), "native")
    db.save_montage(
        "S1", ElectrodeSet(NAMES, rng.randn(N, 3) + 50, subject="S1"), "fsaverage"
    )
    db.save_montage(
        "S2", ElectrodeSet(["RIH%d" % i for i in range(1, 4)], rng.randn(3, 3),
                           subject="S2"), "fsaverage"
    )
    yield db
    db.reload_subjects()


@pytest.fixture
def values():
    return np.arange(float(N))


# -- the scalar view --------------------------------------------------------

def test_a_scalar_view_behaves_like_every_other_scalar_view(montages, values):
    """The inherited half: everything ADDING_A_SPACE.md promises comes for free."""
    view = cortex.Electrode(values, "S1", "native")

    assert view.subject == "S1"
    assert view.n_electrodes == N
    assert list(view.names) == NAMES
    assert view.shape == (N,)
    assert view.space.xfmname is None

    # the frame axis is added by the space's inherited `to_dense`
    assert view.renderer_data.shape == (1, N)
    assert cortex.Electrode(np.zeros((3, N)), "S1", "native").renderer_data.shape == (3, N)

    # bounds are resolved in ScalarView.__init__, so a new space inherits the
    # invariant rather than having to remember it
    assert view.vmin is not None and view.vmax is not None

    assert np.allclose((view + 1).data, values + 1)
    assert np.allclose((view * 2).data, values * 2)
    assert view.name.startswith("__") and len(view.name) == 18
    assert repr(view) == "<electrode data for (S1, native of S1)>"


def test_data_must_have_one_value_per_contact(montages):
    with pytest.raises(ValueError, match="7 values but the 'native' montage"):
        cortex.Electrode(np.arange(7.0), "S1", "native")


def test_no_data_means_a_zero_for_every_contact(montages):
    assert np.array_equal(cortex.Electrode(None, "S1", "native").data, np.zeros(N))


def test_empty_and_random_read_their_shape_from_the_montage(montages):
    assert np.all(cortex.Electrode.empty("S1", "native", 2).data == 2)
    assert cortex.Electrode.random("S1", "native").data.shape == (N,)


def test_raw_renders_through_the_space(montages, values):
    raw = cortex.Electrode(values, "S1", "native").raw
    assert isinstance(raw, cortex.ElectrodeRGB)
    assert raw.renderer_data.shape == (1, N, 4)
    assert raw.renderer_data.dtype == np.uint8


def test_an_unrecognized_keyword_is_rejected(montages, values):
    with pytest.raises(TypeError, match="cmpa"):
        cortex.Electrode(values, "S1", "native", cmpa="hot")


def test_attrs_is_the_route_for_metadata(montages, values):
    view = cortex.Electrode(values, "S1", "native", attrs={"stim": "a.mp4"})
    assert view.attrs["stim"] == "a.mp4"


# -- the montage, which is what this space adds ------------------------------

def test_a_template_montage_reports_the_template_as_its_subject(montages, values):
    """The load-bearing rule: `.subject` is whose surfaces, not whose electrodes.

    Every renderer reads `dataview.subject` to decide which brain to load --
    `get_flatmask`, `db.get_overlay`, and the line in `webgl/data.py` choosing
    which CTMs to pack. Reporting the implanted subject here would draw
    fsaverage coordinates on that patient's cortex.
    """
    view = cortex.Electrode(values, "S1", "fsaverage")
    assert view.subject == "fsaverage"
    assert view.montage_subjects == ("S1",)
    assert view.montage == "fsaverage"


def test_a_native_montage_has_one_subject_playing_both_parts(montages, values):
    view = cortex.Electrode(values, "S1", "native")
    assert view.subject == "S1" and view.montage_subjects == ("S1",)


def test_the_montage_decides_which_coordinates_are_used(montages, values):
    native = cortex.Electrode(values, "S1", "native")
    fsavg = cortex.Electrode(values, "S1", "fsaverage")
    assert not np.allclose(native.electrodes.coords, fsavg.electrodes.coords)


def test_a_montage_set_is_labelled_with_the_surface_subject(montages, values):
    """Which is what makes `anchor()` load the right surfaces.

    `ElectrodeSet.anchor` reads `eset.subject`, and `get_electrodes` labels a set
    with the subject whose directory it came from. Left alone, a template
    montage would anchor against the implanted subject's cortex -- silently, and
    plausibly, since a contact lands on some gyrus either way.
    """
    eset = cortex.Electrode(values, "S1", "fsaverage").electrodes
    assert eset.subject == "fsaverage"
    assert list(eset.owner) == ["S1"] * N


def test_a_missing_montage_says_which_ones_exist(montages, values):
    with pytest.raises(IOError, match="fsaverage, native"):
        cortex.Electrode(values, "S1", "cvs_avg35_inMNI152")


def test_to_sub_moves_the_data_to_another_subjects_surfaces(montages, values):
    view = cortex.Electrode(values, "S1", "native", cmap="hot", vmin=-1, vmax=1)
    moved = view.to_sub("fsaverage")

    assert moved.subject == "fsaverage"
    assert moved.montage_subjects == ("S1",)
    assert np.array_equal(moved.data, view.data)
    assert (moved.cmap, moved.vmin, moved.vmax) == ("hot", -1, 1)
    # and the coordinates really are the other file's
    assert not np.allclose(moved.electrodes.coords, view.electrodes.coords)


def test_to_sub_back_to_the_implanted_subject_is_the_native_montage(montages, values):
    moved = cortex.Electrode(values, "S1", "fsaverage").to_sub("S1")
    assert moved.montage == "native" and moved.subject == "S1"


def test_naming_a_subjects_own_montage_means_native(montages, values):
    assert cortex.Electrode(values, "S1", "S1").montage == "native"


# -- several subjects on one surface ----------------------------------------

def test_concat_puts_two_subjects_contacts_in_one_view(montages, values):
    both = cortex.Electrode.concat(
        sub="fsaverage", data={"S1": values, "S2": np.arange(3.0)}
    )
    assert both.subject == "fsaverage"
    assert both.montage_subjects == ("S1", "S2")
    assert both.n_electrodes == N + 3
    assert np.allclose(both.data, np.concatenate([values, np.arange(3.0)]))


def test_concat_namespaces_names_so_they_stay_unique(montages, values):
    both = cortex.Electrode.concat(
        sub="fsaverage", data={"S1": values, "S2": np.arange(3.0)}
    )
    assert list(both.names)[:2] == ["S1/LSTG1", "S1/LSTG2"]
    assert list(both.names)[-1] == "S2/RIH3"


def test_concat_records_which_subject_each_contact_came_from(montages, values):
    both = cortex.Electrode.concat(
        sub="fsaverage", data={"S1": values, "S2": np.arange(3.0)}
    )
    assert len(both.electrodes.select(owner="S1")) == N
    assert len(both.electrodes.select(owner="S2")) == 3


def test_concat_accepts_views_as_well_as_arrays(montages, values):
    both = cortex.Electrode.concat(
        sub="fsaverage",
        data={"S1": cortex.Electrode(values, "S1", "fsaverage"), "S2": np.arange(3.0)},
    )
    assert both.n_electrodes == N + 3


def test_concat_refuses_a_subjects_own_montage(montages, values):
    with pytest.raises(ValueError, match="both the surface subject"):
        cortex.Electrode.concat(sub="S1", data={"S1": values})


def test_concat_checks_each_subjects_length(montages, values):
    with pytest.raises(ValueError, match="S1 has 3 values"):
        cortex.Electrode.concat(
            sub="fsaverage", data={"S1": np.arange(3.0), "S2": np.arange(3.0)}
        )


# -- the composite columns ---------------------------------------------------

def test_a_2d_view_takes_arrays_or_views(montages, values):
    twod = cortex.Electrode2D(values, values * 2, "S1", "native")
    assert twod.renderer_data.shape == (1, N, 4)
    assert isinstance(twod.raw, cortex.ElectrodeRGB)
    assert repr(twod) == "<2D electrode data for (S1, native)>"

    from_views = cortex.Electrode2D(
        cortex.Electrode(values, "S1", "native"),
        cortex.Electrode(values * 2, "S1", "native"),
    )
    assert isinstance(from_views.dim1, cortex.Electrode)


def test_a_2d_view_over_raw_arrays_needs_the_montage_named(montages, values):
    with pytest.raises(TypeError, match="montage"):
        cortex.Electrode2D(values, values, "S1")


def test_a_2d_view_accepts_an_explicit_montage_matching_its_channels(montages, values):
    """The surface subject is what `_resolve_channels` compares against.

    A composite constructor takes the *owner* as `subject`, matching
    `Electrode`, but the channels report the template. Forwarding the owner
    unchanged would reject this pair with "subject in Electrode objects
    ('fsaverage') is different than specified subject ('S1')".
    """
    twod = cortex.Electrode2D(
        cortex.Electrode(values, "S1", "fsaverage"),
        cortex.Electrode(values * 2, "S1", "fsaverage"),
        subject="S1",
        montage="fsaverage",
    )
    assert twod.subject == "fsaverage"


def test_two_subjects_on_one_template_cannot_be_colormapped_together(montages, values):
    """`align` is what catches this; nothing upstream can.

    `_resolve_channels` only compares `subject`, which two montages agree on as
    soon as they are drawn on the same template. Equal lengths would then let a
    2D view colormap one subject's values against another's.
    """
    s1 = cortex.Electrode(np.arange(3.0), "S1", "fsaverage",
                          electrodes=montages.get_montage("S2", "fsaverage"))
    s2 = cortex.Electrode(np.arange(3.0), "S2", "fsaverage")
    with pytest.raises(ValueError, match="different montages"):
        cortex.Electrode2D(s1, s2).raw


def test_an_rgb_view_stacks_three_channels(montages, values):
    rgb = cortex.ElectrodeRGB(values, values, values, "S1", "native")
    assert rgb.renderer_data.shape == (1, N, 4)
    assert repr(rgb) == "<RGB electrode data for (S1, native)>"

    movie = cortex.ElectrodeRGB(*([np.random.randn(4, N)] * 3), "S1", "native")
    assert movie.renderer_data.shape == (4, N, 4)


def test_the_composite_views_move_between_subjects_too(montages, values):
    twod = cortex.Electrode2D(values, values * 2, "S1", "native").to_sub("fsaverage")
    rgb = cortex.ElectrodeRGB(values, values, values, "S1", "native").to_sub("fsaverage")
    assert twod.subject == "fsaverage" and rgb.subject == "fsaverage"


def test_uniques_yields_the_channels(montages, values):
    twod = cortex.Electrode2D(values, values * 2, "S1", "native")
    rgb = cortex.ElectrodeRGB(values, values, values, "S1", "native")
    assert [type(u).__name__ for u in twod.uniques()] == ["Electrode", "Electrode"]
    assert [type(u).__name__ for u in rgb.uniques(collapse=True)] == ["ElectrodeRGB"]


# -- serialization -----------------------------------------------------------

def test_the_simple_json_describes_the_montages_layout(montages, values):
    simple = cortex.Electrode(values, "S1", "native").to_json(simple=True)
    assert set(simple) == {"name", "subject", "min", "max", "nelec", "frames"}
    assert simple["nelec"] == N
    assert simple["subject"] == "S1"


def test_all_three_columns_survive_an_hdf_round_trip(montages, values, tmp_path):
    views = {
        "scalar": cortex.Electrode(values, "S1", "native", attrs={"stim": "a.mp4"}),
        "fsavg": cortex.Electrode(values, "S1", "fsaverage"),
        "twod": cortex.Electrode2D(values, values * 2, "S1", "native"),
        "rgb": cortex.ElectrodeRGB(values, values, values, "S1", "native"),
    }
    fname = str(tmp_path / "electrodes.hdf")
    cortex.Dataset(**views).save(fname)
    back = cortex.load(fname)

    assert {k: type(back[k]).__name__ for k in sorted(back.views)} == {
        "fsavg": "Electrode",
        "rgb": "ElectrodeRGB",
        "scalar": "Electrode",
        "twod": "Electrode2D",
    }
    assert np.allclose(back["scalar"].data, values)
    assert back["scalar"].attrs["stim"] == "a.mp4"
    assert back["scalar"].montage == "native"
    assert back["fsavg"].montage == "fsaverage"
    assert back["fsavg"].subject == "fsaverage"


def test_two_montages_of_one_array_do_not_share_an_hdf_node(montages, values, tmp_path):
    """Which they would if the node name hashed the values alone.

    A data node carries its space's attributes, and `_write_data_hdf` skips a
    node whose contents already hash to its name -- so the second of two
    identical arrays would inherit the first's montage and reload onto the wrong
    subject's cortex. A volumetric view survives the same collision because HDF
    slot 7 repeats its transform name per view; an electrode space has no
    transform, so slot 7 is null and the node is the only record.
    """
    native = cortex.Electrode(values, "S1", "native")
    fsavg = cortex.Electrode(values, "S1", "fsaverage")
    assert np.array_equal(native.data, fsavg.data)
    assert native.name != fsavg.name

    fname = str(tmp_path / "both.hdf")
    cortex.Dataset(native=native, fsavg=fsavg).save(fname)
    back = cortex.load(fname)
    assert back["native"].subject == "S1"
    assert back["fsavg"].subject == "fsaverage"


def test_a_saved_view_reloads_without_the_filestore(montages, values, tmp_path):
    """The set rides inside the file, so it is not a reference to a montage.

    `from_hdf` is handed the node's attributes and nothing else -- not the file
    -- so it cannot read a `/subjects` group the way `Dataset` reads the surfaces
    and ROIs it packs. Embedding is the only way this works.
    """
    import shutil

    view = cortex.Electrode(values, "S1", "fsaverage")
    fname = str(tmp_path / "portable.hdf")
    cortex.Dataset(x=view).save(fname)

    shutil.rmtree(os.path.join(montages.filestore, "S1", "electrodes"))
    back = cortex.load(fname)
    assert back["x"].n_electrodes == N
    assert np.allclose(back["x"].electrodes.coords, view.electrodes.coords)


def test_a_montage_too_large_to_embed_says_so_and_falls_back(montages, tmp_path):
    """Past h5py's attribute cap the name is written alone, loudly.

    An attribute lives in the object header, which the default file format caps
    at 64 KB; a few hundred contacts with anchors reach that. The data is what
    the user asked to keep, so this warns and writes a weaker file rather than
    refusing to save.
    """
    big = 4000
    huge = ElectrodeSet(
        ["E%d" % i for i in range(big)],
        np.random.RandomState(1).randn(big, 3),
        subject="S1",
    )
    montages.save_montage("S1", huge, "huge")
    view = cortex.Electrode(np.arange(float(big)), "S1", "huge")

    fname = str(tmp_path / "huge.hdf")
    with pytest.warns(UserWarning, match="too large to store"):
        cortex.Dataset(x=view).save(fname)

    back = cortex.load(fname)
    assert back["x"].montage == "huge"
    assert back["x"].n_electrodes == big


# -- how it fits the rest of the package -------------------------------------

def test_the_electrode_space_registers_ahead_of_the_catch_all(montages):
    """Which is what stops SurfaceSpace claiming an electrode node.

    The catch-all accepts anything without a transform, and an electrode space
    has none. `from_hdf` therefore discriminates on `montage`, an attribute this
    space writes itself, and `register_space` puts every non-fallback ahead of
    every fallback.
    """
    order = [c.__name__ for c in cortex.dataset.registered_spaces()]
    assert order.index("ElectrodeSpace") < order.index("SurfaceSpace")
    assert not cortex.dataset.ElectrodeSpace.fallback


def test_an_electrode_view_is_renderable_but_neither_of_the_older_kinds(montages, values):
    from cortex.dataset import ElectrodeView, SurfaceView, VolumetricView

    view = cortex.Electrode(values, "S1", "native")
    assert cortex.dataset.as_renderable(view) is view
    assert isinstance(view, ElectrodeView)
    assert not isinstance(view, (VolumetricView, SurfaceView))
    # The third spatial interface publishes the column's array under its own
    # name, as `volume` and `vertices` do -- it does not implement a second one.
    assert np.array_equal(view.contacts, view.renderer_data)
    assert "contacts" in vars(ElectrodeView)
    assert not any("contacts" in vars(cls) for cls in type(view).__mro__
                   if cls is not ElectrodeView)


def test_the_webgl_payload_is_the_electrode_one(montages, values):
    view = cortex.Electrode(values, "S1", "native")
    payload = view.space.pack_for_webgl(view.renderer_data, raw=False)
    assert isinstance(payload, cortex.dataset.ElectrodeValues)
    assert payload.describe() == {"raw": False}
    # serialised in `reorder`, like the per-vertex encoding, but without the
    # permutation that one applies
    served = payload.reorder(payload.frames, vertex_index=None)
    assert isinstance(served[0], bytes)


def test_a_tuple_carrying_a_set_normalizes_to_an_electrode_view(montages, values):
    eset = montages.get_montage("S1", "native")
    view = cortex.dataset.normalize((values, "S1", eset))
    assert isinstance(view, cortex.Electrode)
    assert view.n_electrodes == N
