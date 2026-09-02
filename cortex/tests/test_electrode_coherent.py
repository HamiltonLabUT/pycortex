"""Choosing a shank's anchors together instead of one contact at a time.

The synthetic surface here is the smallest thing that reproduces the real
failure: two parallel sheets three millimetres apart -- the two banks of a
sulcus -- which inflation pulls far apart. A shaft running down the gap between
them is very nearly equidistant from both, so anchoring each contact
independently lets sub-millimetre noise decide which bank it lands on, and the
device arrives on the inflated surface as a zig-zag between two places
centimetres apart.

That is exactly what happens on a real brain, where it is invisible in the input
and obvious only after inflation. Here the right answer is known by
construction: pick one bank and stay on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex.electrodes import SurfacePair, anchor_to_surfaces
from cortex.electrodes._coherent import (
    MAX_FIDELITY_LOSS_MM,
    CoherenceReport,
    coherent_anchors,
    solve_device,
)

from .test_electrode_anchor import plane


def bank(x0, x1, z, n=21, y0=-30.0, y1=30.0):
    return plane(x0, x1, y0=y0, y1=y1, n=n, z=z)


def two_banks(separation=3.0, thickness=1.0):
    """Two facing sheets, and an "inflated" surface that pulls them apart.

    The pial surface is the two banks at ``z = 0`` and ``z = -separation``. The
    inflated surface moves them 60 mm apart in z and leaves everything else
    alone, which is the one thing inflation does that matters here: two points
    that were three millimetres apart end up sixty.
    """
    upper, upoly = bank(-30.0, -10.0, 0.0)
    lower, lopoly = bank(-30.0, -10.0, -separation)
    pia = np.vstack([upper, lower])
    polys = np.vstack([upoly, lopoly + len(upper)])

    # White matter, one millimetre behind each bank -- outward for the upper
    # sheet, outward for the lower, so the ribbon faces the gap on both sides.
    wm = np.vstack([upper + [0, 0, thickness], lower - [0, 0, thickness]])

    inflated = pia.copy()
    inflated[: len(upper), 2] += 30.0
    inflated[len(upper) :, 2] -= 30.0
    return SurfacePair(pia=pia, polys=polys, wm=wm), inflated, len(upper)


def shaft_in_the_gap(n=12, pitch=4.0, separation=3.0, wobble=0.02):
    """Contacts down the middle of the gap, with a hair of noise.

    The noise is what makes the test mean something: without it every contact is
    exactly equidistant and the tie is broken by floating point. With it, the
    nearer bank alternates -- which is precisely the real failure, arriving from
    measurement noise rather than from anything anatomical.
    """
    rng = np.random.default_rng(0)
    y = -20.0 + np.arange(n) * pitch
    z = -separation / 2.0 + rng.uniform(-wobble, wobble, n)
    return np.column_stack([np.full(n, -20.0), y, z])


# -- the failure, and the fix ----------------------------------------------

def test_independent_anchoring_alternates_between_the_banks():
    """The premise. Without this the rest of the file proves nothing."""
    pair, inflated, split = two_banks()
    coords = shaft_in_the_gap()
    anchors = anchor_to_surfaces(coords, {"lh": pair})

    side = anchors.verts[:, 0] >= split
    assert side.min() != side.max(), "the shaft should straddle both banks"
    # It does not merely straddle, it alternates: several changes of bank along
    # a device that is a straight line.
    assert int(np.abs(np.diff(side.astype(int))).sum()) >= 3


def test_choosing_together_keeps_the_shaft_on_one_bank():
    pair, inflated, split = two_banks()
    coords = shaft_in_the_gap()
    solved = solve_device(coords, pair, inflated)

    side = solved.verts[:, 0] >= split
    assert side.min() == side.max(), "every contact should sit on one bank"
    assert solved.switched.sum() == 0
    assert solved.fidelity_loss_mm.max() <= MAX_FIDELITY_LOSS_MM


def test_the_spacing_is_recovered():
    """The point of the exercise, in millimetres.

    Independent anchoring throws the inflated spacing wildly off because half
    the contacts are 60 mm away on the other bank. Choosing together restores
    it to the pitch it should have.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(pitch=4.0)
    pitch = np.linalg.norm(np.diff(coords, axis=0), axis=1)

    greedy = anchor_to_surfaces(coords, {"lh": pair}).evaluate({"lh": inflated})
    joint = solve_device(coords, pair, inflated)
    together = np.einsum(
        "mij,mi->mj", inflated[joint.verts], joint.weights
    )

    gap_greedy = np.linalg.norm(np.diff(greedy, axis=0), axis=1)
    gap_joint = np.linalg.norm(np.diff(together, axis=0), axis=1)

    assert gap_greedy.max() > 50.0                     # jumps the 60 mm gap
    assert np.abs(gap_joint - pitch).max() < 0.5       # and this does not


# -- the guarantee ---------------------------------------------------------

def test_the_fidelity_cap_is_hard_not_a_penalty():
    """No weighting can push a contact off cortex it genuinely touches.

    This is the property a continuous optimiser cannot offer, and the reason
    anatomical labelling is preserved by construction rather than hoped for:
    candidates outside the cap are not in the state space at all, so the
    coherence term has nothing to trade with.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap()
    for weight in (1.0, 100.0, 1e6):
        solved = solve_device(coords, pair, inflated, coherence_weight=weight)
        assert solved.fidelity_loss_mm.max() <= MAX_FIDELITY_LOSS_MM + 1e-9

    tight = solve_device(coords, pair, inflated, max_fidelity_loss_mm=0.0,
                         coherence_weight=1e6)
    assert tight.fidelity_loss_mm.max() <= 1e-9


def test_zero_coherence_reproduces_independent_anchoring():
    """The lower end of the one parameter, so its meaning is pinned.

    With the coherence term switched off the DP has nothing to prefer but
    fidelity, and fidelity per contact is what ordinary anchoring already
    minimises -- so the two must agree.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap()
    solved = solve_device(coords, pair, inflated, coherence_weight=0.0)
    greedy = anchor_to_surfaces(coords, {"lh": pair})
    assert np.allclose(np.sort(solved.verts, axis=1),
                       np.sort(greedy.verts, axis=1))


def test_a_forced_break_is_reported_rather_than_smoothed_away():
    """Where anatomy genuinely separates, the DP breaks and says so.

    Two contacts placed on banks that inflation pulls apart, with nothing in
    between to anchor to. The cap forbids the compromise, so the spacing breaks
    -- and that break is information about the anatomy, which is why it comes
    back in the report instead of being hidden.
    """
    pair, inflated, split = two_banks()
    # One contact hard against each bank: no near-tie, no choice to make.
    coords = np.array([[-20.0, 0.0, -0.05], [-20.0, 4.0, -2.95]])
    solved = solve_device(coords, pair, inflated)
    assert (solved.verts[0, 0] >= split) != (solved.verts[1, 0] >= split)
    assert solved.switched.sum() == 1


# -- the montage-level entry point ------------------------------------------

def test_only_shanks_are_re_anchored():
    """A grid's contacts do each belong to the column nearest them.

    Re-choosing their anchors would be solving a problem they do not have, so
    ``coherent_anchors`` leaves them exactly as anchoring left them.
    """
    pair, inflated, _ = two_banks()
    shaft = shaft_in_the_gap(n=8)
    grid = np.array([
        [-24.0 + i * 3.0, -6.0 + j * 3.0, 1.5] for i in range(3) for j in range(3)
    ])
    coords = np.vstack([shaft, grid])
    groups = ["LTD"] * 8 + ["LG"] * 9
    types = ["seeg"] * 8 + ["grid"] * 9

    before = anchor_to_surfaces(coords, {"lh": pair})
    after, report = coherent_anchors(
        before, coords, {"lh": pair}, {"lh": inflated}, groups, types
    )
    assert report.devices == 1 and report.contacts == 8
    assert not np.array_equal(after.verts[:8], before.verts[:8])
    assert np.array_equal(after.verts[8:], before.verts[8:])
    assert np.array_equal(after.weights[8:], before.weights[8:])


def test_an_unlabelled_shank_is_recognised_from_its_geometry():
    """Two of three real montages record no ``group_type`` at all.

    Every one of TCH06's twenty-one groups is a depth electrode, so keying only
    on the label would leave all 186 of its contacts anchored independently for
    no better reason than a missing column.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(n=8)
    before = anchor_to_surfaces(coords, {"lh": pair})
    _, report = coherent_anchors(
        before, coords, {"lh": pair}, {"lh": inflated},
        ["LTD"] * 8, [""] * 8,
    )
    assert report.devices == 1 and report.contacts == 8


def test_an_explicit_grid_label_excludes_a_device_that_is_straight():
    """A 1xN grid is a legal thing to record; do not second-guess the column.

    Label and geometry both have to agree before a device is re-anchored, and
    they gate in opposite directions: a ``grid`` or ``strip`` label excludes
    outright however needle-straight the contacts happen to be, and a device
    that is not straight is excluded however confidently it is labelled
    ``seeg``.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(n=8)
    before = anchor_to_surfaces(coords, {"lh": pair})
    _, report = coherent_anchors(
        before, coords, {"lh": pair}, {"lh": inflated},
        ["LG"] * 8, ["grid"] * 8,
    )
    assert report.devices == 0


def test_a_scattered_group_is_excluded_even_when_labelled_seeg():
    """The label states an intent; the geometry says whether the model applies.

    Six contacts sharing a name but strewn across the surface are not a shank,
    and the coherence term -- which compares an inflated gap against a native
    spacing -- has nothing to say about them. Left alone, the DP would shuffle
    their anchors within the cap for no benefit, which is how this was found:
    a quickflat fixture of scattered contacts labelled ``seeg`` had its depths
    quietly moved.
    """
    pair, inflated, _ = two_banks()
    rng = np.random.default_rng(3)
    coords = np.column_stack([
        rng.uniform(-28.0, -12.0, 6),
        rng.uniform(-25.0, 25.0, 6),
        rng.uniform(-2.5, -0.5, 6),
    ])
    before = anchor_to_surfaces(coords, {"lh": pair})
    after, report = coherent_anchors(
        before, coords, {"lh": pair}, {"lh": inflated},
        ["LD"] * 6, ["seeg"] * 6,
    )
    assert report.devices == 0
    assert np.array_equal(after.verts, before.verts)


def test_a_montage_with_no_groups_is_left_entirely_alone():
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(n=6)
    before = anchor_to_surfaces(coords, {"lh": pair})
    after, report = coherent_anchors(
        before, coords, {"lh": pair}, {"lh": inflated}, groups=None
    )
    assert after is before
    assert report.devices == 0


def test_a_missing_coordinate_is_excluded_rather_than_anchored():
    """A NaN row is an unconnected amplifier channel, not an electrode."""
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(n=8)
    coords[3] = np.nan
    before = anchor_to_surfaces(coords, {"lh": pair})
    after, report = coherent_anchors(
        before, coords, {"lh": pair}, {"lh": inflated},
        ["LTD"] * 8, ["seeg"] * 8,
    )
    assert report.contacts == 7
    assert np.array_equal(after.verts[3], before.verts[3])


def test_the_report_says_what_it_could_not_do():
    report = CoherenceReport(3, 40, 1.8, 5, 100)
    text = report.summary()
    assert "3 devices" in text and "40 of 100" in text
    assert "1.80 mm" in text
    assert "5 pairs still torn" in text


def test_the_anchor_shift_cap_is_what_keeps_a_contact_on_its_gyrus():
    """The second cap, and why the first one is not enough.

    ``max_fidelity_loss_mm`` bounds how much *further from the contact* a column
    may be. That turns out not to bound how far away it is **on the sheet**: two
    columns can each sit two millimetres from a contact and nine millimetres
    from each other, on opposite banks or at different depths of one sulcus.
    Measured on a real montage with only the fidelity cap in force, anchors moved
    up to 9.5 mm -- far enough to cross a gyral crown, which is the outcome this
    whole mechanism exists to prevent.

    So displacement is capped directly, and measured from the anchor ordinary
    anchoring chose rather than from this solver's own best candidate. The two
    differ, because the wider neighbourhood used here sometimes finds a genuinely
    closer column; a cap measured from that one would let a contact drift while
    reporting that it had not moved.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap()
    pia = np.asarray(pair.pia, dtype=np.float64)
    greedy = anchor_to_surfaces(coords, {"lh": pair})
    was = np.einsum("mij,mi->mj", pia[greedy.verts], greedy.weights)

    for cap in (0.5, 1.0, 3.0):
        solved = solve_device(
            coords, pair, inflated,
            incumbent=(greedy.verts, greedy.weights),
            max_anchor_shift_mm=cap,
            coherence_weight=1e6,          # push as hard as possible
        )
        now = np.einsum("mij,mi->mj", pia[solved.verts], solved.weights)
        assert np.linalg.norm(now - was, axis=1).max() <= cap + 1e-9


def test_the_incumbent_is_always_reachable():
    """A contact with no admissible alternative anchors exactly where it was.

    The incumbent is injected as a candidate rather than trusted to turn up in
    the search, which is what makes the cap a guarantee instead of a tendency:
    the solver can always fall back to the previous answer, so it is never forced
    onto a distant candidate merely because nothing nearer was enumerated.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(n=6)
    greedy = anchor_to_surfaces(coords, {"lh": pair})
    solved = solve_device(
        coords, pair, inflated,
        incumbent=(greedy.verts, greedy.weights),
        max_anchor_shift_mm=0.0,
        coherence_weight=1e6,
    )
    assert np.array_equal(np.sort(solved.verts, axis=1),
                          np.sort(greedy.verts, axis=1))


def test_derived_quantities_describe_the_column_that_was_chosen():
    """Depth, thickness and offset are statements about a column.

    Changing which column a contact is anchored to invalidates all three, and a
    stale depth would be read by the placement policy and by the viewer's depth
    window with nothing to say it no longer matched the anchor beside it.
    """
    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap()
    solved = solve_device(coords, pair, inflated)

    pia = np.asarray(pair.pia, dtype=np.float64)
    wm = np.asarray(pair.wm, dtype=np.float64)
    for i in range(len(coords)):
        tri = solved.verts[i]
        w = solved.weights[i]
        P = w @ pia[tri]
        W = w @ wm[tri]
        expected = np.dot(coords[i] - P, W - P) / np.dot(W - P, W - P)
        assert solved.depth[i] == pytest.approx(expected, abs=1e-9)
        assert solved.thickness_mm[i] == pytest.approx(
            float(np.linalg.norm(W - P)), abs=1e-9
        )


@pytest.mark.parametrize("anchor_mode", ["auto", "per_contact", "per_device"])
def test_the_frame_is_not_left_stale_by_re_anchoring(anchor_mode):
    """Re-anchoring invalidates the frame, on every path.

    The frame records a displacement *from the anchor triangle*, so changing
    which triangle a contact is anchored to invalidates it. ``regroup_anchors``
    happens to rebuild frames on most paths and returns early on
    ``per_contact``, which left a stale frame reconstructing positions 14 mm out
    -- silently, because a stale frame is a perfectly well-formed one and every
    other assertion still passed.

    The check is the exactness property: evaluated on the surface it was
    measured against, a frame reproduces the input coordinate. A stale one
    cannot.
    """
    from cortex.electrodes import ElectrodeSet

    pair, inflated, _ = two_banks()
    coords = shaft_in_the_gap(n=10)
    eset = ElectrodeSet(
        names=["LTD%d" % i for i in range(len(coords))],
        coords=coords,
        group=["LTD"] * len(coords),
        group_type=["seeg"] * len(coords),
    )
    eset.anchor(surfaces={"lh": pair}, inflated_surfaces={"lh": inflated},
                anchor_mode=anchor_mode)
    back = eset.anchors.evaluate({"lh": pair.pia}, offset="frame")
    assert np.abs(back - coords).max() < 1e-6
