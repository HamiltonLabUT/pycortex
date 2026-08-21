"""Which contacts get joined by a line.

The assertion that matters is a *count*: an n-by-n grid has exactly ``2n(n-1)``
edge neighbours, and a wrong lattice does not hit that number by accident. Too
many means the diagonals crept in -- the failure that makes a grid read as a
sheet of triangles rather than as a grid -- and too few means the lattice was
torn where the cortex stretched it.

The synthetic montages here need no subject; the one conformed grid at the end
needs S1's surfaces, and is what pins the count against a lattice that a folded
cortex has genuinely deformed rather than one drawn on graph paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from cortex.electrodes import group_edges
from cortex.electrodes._webgl import to_viewer_json

PITCH = 8.0


def flat_grid(n=8, pitch=PITCH):
    """An n-by-n lattice in a plane, and the names to go with it."""
    coords = np.array([[i * pitch, j * pitch, 0.0] for i in range(n) for j in range(n)])
    return coords, ["G%d" % (k + 1) for k in range(n * n)]


def lattice_truth(n=8):
    """The pairs a row-major n-by-n grid ought to produce."""
    edges = set()
    for i in range(n):
        for j in range(n):
            k = i * n + j
            if j + 1 < n:
                edges.add((k, k + 1))
            if i + 1 < n:
                edges.add((k, k + n))
    return edges


def shank(n=10, spacing=3.5):
    """A straight depth probe, contacts numbered from the tip."""
    coords = np.array([[0.0, 0.0, k * spacing] for k in range(n)])
    return coords, ["D%d" % (k + 1) for k in range(n)]


# -- grids ------------------------------------------------------------------


def test_a_grid_is_joined_along_its_rows_and_columns():
    coords, names = flat_grid()
    edges = group_edges(coords, ["G"] * 64, ["grid"] * 64, names=names)
    assert set(edges) == lattice_truth()


def test_a_grid_is_not_joined_across_its_diagonals():
    """The count above already implies this; this says what a failure means.

    A diagonal is ``sqrt(2)`` pitches long, so a lattice that admitted them
    would show up as edges half again as long as the rest.
    """
    coords, names = flat_grid()
    edges = group_edges(coords, ["G"] * 64, ["grid"] * 64, names=names)
    lengths = [np.linalg.norm(coords[a] - coords[b]) for a, b in edges]
    assert np.allclose(lengths, PITCH)


def test_a_grid_whose_pitch_varies_keeps_its_lattice():
    """A grid conformed to a folded surface has no single pitch.

    It is compressed inside a sulcus and stretched over a gyral crown, easily by
    a third, which is why the neighbour test compares one local distance against
    another instead of thresholding against a median. A threshold tears holes in
    the lattice exactly where the cortex bends most.
    """
    n = 8
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    coords = np.stack([
        u * PITCH + 3.0 * np.sin(u * 0.8),
        v * PITCH + 3.0 * np.cos(v * 0.6),
        6.0 * np.sin(u * 0.5 + v * 0.4),
    ], axis=-1).reshape(-1, 3)
    # The pitch really does vary, or this would be testing the flat case again.
    separation = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(separation, np.inf)
    nearest = separation.min(axis=1)
    assert nearest.max() / nearest.min() > 1.3

    edges = group_edges(coords, ["G"] * 64, ["grid"] * 64,
                        names=["G%d" % (k + 1) for k in range(64)])
    assert set(edges) == lattice_truth()


def test_a_grid_finds_its_own_width_rather_than_assuming_a_square():
    """Nothing records a grid's row and column count, so 4x16 has to be read
    off the positions as surely as 8x8 is."""
    rows, cols = 4, 16
    coords = np.array([[i * PITCH, j * PITCH, 0.0]
                       for i in range(rows) for j in range(cols)])
    names = ["G%d" % (k + 1) for k in range(rows * cols)]
    edges = group_edges(coords, ["G"] * 64, ["grid"] * 64, names=names)
    assert len(edges) == rows * (cols - 1) + cols * (rows - 1)
    assert np.allclose([np.linalg.norm(coords[a] - coords[b]) for a, b in edges], PITCH)


def test_a_grid_missing_a_contact_keeps_the_rest_of_its_lattice():
    """A contact the placement policy rejected leaves a hole, not a shear.

    Dropping the *first* one is the case that matters: the numbering then starts
    at 2, and a layout that cut its rows from the lowest number present would
    put every row boundary one column out.
    """
    coords, names = flat_grid()
    for dropped in (0, 29, 63):
        keep = [k for k in range(64) if k != dropped]
        edges = group_edges(coords[keep], ["G"] * 63, ["grid"] * 63,
                            names=[names[k] for k in keep])
        # Every edge of the full lattice that did not touch the missing contact.
        expected = {(keep.index(a), keep.index(b)) for a, b in lattice_truth()
                    if a != dropped and b != dropped}
        assert set(edges) == expected, "dropping G%d shifted the lattice" % (dropped + 1)


def test_a_grid_whose_names_carry_no_number_falls_back_to_its_geometry():
    """An explicit group column with names like "occipital" reads as a grid but
    says nothing about the order of its contacts. The positions still do."""
    coords, _ = flat_grid()
    edges = group_edges(coords, ["G"] * 64, ["grid"] * 64, names=["G"] * 64)
    assert set(edges) == lattice_truth()


def test_a_two_by_two_grid_is_a_square_and_not_a_tetrahedron():
    coords = np.array([[0.0, 0, 0], [PITCH, 0, 0], [0, PITCH, 0], [PITCH, PITCH, 0]])
    edges = group_edges(coords, ["G"] * 4, ["grid"] * 4, names=["G1", "G2", "G3", "G4"])
    assert set(edges) == {(0, 1), (0, 2), (1, 3), (2, 3)}


def test_a_grid_that_is_really_one_row_becomes_a_chain():
    """`group_type` is free-form, so a strip labelled "grid" has to still work."""
    coords = np.array([[k * PITCH, 0.0, 0.0] for k in range(6)])
    edges = group_edges(coords, ["G"] * 6, ["grid"] * 6, names=None)
    assert edges == [(k, k + 1) for k in range(5)]


# -- linear devices ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["depth", "seeg", "strip"])
def test_a_linear_device_is_chained_from_first_contact_to_last(kind):
    coords, names = shank()
    edges = group_edges(coords, ["D"] * 10, [kind] * 10, names=names)
    assert edges == [(k, k + 1) for k in range(9)]


def test_a_shank_is_chained_in_contact_order_not_array_order():
    """Sorting a montage's names as text puts D10 between D1 and D2.

    Chaining that array in the order it arrives draws a zigzag up and down the
    shank, which is a picture of the sort order rather than of the electrode.
    """
    coords, names = shank()
    shuffle = [3, 7, 0, 9, 1, 5, 2, 8, 4, 6]
    edges = group_edges(coords[shuffle], ["D"] * 10, ["depth"] * 10,
                        names=[names[i] for i in shuffle])
    # Back through the shuffle, the chain has to be the consecutive one.
    original = sorted(tuple(sorted((shuffle[a], shuffle[b]))) for a, b in edges)
    assert original == [(k, k + 1) for k in range(9)]


def test_an_unnumbered_shank_falls_back_to_array_order():
    coords, _ = shank(n=4)
    edges = group_edges(coords, ["D"] * 4, ["depth"] * 4,
                        names=["tip", "mid", "upper", "top"])
    assert edges == [(0, 1), (1, 2), (2, 3)]


# -- what never happens -----------------------------------------------------


def test_edges_never_cross_a_group():
    coords, names = flat_grid(n=3)
    both = np.vstack([coords, coords + 500.0])
    edges = group_edges(both, ["A"] * 9 + ["B"] * 9, ["grid"] * 18, names=names * 2)
    assert all((a < 9) == (b < 9) for a, b in edges)
    assert len(edges) == 2 * len(lattice_truth(n=3))


def test_a_group_of_one_is_joined_to_nothing():
    assert group_edges(np.zeros((1, 3)), ["A"], ["grid"], names=["A1"]) == []


def test_a_contact_with_no_coordinate_takes_part_in_no_edge():
    """It is not drawn, so there is nowhere for a line to reach."""
    coords = np.array([[0.0, 0, 0], [PITCH, 0, 0], [np.nan, np.nan, np.nan]])
    assert group_edges(coords, ["A"] * 3, ["depth"] * 3,
                       names=["A1", "A2", "A3"]) == [(0, 1)]


def test_every_edge_names_the_lower_index_first():
    coords, names = flat_grid(n=4)
    edges = group_edges(coords, ["G"] * 16, ["grid"] * 16, names=names)
    assert all(a < b for a, b in edges)
    assert edges == sorted(set(edges))


# -- and what the viewer is handed ------------------------------------------


def test_the_payload_carries_the_edges():
    from cortex.tests.test_electrodes_webgl import build_grid

    eset = build_grid()
    payload = to_viewer_json(eset)
    assert payload["connections"] is True
    assert payload["line_color"] == payload["color"]
    # A conformed 8x8 grid, so this is the lattice count on a real cortex.
    assert len(payload["edges"]) == 2 * 8 * 7
    assert all(0 <= a < len(payload["electrodes"]) and 0 <= b < len(payload["electrodes"])
               for a, b in payload["edges"])


def test_connections_can_be_turned_off_before_the_page_is_written():
    from cortex.tests.test_electrodes_webgl import build_grid

    payload = to_viewer_json(build_grid(), connections=False)
    assert payload["edges"] == []
    assert payload["connections"] is False


def test_the_line_colour_can_differ_from_the_marker_colour():
    from cortex.tests.test_electrodes_webgl import build_grid

    payload = to_viewer_json(build_grid(), line_color="#3366ff")
    assert payload["line_color"] == "#3366ff"


def test_edges_index_the_contacts_that_were_sent():
    """Not the set they came from.

    `to_viewer_json` drops the contacts the placement policy rejected, so an
    edge list numbered against the original set would join the wrong markers --
    and silently, since the indices stay in range.
    """
    from cortex.tests.test_electrodes_webgl import build_grid

    eset = build_grid()
    payload = to_viewer_json(eset)
    names = [c["name"] for c in payload["electrodes"]]
    joined = {(names[a], names[b]) for a, b in payload["edges"]}
    # G1 is a corner of the grid, so it has exactly two neighbours, and they are
    # the next contact along its row and the one in the next column.
    assert {b for a, b in joined if a == "G1"} | {a for a, b in joined if b == "G1"} == {
        "G2", "G9"
    }
