import hashlib

import numpy as np

from cortex.dataset.braindata import _hash


def test_hash_uses_tobytes():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = hashlib.sha1(array.tobytes()).hexdigest()
    assert _hash(array) == expected


def test_raw_carries_the_views_attrs():
    """`.raw` used to forward `priority` alone, so every other key stored beside
    it -- `rate`, `stim`, and anything a space adds later -- was dropped on the
    way to the RGB view. Forwarding the mapping is a strict generalisation:
    `priority` is one of its keys."""
    import cortex

    nverts = cortex.db.get_surf("S1", "fiducial", merge=True)[0].shape[0]
    view = cortex.Vertex(np.random.randn(nverts), "S1", priority=3,
                         attrs={"rate": 2.0})
    raw = view.raw
    assert raw.priority == 3
    assert raw.attrs["rate"] == 2.0
