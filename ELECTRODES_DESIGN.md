# Electrodes in pycortex — scope and design decisions

Companion to the iEEG design document. Written against `types-data-protocols`
(`d2fa551d`). Section 1 records what reading the branch changed about the plan;
section 2 is the Scope the design document asks for; section 3 lists the
decisions that are still open.

## 1. What the codebase already gives us

Five findings, in descending order of how much work they save.

### 1.1 The webgl viewer already pins 3-D objects to morphing surfaces

`mriview.get_position(posdata, surfmix, thickmix, idx)`
(`cortex/webgl/resources/js/mriview_utils.js:140`) returns the position **and**
normal of a vertex under the current inflation (`surfmix`) and cortical-depth
(`thickmix`) state. Two consumers already use it for exactly the problem the
design document poses:

- `facepick.js:340` — `setMix` walks its markers and re-copies each one's
  position on every `mix` event, which is how the picker's 3-D axes stay stuck to
  a vertex while the brain inflates.
- `svgoverlay.js:182` — the same thing for ROI labels.

So electrode markers do **not** need shader work, a new webgl payload encoding,
or the "own attribute set and own draw call" that
`cortex/dataset/ADDING_A_SPACE.md` warns about. They need a `THREE.Group` of
instanced meshes and a `mix` listener. That is the single largest reduction in
scope available.

The one gap: `get_position` takes a scalar vertex index. Barycentric anchoring
needs it to take three indices and three weights. That is a ~15-line sibling
function, `get_position_bary`, in the same module.

### 1.2 Adding markers to the hemisphere pivots gets the rest of the geometry free

`mriview_surface.js:262-298` builds a `pivots[hemi].{front,back}` pair per
hemisphere and adds both the picker markers and the SVG labels to
`pivots[hemi].back`. Those groups carry the hemisphere separation
(`setLeftVis`/`shift`), the "open the book" pivot rotation, and the
`rotation.x = clipped * -PI/2` flip applied when the surface goes fully flat
(`mriview_surface.js:505`). Electrode markers parented there inherit all of it.
Anything positioned in world space instead would have to reimplement three
transforms and would drift.

### 1.3 quickflat needs no cache work

`_make_flatmask` (`cortex/quickflat/utils.py:381`) returns
`extents = [xmin, xmax, ymin, ymax]` in flat-surface coordinates, and the flatmap
is drawn with that extent. So an electrode's flat position — read straight out of
`db.get_surf(subject, "flat", merge=True, nudge=True)` — is already a matplotlib
*data* coordinate. `add_electrodes` is an ordinary compositing decorator next to
`add_rois`, roughly forty lines, with no new cache and no change to
`make_flatmap_image`.

### 1.4 TkRegRAS is very likely already the surface coordinate system

`db.get_surf` reads the imported FreeSurfer geometry through `formats.read`, and
the import path (`cortex/freesurfer.py:490`, `parse_surf`) keeps FreeSurfer's own
surface RAS without applying `c_ras`. Surface RAS *is* TkRegRAS, so electrode
coordinates should drop straight onto `fiducial`/`pia` with no transform.

Two caveats that must be enforced in code rather than assumed:

- **Verify, don't trust** — but know what verification can reach. `check_alignment`
  reports each electrode's distance to the pial surface and how much of that
  distance is a single common direction. Measured on S1 with a synthetic 64-contact
  grid: correct placement gives a 1.6 mm median offset, a 15 mm error gives 12.3 mm
  and a 40 mm error gives 33 mm, so a shift that lifts electrodes *off* the sheet is
  caught easily. A shift *along* the sheet is not caught at all and cannot be — an
  8 mm tangential error leaves the median offset at 1.0 mm, because the grid slides
  onto a neighbouring gyrus and sits just as snugly. Only the anatomical labels can
  see that one. Two further caveats worth stating in the API rather than
  discovering later: cortex folds, so a *scattered* set is far more forgiving than
  a contiguous one — judge a montage by its grids, not its outliers — and a montage
  that is mostly sEEG will report a large median offset and read as suspicious when
  it is fine, since depth contacts are legitimately centimetres from the pia.
- **Never anchor on a nudged surface.** `db.get_surf(..., nudge=True)` shifts
  non-fiducial hemispheres in x (`cortex/database.py:554`). Anchoring happens on
  `pia`/`fiducial` with `nudge=False`; nudging is a display-time concern.

### 1.5 The 2x3 view grid fits, but the design document's module is wrong

`Volume` and `Vertex` live in `cortex/dataset/views.py`, not `cortex.database` —
`cortex.database` is the filestore. The new classes belong in `cortex.dataset`,
exported as `cortex.Electrode` alongside `cortex.Volume` and `cortex.Vertex`.

The grid extends cleanly: `n=1` data is `Electrode`, the "two encoding models you
wish to compare" case is `Electrode2D`, and `ElectrodeRGB` completes the row.
The STRF / evoked case (`[n_channels x features x lags]`) does **not** fit — it
is not a colormappable array over the geometry. It belongs on the electrode set
as auxiliary per-channel data feeding the billboards, not in the space.

## 2. Scope

### 2.1 The central structural decision: two objects, not one

- **`ElectrodeSet`** — geometry and metadata. Names, TkRegRAS coordinates,
  group, group type, size, anatomy, status, plus the derived surface anchors.
  Subject-level and data-free, the way `svgoverlay` holds ROIs. This is what
  `add_electrodes` and the webgl viewer consume.
- **`Electrode` / `Electrode2D` / `ElectrodeRGB`** — `n_channels`-long data
  arrays over an `ElectrodeSpace`, colormapped, HDF-serialisable, movie-capable.
  Exactly parallel to `Vertex`, and they get colormapping, `.raw`, `uniques`,
  alpha, arithmetic and HDF from the three abstract columns for free.

`ElectrodeSpace.spec_keys = ("electrodes",)` names the set, mirroring
`VolumeSpace`'s `("xfmname",)`, so `coerce` can check `len(data) == n_electrodes`
and `from_spec` can rebuild the space from HDF.

Splitting them is what keeps a metadata-only workflow (draw the grid, no values)
and a data workflow (colour by encoding-model correlation) from contaminating
each other, and it keeps STRFs out of the space.

### 2.2 Anchoring

The anchor is found against the **pia-to-white-matter slab**, not against a
single surface: the search runs on the mid-surface `(pia + wm) / 2`, which is
the least biased place to look for the column an electrode belongs to, and the
barycentric weights and depth then follow from that triangle's pial and
white-matter versions. Anchoring against one surface alone would make the depth
coordinate in 2.3 unrecoverable, and searching on the pia alone pulls
superficial contacts onto the near bank of a sulcus.

Store, per electrode, computed once:

```
hemi, verts[3], bary_w[3], depth, offset_mm, placement
```

where `depth` is the normalised pia-to-white-matter coordinate defined in 2.3
and `offset_mm` is the perpendicular distance from the electrode to the column
it was assigned to.

**Vertex indices, not a face index.** This looks like a detail and is not.
A face index is *not* shared between a subject's surfaces: flattening cuts the
medial wall away, so on S1 the pial surface has 305,782 triangles and the flat
surface 291,351, and face `k` is a different triangle on each. Storing a face
index evaluates silently and wrongly on the flat surface -- it was written that
way first, and the test that caught it is
`test_a_flat_position_is_the_flat_coordinate_of_that_vertex`. Vertex indexing is
shared by every surface, which is the only thing that makes an anchor portable.

Position on any surface is then `bary_w @ pts[verts]`, needing no polygons at
all. Barycentric rather than nearest-vertex because contacts are 3-10 mm apart
while vertices are ~0.5-1 mm apart: nearest-vertex snapping visibly distorts
within-grid spacing, and barycentric costs three extra floats.

**`placement` is an explicit enum on every electrode**, never a silent drop:
`on_surface`, `projected`, `too_far`, `unknown_anatomy`. The design document's
rule ("if anatomy is Unknown, drop") keys on an *optional* metadata string; with
the stated abnormal anatomy that will drop contacts that are merely unlabelled.
The policy object should therefore be geometric and thresholded, with the
anatomy rule as one configurable term, and every excluded electrode returned in a
report rather than vanishing.

### 2.3 Cortical depth is the third anchor coordinate

Depth is not a visibility flag, it is a coordinate, and pycortex already has a
control for it. The `thickmix` uniform interpolates every vertex between pia
(`thickmix = 0`) and white matter (`thickmix = 1`) — exposed in the viewer as the
`depth` slider (`mriview_surface.js:90`) and handled inside `get_position`
(`mriview_utils.js:156-172`). So each electrode carries a **normalised cortical
depth** alongside its barycentric anchor, and the depth slider selects which
electrodes are drawn.

Computing it, at the anchor: let `P` be the barycentric-interpolated pial point
and `W` the same point on the white-matter surface. Then

```
d = dot(e - P, W - P) / |W - P|**2
```

with `d = 0` on pia, `d = 1` at the white-matter boundary, `0 < d < 1` inside grey
matter, `d < 0` outside the brain, and `d > 1` past the white-matter surface.

Three consequences:

- **An electrode's position tracks `thickmix` for free.** `get_position` already
  interpolates pia-to-wm; the barycentric variant inherits it. The marker moves
  with the surface under both inflation and depth.
- **Visibility is a function of `|d - thickmix|`.** Fade opacity across a
  tolerance band rather than switching — a hard cutoff pops as the slider moves,
  and the fade also communicates *how far* off-depth a contact is. An ECoG grid
  (`d` near 0) is fully visible in the default view; sliding toward white matter
  brings the deeper contacts up as the surface ones fade.
- **The common ECoG case lands correctly with no special-casing.** Grid and strip
  contacts sit slightly *above* pia (dura, contact thickness), giving small
  negative `d`, which clamps to 0 and shows in the default view.

The case this does not cover: `d > 1`, a deep sEEG contact past the white-matter
surface — hippocampus, amygdala, insula, deep white matter. `thickmix` only spans
0 to 1, so such a contact has no slider position of its own. Widening the slider's
range would change what an existing pycortex control means for every other user,
so instead clamp `d` to 1, draw those contacts with a visually distinct marker
(outlined rather than filled), and report the true depth in millimetres in the
click-through metadata panel. This keeps them visible and honestly labelled
without asserting a surface position they do not have.

`coords_tkras` remains the truth throughout; `d` is derived and cached with the
anchor.

On the "invertible/lossless" constraint: the anchor satisfies it, the projection
does not, and the distinction should be visible in the API. `coords_tkras` is the
truth and is never mutated; anchors are a derived cache carrying a checksum of the
surface they were computed against. Recovering 20 mm of depth from a surface
position is not possible and no amount of care makes it so.

### 2.4 New files

| Path | Contents |
| --- | --- |
| `cortex/electrodes/__init__.py` | Public API: `ElectrodeSet`, `load_electrodes`, `infer_group`, `PlacementPolicy` |
| `cortex/electrodes/_set.py` | `ElectrodeSet`, per-channel metadata, `select()`/filtering, group inference from channel names |
| `cortex/electrodes/_anchor.py` | TkRegRAS -> anchor; `evaluate(anchors, surface_type)` -> positions; the coordinate-sanity check from 1.4 |
| `cortex/electrodes/_io.py` | BIDS `*_electrodes.tsv` reader/writer plus the filestore JSON form |
| `cortex/dataset/_electrode_space.py` | `ElectrodeSpace(BrainSpace)` |
| `cortex/dataset/electrode_views.py` | `ElectrodeView` mixin + `Electrode`, `Electrode2D`, `ElectrodeRGB` |
| `cortex/webgl/resources/js/electrodes.js` | `electrodes.ElectrodeSet`: instanced markers, `mix` handler, raycast hover/click, billboard tooltip, metadata panel, menu folder |
| `cortex/tests/test_electrodes.py` | `ElectrodeSet`, group inference, IO round-trip |
| `cortex/tests/test_electrode_anchor.py` | Anchoring maths against a synthetic surface; placement policy; every `placement` outcome |
| `cortex/tests/test_electrode_space.py` | The space + three views, modelled on `test_new_space.py` |
| `cortex/tests/test_electrode_webgl.py` | Headless viewer, following `test_webgl_headless.py` |
| `examples/electrodes/plot_electrodes_flatmap.py` | Flatmap example |
| `examples/electrodes/plot_electrodes_webgl.py` | Viewer example |
| `docs/electrodes.rst` | User guide |

The design document's `cortex.database.Electrode` / `cortex.database.ElectrodeData`
naming is replaced by `cortex.Electrode` (the data view, from `cortex.dataset`)
and `cortex.electrodes.ElectrodeSet` (the geometry), per 1.5.

BIDS `*_electrodes.tsv` is worth adopting as the interchange format: its
`name / x / y / z / size / group / hemisphere / type / status` columns map almost
one-to-one onto the document's required and optional fields, and it means other
labs' files load without a bespoke parser.

### 2.5 Modified files

**Data model**
- `cortex/dataset/__init__.py` — export the three views and the space; extend the
  2x3 grid table in the module docstring to 3x3.
- `cortex/dataset/views.py` — an electrode branch in `normalize()`'s overloads.
  `_detect_space` needs nothing: it walks `registered_spaces()`, and
  `ElectrodeSpace` registers ahead of the `SurfaceSpace` fallback automatically
  as long as it writes and matches its own HDF discriminator.
- `cortex/__init__.py` — export `Electrode`, `Electrode2D`, `ElectrodeRGB`, and
  the `electrodes` module.
- `cortex/database.py` — `get_paths` gains an `electrodes` entry
  (`<filestore>/<subject>/electrodes/<name>.json`); `get_electrodes` /
  `save_electrodes` on `Database`; a `SubjectDB` attribute so
  `cortex.db.S1.electrodes` works like `.surfs`.

**quickflat**
- `cortex/quickflat/composite.py` — `add_electrodes(fig, electrodes, values=None,
  marker=..., size=..., cmap=..., vmin=..., vmax=..., filter=...)`.
- `cortex/quickflat/__init__.py` — export it.
- `cortex/quickflat/view.py` — `with_electrodes` on `make_figure`, threaded
  through `make_png` and `make_svg`, following the existing `with_rois` pattern.

**webgl**
- `cortex/webgl/resources/js/mriview_utils.js` — add `get_position_bary`.
- `cortex/webgl/resources/js/mriview_surface.js` — construct the electrode set
  beside `this.picker` and `this.svg` (~lines 279-300), parent it into
  `pivots[hemi].back`, wire the `mix` event, add its menu folder.
- `cortex/webgl/resources/js/mriview.js` — top-level UI folder, and the
  bottom-left metadata panel div.
- `cortex/webgl/view.py` — `electrodes=` on `show` and `make_static`;
  serialise set, anchors and per-channel values into the page.
- `cortex/webgl/data.py` — emit the electrode value payload if values ride with
  the active dataset.
- `cortex/webgl/template.html` — one more `<script>` tag. `htmlembed.py` walks
  script tags generically, so static export needs no change.

**docs**
- `docs/api_reference_flat.rst` — add the three classes to the autosummary at
  lines 65-70.
- `cortex/dataset/ADDING_A_SPACE.md` — a short postscript. This is the first
  space added through that document, and it exercises a claim the document makes
  ("geometry that is not per-vertex needs its own attribute set and its own draw
  call") that turns out to be avoidable via the marker path; worth recording.

### 2.6 Phasing

Each phase is independently verifiable and independently useful.

- **P0** — *done.* `ElectrodeSet`, barycentric anchoring, the depth coordinate,
  BIDS and JSON IO, placement policy, coordinate check, filestore round-trip.
  Pure Python, no rendering. 95 tests: `test_electrode_anchor.py` (synthetic
  two-plane cortex, so the arithmetic is checkable by hand),
  `test_electrodes.py` (the set, selection, both file formats) and
  `test_electrodes_subject.py` (the real S1 surfaces and a scratch filestore).
- **P1** — `add_electrodes` for quickflat. First visible output, no JavaScript.
- **P2** — `ElectrodeSpace` and the three views: colormapping, HDF, movies.
- **P3** — webgl markers, `mix` tracking, shapes, sizes, filtering, hover and
  click.
- **P4** — billboards for STRF and evoked data.

## 3. Decisions taken

1. **Depth drives which electrodes are drawn.** Electrodes anchor between the
   pial and white-matter surfaces and carry a normalised depth; the existing
   `thickmix` / `depth` control selects them. Written up in 2.3. This replaces an
   earlier proposal to hide depth contacts by default, which threw away
   information the viewer can already express.
2. **An `ElectrodeSet` lives in the filestore**, at
   `<filestore>/<subject>/electrodes/<name>.json`, so `cortex.db.S1.electrodes`
   resolves like `.surfs` and the viewer finds it without being handed it.
3. **Marker colours ship as values, not RGBA.** The marker shader samples the
   colormap texture the viewer already loads, so `vmin`/`vmax` sliders and movie
   scrubbing recolour electrodes exactly as they recolour `Vertex` data. This is
   the one place electrodes touch shader code, and it lands in P3.

## 4. Still to settle

- **quickflat's depth argument.** `make_figure` already takes `depth` for
  volumetric sampling. `add_electrodes` needs its own depth selection, and
  whether the two should share a value or stay independent is a usability call
  better made once P1 is drawable. Default proposed: `depth=None` draws every
  electrode, a float selects a band.
- **Marker shape vocabulary.** The design document asks for a shape dropdown per
  electrode type. The set of shapes, and whether shape is driven by `group_type`
  by default, is worth fixing before P3 so the instanced-mesh code builds the
  right number of geometries.
