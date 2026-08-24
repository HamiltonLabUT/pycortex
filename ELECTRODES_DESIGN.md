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

### 1.4 TkRegRAS is *not* the surface coordinate system — this was wrong

**Corrected after it cost real data.** This section originally read "TkRegRAS is
very likely already the surface coordinate system", on the grounds that
`db.get_surf` reads the imported geometry through `formats.read` and that
`parse_surf` (`cortex/freesurfer.py:411`) keeps FreeSurfer's surface RAS without
applying `c_ras`. That is true of `parse_surf` and irrelevant: `parse_surf` is
the *direct FreeSurfer reader*, not the path that fills the filestore.

The path that fills the filestore is `import_subj`, and it converts every
surface with **`mris_convert --to-scanner`** (`cortex/freesurfer.py:243`),
deliberately, so the surfaces share a frame with the volume data the transform
machinery works in. `--to-scanner` adds `c_ras`. So a subject imported the
normal way has **scanner-RAS** surfaces, and TkRegRAS electrode coordinates
dropped straight onto them land `|c_ras|` away.

Measured on two real subjects in this filestore, both imported by `import_subj`:

| subject | GIFTI dataspace | `c_ras` | error |
| --- | --- | --- | --- |
| S0019_complete | 1 (`SCANNER_ANAT`) | (0.98, −4.55, 0.61) | 4.70 mm |
| TCH06_complete | 1 (`SCANNER_ANAT`) | (1.84, 1.92, −2.03) | 3.35 mm |

Every contact in both montages anchored to the wrong triangle. No hemisphere
was misassigned and no contact left the brain, which is exactly why it survived
review: the error moves a contact to the neighbouring gyrus and stops.

The bundled `S1` declares dataspace 0 and was not converted this way, so it
happens to be exempt — which is worth knowing, because S1 is what the test
suite runs on and it therefore cannot catch this class of bug at all.

**Nothing geometric detects it.** Both natural checks are blind here, for the
same reason: cortex folds densely enough that a displaced contact finds another
bank to sit against. `check_alignment` reported a 0.50 mm systematic shift for
TCH06's genuinely-3.35 mm-wrong montage, and distance-to-nearest-surface could
not separate the shifted montage from the correct one either. Only the surface
file's own provenance — the GIFTI `dataspace` and `VolGeomC_R/A/S` written by
`mris_convert` — says which space the geometry is in.

`surface_space_offset` (`cortex/electrodes/_set.py`) reads exactly that, and is
applied at anchor time and when serialising for the viewer. It raises rather
than guesses when a surface declares scanner space without recording the
`c_ras` needed to undo it: the alternative is another few silent millimetres.

Three caveats that must be enforced in code rather than assumed:

- **Read the provenance, don't reason about the reader.** The mistake above was
  reasoning from `parse_surf`'s behaviour to the filestore's contents, with a
  `mris_convert` call in between. The GIFTI records what was actually run.

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
  It also excludes rows with no finite coordinate and says how many — a real
  montage had 14 of 100, and before that they propagated: one NaN row turned
  every figure in the report into NaN, so the check silently stopped checking
  on exactly the files most likely to need it.
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
hemi, verts[3], bary_w[3], depth, offset_mm, dist_pia_mm, dist_wm_mm, placement
```

where `depth` is the normalised pia-to-white-matter coordinate defined in 2.3,
`offset_mm` is the perpendicular distance from the electrode to the column it
was assigned to, and `dist_pia_mm` / `dist_wm_mm` are the distances to the
nearest point on each bounding surface — measured against the surfaces
themselves over the candidate faces, not derived from `depth` and `offset_mm`,
which would only ever describe this electrode's own column.

**Selection is by distance to the cortical column, not to the mid-surface.**
The candidate triangles come from the mid-surface, but choosing among them by
mid-surface distance biases depth toward the middle of the ribbon: a contact
sitting on the pia over a curved patch finds a nearer mid-surface point on a
*neighbouring* column. Measured on S1 over 160 contacts at known depths, that
gives a mean depth error of 0.073 and a worst case of 0.38 — enough that a
contact placed on the pia reports depth 0.37. Selecting instead by distance to
the clamped pia-to-wm segment costs nothing (the quantities are already
computed) and gives 0.006 and 0.17. Since depth is the axis the viewer's slider
rides on, that error matters more than its size suggests. Pinned by
`test_depth_is_recovered_across_the_ribbon`.

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

**How far from cortex is too far: 4 mm to the nearest bounding surface.**
`placement` is `too_far` when an electrode is further than
`PlacementPolicy.max_surface_distance_mm` from the nearest point on *either*
bounding surface, pial or white matter. Everything else the policy can bound —
offset from the assigned column, height above the pia, depth past the white
matter — is off unless asked for.

This replaced a 10 mm bound on the column offset that was never an anatomical
number. It came from `check_alignment(threshold_mm=10.0)`, where 10 mm *is*
calibrated but answers a different question: a correctly placed grid gives a
1.6 mm median offset and a 15 mm coordinate-space error gives 12.3 mm, so 10 mm
separates "plausible" from "wrong space". Reused per contact it was a
registration backstop standing in for a criterion nobody had written.

Measured on S1, distance to the nearer bounding surface:

| contacts | p50 | p90 | max |
| --- | --- | --- | --- |
| subdural grid, 1.5 mm above the pia | 1.50 | 1.50 | 1.56 |
| sEEG shaft, 0–56 mm inward | 0.88 | 1.66 | 2.25 |

The second row is why one number covers both ends of the montage vocabulary,
and it is not obvious: a shaft driven nearly six centimetres in stays under
2.3 mm from cortex, because it is threading folds the whole way. So bounding
*both* directions does not cost the depth electrodes anything. What 4 mm
excludes is the deep white-matter core, a ventricle, and the outside of the
head. Pinned by `test_a_resting_grid_and_a_deep_shaft_both_clear_four_millimetres`.

The same folding is why this is **not** a registration check. A contiguous grid
lifted 15 mm fails every contact; a scattered montage shifted the same 15 mm
keeps more than half, each stray point finding some bank to land near. That is
`AlignmentReport`'s caveat restated — judge a montage by its grids, not its
outliers — and `test_the_rule_catches_a_lifted_grid_but_not_scattered_contacts`
holds both halves in place.

The threshold is reachable where the decision is visible: `add_electrodes(...,
max_surface_distance_mm=...)` re-decides it for one figure without touching the
set's stored `placement`, so a drawing never changes what a later viewer shows.

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
- **Visibility is *not* a function of `|d - thickmix|`** — this was tried and
  reverted. Making the slider gate visibility means the slider's own default,
  mid-ribbon, decides what a montage looks like on load, and mid-ribbon is the
  one depth a subdural contact can never be at. On a real montage it hid 53 of
  143 placeable contacts, all 35 of the outside-the-pia ones among them,
  silently. Anchoring the window to the pia instead only moved the problem:
  it then asks the same question as `PlacementPolicy.max_surface_distance_mm`
  and answers it more strictly, so a contact that passed the 4 mm projection
  gate was hidden by a 2 mm window anyway — two gates for one question, and the
  stricter one invisible and unconfigurable from python.

  So the depth window is **off by default** and the placement policy alone
  decides what is drawn. The window survives as an exploration control: set it,
  turn on `depth_follows_slider`, and sweeping the slider walks the montage
  through the ribbon. That is what it is good for, and it is opt-in because
  wanting it is a deliberate act rather than the common case.
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
- **P1** — *done.* `add_electrodes` for quickflat, `with_electrodes` on
  `make_figure` (and so on `make_png`/`make_svg`, which forward `**kwargs`), and
  `examples/electrodes/plot_electrodes_flatmap.py`. 24 tests in
  `test_quickflat_electrodes.py`, asserting on the marker collections rather
  than on pixels.

  Two decisions worth recording. Electrode depth is a **separate argument from
  `make_figure`'s existing `depth`**: that one is the volumetric *sampling*
  depth for `add_data`, and someone sampling their fMRI at `depth=0.5` has not
  thereby asked to hide all but mid-ribbon contacts. And electrodes are drawn
  **after `add_cutout`**, because they are scatter markers rather than image
  layers: `add_cutout` iterates its `layers` calling `get_array()` and would
  choke on a `PathCollection`. Drawing them afterwards means the cutout's reset
  axis limits clip them instead, which is right except for a marker inside the
  cutout's bounding box and outside its outline.

  Marker shape follows group type by default — circle for a grid, square for a
  strip, diamond for seeg and depth — so a reader can tell which positions to
  trust without consulting a legend. Marker *size* is constant in figure units
  and deliberately not scaled by areal distortion, per 2.2.
- **P2** — `ElectrodeSpace` and the three views: colormapping, HDF, movies.
- **P3a** — *done.* Markers in the viewer: one mesh per contact parented into
  `pivots[hemi].back`, re-placed on every `mix` event via a barycentric sibling
  of `get_position`, shape by group type, one flat colour, and a menu folder for
  visibility, radius and stand-off. No shader changes, no new payload format —
  the anchors ride inside `viewopts`.
- **P3b** — colours from data, hover tooltip, click-through metadata panel,
  filtering by group and anatomy.
- **P3c** — *done.* Connection lines: a thin segment between contacts that are
  neighbours on the same device, with a `connections` checkbox in the electrodes
  folder, on by default. Not in the original phasing; it belongs beside P3b's
  "filtering by group", since both are about making a montage read as devices
  rather than as a cloud of dots.

  Which pairs are neighbours is decided in Python
  (`cortex/electrodes/_connect.py`) from the measured coordinates, once, and
  sent as an index list. Deriving it in the browser from the drawn positions
  would make the topology depend on the inflation slider, with edges appearing
  and disappearing mid-morph, when the fact being drawn is a property of the
  physical device. See 6 below for how a grid's lattice is recovered.
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

## 4. What a real montage changed

Reading one real clinical montage -- an `img_pipe`-style `TDT_elecs_all.mat`,
366 rows -- broke three things and improved a fourth. Recorded here because
none of them were guessable from the data specification.

**Placeholder rows exist, and they are not electrodes.** Thirty-six rows carried
non-finite coordinates, names `NaN1`-`NaN32`, device type `NaN` and blank
anatomy: unconnected amplifier channels. A NaN reaching `cKDTree.query`
corrupts the whole result rather than failing, so non-finite rows are now held
back from the search and marked `no_coordinate`.

**Those names collide.** `NaN1`-`NaN4` each appeared twice, because a second
disconnected block restarted the numbering. `ElectrodeSet` requires unique names
-- everything downstream identifies a contact by name -- so the reader drops
coordinate-less rows by default, which removes every duplicate along with them.
Keeping them (`drop_missing=False`, for when row indices must line up with a
data array over all original channels) still refuses such a file, which is
documented rather than fixed.

**Key on the coordinate, not the name.** In that file the name test and the
coordinate test selected exactly the same 36 rows. `NaN1` is one lab's spelling
of "no electrode here"; a non-finite coordinate is everyone's.

**The anatomy column is the one thing geometry cannot replace.** A single column
carried four vocabularies at once -- Desikan-Killiany parcels
(`superiortemporal`), Destrieux parcels (`ctx_lh_G_temporal_inf`), FreeSurfer
aseg structures (`Left-Hippocampus`, `Left-Putamen`, `Left-Cerebral-White-Matter`)
-- plus the literal `Unknown` and blanks. The aseg entries answer the question
raised in 2.3 and visible in the P0 preview: which contacts have no cortex to be
on. Geometry cannot tell, because a contact in white matter still has cortex a
millimetre away on some sulcal bank; the label can. Hence a fifth placement
outcome, `non_cortical`, on by default. It marks rather than excludes -- where a
hippocampal contact projects to is usually exactly what a reader wants shown --
and it is a far better reading of the design document's "if anatomy is Unknown,
drop" than a literal one, which would have caught three contacts in that montage
and missed the eleven labelled as white matter or subcortical.

Labels are stored verbatim. Normalising the vocabularies would discard the
distinction that makes them useful.

## 5. Vertex order is shuffled twice, and both trips are mandatory

The one thing in P3a that was not guessable, and the reason its test asserts a
distance rather than inspecting a screenshot.

A pycortex vertex index does not name the same vertex in the browser's buffers.
Two separate permutations sit between them:

1. **The CTM's spatial sort.** The viewer packs with OpenCTM's `mg2`, which
   reorders vertices for compression. `brainctm` saves the permutation beside
   the `.ctm` as an `.npz`, and per-vertex *data* is already permuted through it
   in python by `Package.reorder` before being sent.
2. **Three.js's chunk-local shuffle**, applied on load so each draw chunk fits
   inside a 16-bit index. That one is `indexMap` / `reverseIndexMap`, and it is
   local: on S1 it moves vertex 50000 to 51205, while the CTM sort moves it to
   123806.

So the full correspondence is `buffer[i] ↔ pycortex[index[reverseIndexMap[i]]]`.
Electrodes make the first trip in python — `to_viewer_json` converts through the
`.npz` — and the second in JavaScript, which is exactly what `svgoverlay` does,
and why it needs only `map[idx]`: its `data-ptidx` values are *already* in CTM
order.

**Making only one of the two trips is not a visible failure.** The markers still
land on the surface, still track inflation and flattening, still render without
a single JavaScript error, and photograph like a finished feature. They are
simply on the wrong vertices — scattered across the hemisphere instead of
clustered. Three separate candidate mappings were tried against the screenshots
before anyone measured anything, and all three looked equally plausible.

What settled it was a number with an independently known answer: the contacts of
an 8×8 grid at 8 mm pitch sit about 7 mm apart on cortex. Wrong indices gave
51 mm. `test_electrodes_webgl.py` asserts that spacing against the value python
computes for the same set, and separately asserts that the CTM permutation is
not the identity — because if it ever became one, sending unconverted indices
would start working by accident and the guard would rot.

## 6. A grid's lattice comes from its numbering, not from its spacing

Connection lines need to know which contacts are neighbours, and for a linear
device that is just the order along the shank. A grid is the hard case, because
**no format the readers support records a grid's row and column count**. Neither
BIDS' `electrodes.tsv` nor img_pipe's `elecmatrix` carries one, so a grid
arrives as nothing more than a set of positions sharing a group name.

The obvious answer -- join contacts closer together than some multiple of the
median spacing -- does not survive contact with a real montage. A grid conformed
to a folded surface has no single pitch: it is compressed inside a sulcus and
stretched over a gyral crown. On the 8x8 test grid draped over S1's lateral
temporal cortex, the distance to a contact's nearest neighbour ranges from 1.1
to 9.8 mm, an order of magnitude, against a nominal 8 mm pitch. Any threshold
that admits the stretched pairs also admits diagonals somewhere else, and a
tuned one tears holes in the lattice exactly where the cortex bends most --
measured at 16 of 112 edges missing on that grid.

What does work is remembering that a grid is a **rigid rectangular sheet whose
contacts are numbered along its rows**. Its lattice is therefore a fact about
the numbering, and the only unknown is the row width. Recovering one number from
the geometry is a far steadier question than recovering the whole lattice from
it, because the candidate answers are few and far apart: the wrong width steps
sideways along a row instead of down a column, and lands multiples of the pitch
away. `_connect.py` asks it in two parts --

- the **width**, from the contacts numbered *width* apart, which are the ones
  directly above and below each other. These pairs do not depend on where the
  rows are cut, which is what lets the width be settled first;
- the **phase**, where the row boundaries fall, from the consecutive pairs: at a
  boundary, consecutive numbers sit at opposite ends of the grid rather than
  side by side. It is a separate question because a montage whose first contact
  was unplaceable starts counting at the second, putting every boundary one
  column out.

-- and gets the exact 112-edge lattice on that crumpled grid, on 4x16 as
readily as 8x8, and with any single contact removed.

The fallback, for a montage with an explicit group column and names that carry
no contact number, is the **relative neighbourhood graph**: join two contacts
when no third is closer to both than they are to each other. It is a comparison
between distances rather than a threshold on them, so it carries no notion of
scale and holds up under moderate drape, but it degrades on a badly crumpled
grid in the way any purely geometric answer must.

An explicit rows-by-columns field on `ElectrodeSet` would retire the width
search, and remains the obvious upgrade if a montage ever turns up that the
recovery gets wrong. It is not needed to draw the montages we have.

## 7. Still to settle

- **quickflat's depth argument.** `make_figure` already takes `depth` for
  volumetric sampling. `add_electrodes` needs its own depth selection, and
  whether the two should share a value or stay independent is a usability call
  better made once P1 is drawable. Default proposed: `depth=None` draws every
  electrode, a float selects a band.
- **Marker shape vocabulary.** The design document asks for a shape dropdown per
  electrode type. The set of shapes, and whether shape is driven by `group_type`
  by default, is worth fixing before P3 so the instanced-mesh code builds the
  right number of geometries.
