Intracranial electrodes
=======================

pycortex places intracranial recording contacts -- ECoG grids and strips, sEEG
and depth shafts -- onto a subject's cortical surfaces, draws them on a flatmap
or in the WebGL viewer, and carries one value per contact the way
:class:`cortex.Vertex` carries one value per vertex.

Two objects do this, and keeping them apart is deliberate:

:class:`cortex.electrodes.ElectrodeSet`
    Geometry and metadata -- names, coordinates, groups, anatomical labels. No
    measurements. This is what a figure or the viewer draws, and it is the
    electrode counterpart of the SVG ROI overlay.

:class:`cortex.Electrode` and friends
    Data *over* an electrode set, colormapped and serialisable, the counterpart
    of :class:`cortex.Volume` and :class:`cortex.Vertex`.

That split is what lets a metadata-only workflow (place the grid, label it,
check the anatomy) and a data workflow (colour every contact by its
encoding-model correlation) coexist without either carrying the other's
baggage.


Reading a montage
-----------------

:func:`cortex.electrodes.load_electrodes` picks a reader by file extension::

    import cortex.electrodes as ce

    eset = ce.load_electrodes("sub-01_electrodes.tsv", subject="S1")

Three formats are understood:

``*.tsv`` / ``*.csv``
    BIDS ``*_electrodes.tsv``. Its ``name / x / y / z / size / group /
    hemisphere / type / status`` columns map almost one-to-one onto what an
    electrode set holds, which is why it is the interchange format here: other
    labs' files load without a bespoke parser. Column names are matched
    case-insensitively through a table of aliases.

``*.mat``
    An ``img_pipe``-style montage (``elecmatrix`` / ``eleclabels`` /
    ``anatomy``), via :func:`~cortex.electrodes.read_elecs_mat`.

``*.json``
    pycortex's own filestore form, via
    :func:`~cortex.electrodes.load_electrodes_json`.

``subject`` names the pycortex subject whose surfaces these coordinates live
in. It has to be a subject in the filestore, not merely one FreeSurfer has
segmented, because anchoring reads the stored surfaces.

Every field is a parallel array of length ``n``. Absent optional strings are
``""`` and absent numbers are NaN, so there is no ``None`` to guard against and
slicing the set slices every field together.

Grouping
~~~~~~~~

A contact belongs to a *group* -- one physical device. Without a ``group``
column it is inferred from the channel name by
:func:`~cortex.electrodes.infer_group`: ``LSTG1``, ``LSTG_2`` and ``LSTG-03``
all give ``LSTG``, since a shared prefix followed by a number means a shared
device. A name with no trailing number is its own group::

    >>> eset.groups
    ['LSTG', 'LAMY']

``group_type`` is the kind of device -- ``"grid"``, ``"strip"``, ``"seeg"`` or
``"depth"``. It is free-form in practice, so that tuple
(:data:`~cortex.electrodes.GROUP_TYPES`) is the vocabulary the defaults
understand rather than a constraint.

Rows that are not electrodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real clinical files carry placeholder rows for unconnected amplifier channels:
non-finite coordinates, names like ``NaN1``-``NaN32``, blank anatomy. A NaN
reaching the nearest-neighbour search corrupts the whole result rather than
failing, so rows with no finite coordinate are held back from it and marked
:data:`~cortex.electrodes.NO_COORDINATE`. The ``.mat`` reader drops them
outright by default, which also removes the duplicate names a second
disconnected block introduces. Pass ``drop_missing=False`` to keep them, for
when row indices must line up with a data array over all original channels --
but a file whose placeholder names collide is then refused, since an electrode
set requires unique names.


Are the coordinates in the right space?
---------------------------------------

Electrode coordinates are **TkRegRAS**, the space FreeSurfer writes its surfaces
in. Pycortex's stored surfaces are usually *not* in that space: ``import_subj``
converts them with ``mris_convert --to-scanner``, which adds ``c_ras``, so they
are in scanner RAS. The difference is a few millimetres -- 4.70 mm on one
subject in this filestore, 3.35 on another -- which is enough to move every
contact to the neighbouring gyrus.

:func:`~cortex.electrodes.ElectrodeSet.anchor` handles this for you, reading the
GIFTI's own record of which space it is in and shifting a *copy* of the
coordinates; :attr:`~cortex.electrodes.ElectrodeSet.coords` is never modified.
You only need to think about it if you pass ``surfaces=`` explicitly, which
means taking responsibility for the frame they are in.

A montage from another pipeline may be in neither space, and nothing about a
wrong frame is visible on a flatmap -- a grid shifted 15 mm still looks like a
grid.

:func:`~cortex.electrodes.check_alignment` asks the question directly::

    >>> from cortex.electrodes import check_alignment, load_surface_pairs
    >>> report = check_alignment(eset.select(group_type="grid").coords,
    ...                          load_surface_pairs("S1"))
    >>> print(report.summary())
    64 electrodes vs. the pial surface:
      median offset     0.94 mm
      max offset        1.58 mm
      systematic shift  (0.02, -0.11, 0.06) mm, |0.13|

Run it on the surface contacts. Depth electrodes are legitimately centimetres
from the pia and would read as suspicious. A large *systematic* shift is the
signature of a frame mismatch; scattered large offsets are not.

:func:`~cortex.electrodes.check_hemispheres` is the other check worth running
before anything else. A stated ``hemisphere`` column is read but never trusted
-- which hemisphere a contact is in is decided by which surface it is nearer to
-- and this reports where the two disagree. A whole group on the wrong side is
the signature of a left-right flip upstream, and it too is invisible on a
flatmap, where the hemispheres sit side by side and a mirrored grid still looks
like a grid.


Anchoring
---------

An electrode has no vertex identity, so nothing about a stored 3-D position
survives inflation or flattening. Anchoring records instead *where the contact
sits relative to the surface*: which triangle it is over, where inside that
triangle, and how deep through the cortical ribbon. One anchor then evaluates
on any surface the subject has::

    anchors = eset.anchor()
    print(anchors.summary())

:meth:`~cortex.electrodes.ElectrodeSet.anchor` loads the subject's pial and
white-matter surfaces, computes the anchors, stores them on the set and returns
them. It is the step everything else needs; drawing an un-anchored set anchors
it for you.

An :class:`~cortex.electrodes.ElectrodeAnchors` holds, per contact:

``verts``, ``weights``
    The three vertices of the triangle and the barycentric weights within it.
    Vertices rather than a face index, because a face index is *not* shared
    between a subject's surfaces -- flattening cuts the medial wall away, so
    the flat surface has thousands fewer triangles and face *k* is a different
    triangle on each. Vertex indexing is shared by every surface, which is the
    only thing that makes an anchor portable.

``hemi``
    ``"lh"`` or ``"rh"``, whichever surface the contact is nearer.

``depth``, ``depth_mm``, ``thickness_mm``
    Normalised pia-to-white-matter depth (0 at the pia, 1 at the white matter,
    unclamped), the same quantity in millimetres, and the local cortical
    thickness it was normalised by. NaN for a subject with no white-matter
    surface, rather than a fabricated zero.

``offset_mm``, ``dist_pia_mm``, ``dist_wm_mm``
    Distance to the contact's own cortical column, and to each bounding
    surface.

:meth:`~cortex.electrodes.ElectrodeSet.positions` evaluates the anchors on any
surface pycortex holds::

    xyz = eset.positions("flat")        # also "fiducial", "inflated", "pia", "wm"

``nudge`` is on by default, matching :meth:`cortex.database.Database.get_surf`,
which shifts the hemispheres apart for every surface but the fiducial. Anchoring
itself always happens un-nudged.


On the surface, or off it
-------------------------

By default ``positions`` puts every contact *on* the surface. That is right for
a subdural grid, whose contacts sit a fraction of a millimetre from the column
they anchor to, and wrong for a depth electrode, which is not on the surface at
all::

    xyz = eset.positions("inflated", offset="frame")

``offset="frame"`` draws each contact at its real distance off the surface,
by storing the displacement from its anchor point in an orthonormal frame the
anchor triangle defines. Because the same three vertex indices name a triangle
on every surface of the subject, that frame can be rebuilt on the inflated or
flat surface and the displacement re-expressed there.

Why it matters, measured on ``S1``: a 15-contact shaft at a uniform 4 mm pitch
driven 56 mm inward projects onto the inflated surface with consecutive spacings
running from 1.7 to **40.5 mm**, because each contact anchors to whichever sulcal
bank it happens to pass. Carried through one frame, they are all equal.

No individual contact looks wrong -- projection moves none of them more than
5.7 mm, even one 56 mm deep, because cortex folds densely enough that nothing is
ever far from *some* bank. The damage is entirely in the relative geometry, so
per-contact checks cannot see it.

Devices are grouped automatically by ``group_type``: ``seeg`` and ``depth``
groups share one frame and stay rigid, everything else keeps a frame per contact
and drapes over the folds.

Many montages record no ``group_type`` at all, so where it is missing the
grouping falls back to the geometry -- a depth electrode is a rigid needle and
nothing else in the vocabulary is. Measured across real montages, shafts sit
within 0.95 mm of their own best-fit line and a 64-contact grid sits 49 mm off
it, so the two do not overlap. A strip is linear but follows the convexity it
lies on, and is correctly not treated as rigid. An explicit ``group_type`` is
always believed; this only decides the unlabelled case.

:func:`~cortex.electrodes.regroup_anchors` and ``anchor(anchor_mode=...)``
override all of it.

What survives the trip:

============================  =========================  ======================
quantity                      seeg / depth               grid / strip
============================  =========================  ======================
within-group spacing          **uniform**, scaled        follows local stretch
straightness, entry angle     **exact**                  n/a
distance off the sheet        scaled                     exact, in mm
============================  =========================  ======================

A grid deliberately keeps none of that rigidity: inflation stretches the sheet
unevenly, and that unevenness is the signal -- it is what says which contacts
were buried in a sulcus.

Asking for the ``pia`` this way returns the input coordinates exactly, which is
a useful check that a montage is in the space you think it is.


Placement: what the policy did, and to what
-------------------------------------------

Not every contact has cortex to be drawn on. Rather than dropping those,
anchoring *marks* them: every contact keeps its coordinate and its anchor, and
``placement`` records the outcome.

:data:`~cortex.electrodes.ON_SURFACE`
    Inside the cortical ribbon.
:data:`~cortex.electrodes.PROJECTED`
    Outside the ribbon, but near enough to project onto it.
:data:`~cortex.electrodes.TOO_FAR`
    Further from cortex than the policy allows.
:data:`~cortex.electrodes.NON_CORTICAL`
    The anatomical label names white matter or a subcortical structure.
:data:`~cortex.electrodes.UNKNOWN_ANATOMY`
    The label is counted as unknown, when that rule is enabled.
:data:`~cortex.electrodes.NO_COORDINATE`
    No finite coordinate to place.

:class:`~cortex.electrodes.PlacementPolicy` configures it, and one number
decides it by default: ``max_surface_distance_mm``, how far from the nearer
bounding surface a contact may sit and still be projected. Four millimetres.
Measured on a real subject, a subdural grid resting on the pia stays under
1.6 mm and a depth electrode driven 56 mm inward stays under 2.3 mm -- cortex
folds, so a shaft is never far from *some* sulcal bank. What four millimetres
excludes is a contact in the deep white-matter core, in a ventricle, or outside
the head. Set it to ``np.inf`` to project everything::

    from cortex.electrodes import PlacementPolicy
    eset.anchor(policy=PlacementPolicy(max_surface_distance_mm=8.0))

This is emphatically **not** a registration check, and the same folding that
makes it generous is why: a *contiguous* grid lifted 15 mm off the convexity it
rested on fails every contact, but a *scattered* montage shifted the same 15 mm
keeps more than half of them, because each stray point finds some bank to land
near. Use ``check_alignment`` for that question.

``NON_CORTICAL`` is on by default and is the one thing geometry cannot supply.
Geometry cannot tell that a contact is in white matter, because a white-matter
contact still has cortex a millimetre away on some sulcal bank; the anatomical
label can. It marks rather than excludes -- where a hippocampal contact
projects to is usually exactly what a reader wants shown -- and
``select(placeable=True)`` keeps it.

:meth:`~cortex.electrodes.ElectrodeSet.reclassify` re-applies a policy without
redoing the geometry, which is the cheap way to try a different threshold.


Selecting a subset
------------------

:meth:`~cortex.electrodes.ElectrodeSet.select` filters on any string field,
including ``hemi`` and ``placement`` once anchored, and returns a new set::

    lstg   = eset.select(group="LSTG")
    good   = eset.select(group_type=["grid", "strip"], status="good")
    shown  = eset.select(placeable=True)
    shallow = eset.select(where=eset.depth < 0.5)

An empty result is an ordinary answer -- no contact in that group, none at that
depth -- so an empty set is legal and does not raise.


Drawing on a flatmap
--------------------

:func:`cortex.quickflat.add_electrodes` draws markers onto an existing figure;
``make_figure`` (and so ``make_png`` and ``make_svg``) takes
``with_electrodes``::

    fig = cortex.quickflat.make_figure(
        blank, with_curvature=True,
        with_electrodes=eset,
        electrode_values=eset.depth,
        electrode_kwargs=dict(cmap="RdYlBu_r", vmin=0, vmax=1, size=55),
    )

``with_electrodes`` accepts an :class:`~cortex.electrodes.ElectrodeSet`, a
:class:`cortex.Electrode` (which carries its own values), ``True`` for the
subject's default filestore set, or the name of one.

Marker shape follows group type by default -- a circle for a grid, a square for
a strip, a diamond for seeg and depth -- so a reader can tell which positions to
trust without consulting a legend. A depth contact's surface position is a
locality, not a location. Marker *size* is constant in figure units and
deliberately **not** scaled by areal distortion: a flatmap stretches some
cortex and compresses other cortex, and scaling markers by that would say
something about the contacts that is not true of them. ``size_by="size"`` sizes
them by their recorded contact diameter instead.

``depth`` selects a band -- ``depth=(0.0, 1.0)`` keeps only the contacts inside
grey matter, ``None`` (the default) draws them all. It is a **separate argument
from** ``make_figure``'s own ``depth``, which is the volumetric sampling depth
for ``add_data``: someone sampling their fMRI at ``depth=0.5`` has not thereby
asked to hide all but mid-ribbon contacts.

Electrodes are drawn after ``add_cutout``, because they are scatter markers
rather than image layers and a cutout iterating its layers would choke on a
``PathCollection``. The cutout's reset axis limits clip them instead, which is
right except for a marker inside the cutout's bounding box and outside its
outline.

.. seealso:: :doc:`auto_examples/electrodes/plot_electrodes_flatmap`


In the WebGL viewer
-------------------

::

    cortex.webgl.show(cortex.Vertex.random("S1"), electrodes=eset)
    cortex.webgl.make_static("/tmp/viewer", data, electrodes=eset)

What is sent to the browser is not a position but the anchor, and the marker is
re-placed on every ``mix`` event, so contacts travel with the cortex as you drag
the ``unfold`` slider.

With ``offsets`` on -- the default -- the marker is drawn at its real distance
off the surface, by rebuilding its stored frame against whatever the inflation
slider is showing. On the anatomical surface that reconstruction *is* the
measured TkRegRAS coordinate, exactly, so there is one rule rather than a
special case: a depth contact is inside the brain because that is where it is,
and it stays a straight track as the cortex inflates. Turn ``surface_opacity``
down to see contacts that are inside.

Turning ``offsets`` off pins every contact to the sheet. That is the older
behaviour and still the better picture for a subdural grid read against the
surface it sits on: the measured coordinate on the anatomical surface, crossing
to the projected anchor as the morph begins.

The ``electrodes`` folder holds:

``visible``
    Draw the markers at all.
``labels``
    Channel names beside every contact. With it off, hovering shows one at a
    time.
``connections``
    Join contacts that are neighbours on the same device (see below). On by
    default.
``offsets``
    Draw contacts at their real distance off the surface rather than on it. On
    by default. Turn it off to read a grid against the sheet it sits on.
``depth_window``
    Hide contacts more than this many millimetres from the depth being sampled.
    The top of the slider means no filtering rather than a 20 mm window, which
    would be an arbitrary number pretending to be a limit.
``depth_follows_slider``
    Measure that window from the surface's own depth slider instead of from the
    pia. Off by default.
``shape``
    ``auto`` (by group type), or force ``sphere``, ``cube`` or ``diamond``.
``filter``
    A sub-folder with a dropdown per metadata field -- group, type, anatomy,
    status -- built only for the fields this montage actually varies in.

Clicking a contact fills the metadata panel with its name, group, anatomy,
placement, contact diameter and depth. Markers are sized from the recorded
contact diameter where the montage has one.

Turning the cortex translucent switches the markers to depth-test-free
rendering automatically: blending does not remove the surface from the depth
buffer, so without that a contact inside the brain stays culled however
see-through the cortex looks.

.. seealso:: :doc:`auto_examples/electrodes/plot_electrodes_webgl`

Connection lines
~~~~~~~~~~~~~~~~

A thin segment between contacts that are neighbours on the same device, so a
montage reads as a grid and three shanks rather than as a cloud of dots. Which
pairs those are is decided in Python, once, by
:func:`cortex.electrodes.group_edges`, and sent as an index list. Deriving it in
the browser from the drawn positions would make the topology depend on the
inflation slider, with edges appearing and disappearing mid-morph, when the fact
being drawn is a property of the physical device.

For a linear device -- strip, sEEG, depth -- that is just the order along the
shank, taken from the contact numbers in the names.

A grid is the hard case, because **no format the readers support records a
grid's row and column count**. Neither BIDS' ``electrodes.tsv`` nor img_pipe's
``elecmatrix`` carries one, so a grid arrives as nothing more than a set of
positions sharing a group name. Joining contacts closer together than some
multiple of the median spacing does not survive contact with a real montage: a
grid conformed to a folded surface has no single pitch, and on an 8x8 grid
draped over lateral temporal cortex the nearest-neighbour distance ranges from
1.1 to 9.8 mm against a nominal 8 mm. Any threshold that admits the stretched
pairs also admits diagonals somewhere else.

What works is that a grid is a rigid rectangular sheet **numbered along its
rows**, so its lattice is a fact about the numbering and the only unknown is the
row width -- one number, whose wrong answers land multiples of the pitch away.
``_connect.py`` recovers the width from the contacts numbered *width* apart
(the ones directly above and below each other) and then the phase, where the row
boundaries fall, from the consecutive pairs. For a montage whose names carry no
contact number it falls back to the relative neighbourhood graph: join two
contacts when no third is closer to both than they are to each other, a
comparison between distances rather than a threshold on them.


Data over electrodes
--------------------

:class:`cortex.Electrode` is one value per contact with a colormap, used exactly
as :class:`cortex.Volume` and :class:`cortex.Vertex` are::

    vol  = cortex.Volume(data, "S1", "fullhead")
    elec = cortex.Electrode(data, "S1", "native")

:class:`cortex.Electrode2D` takes two arrays and a 2-D colormap;
:class:`cortex.ElectrodeRGB` takes three channels and an optional alpha. All
three go into a :class:`cortex.Dataset`, save to HDF and load back, and animate:
``(t, n_contacts)`` is a movie, scrubbed by the viewer's frame control.

``marker_size`` and ``marker_shape`` set the drawn size and shape per contact,
overriding the defaults from the montage; both quickflat and the viewer honour
them. They do not animate -- a movie's frames all draw at the same size.

The second argument is the implanted subject and the third is the *montage*.
Note that :attr:`~cortex.Electrode.subject` does not report the former back::

    >>> cortex.Electrode(d, "S1", "fsaverage").subject
    'fsaverage'
    >>> cortex.Electrode(d, "S1", "fsaverage").montage_subjects
    ('S1',)

That is not a quirk but the only thing that can work: every renderer reads
``dataview.subject`` to decide which brain to load, so reporting the implanted
subject there would draw fsaverage coordinates on that patient's cortex.


Montages in the filestore
-------------------------

A *montage* is one subject's electrodes as localised in some subject's anatomy.
``"native"`` is the patient's own scan; any other name is a template the lab
registered them to::

    cortex.db.save_montage("S1", eset, montage="native")
    eset = cortex.db.get_montage("S1", "fsaverage")
    cortex.db.list_montages("S1")

The file is filed under the subject whose head the electrodes are in, while the
coordinates belong to the surfaces of a possibly different subject::

    <filestore>/S1/electrodes/electrodes_native.json     -> S1's surfaces
    <filestore>/S1/electrodes/electrodes_fsaverage.json  -> fsaverage's

:meth:`~cortex.database.Database.get_montage` sets ``subject`` to the *surface*
subject and ``owner`` to the implanted one, which is the point of the method:
anchoring loads surfaces by ``eset.subject``, so returning a template montage
still labelled with the implanted subject would anchor fsaverage coordinates
against that patient's cortex, silently and plausibly.

This is a naming convention over
:meth:`~cortex.database.Database.get_electrodes` and
:meth:`~cortex.database.Database.save_electrodes`, which read and write any
``<filestore>/<subject>/electrodes/<name>.json`` and are still the way to keep a
set that is not a montage.

Several subjects at once
~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`cortex.Electrode.concat` puts several subjects' contacts on one common
surface as a single view, with one colormap and one pair of bounds::

    combined = cortex.Electrode.concat("fsaverage", {"S1": s1_values, "S2": s2_values})
    combined.electrodes.select(owner="S1")

Contact names are namespaced ``"S1/LSTG1"``, since an electrode set requires
unique names and two subjects will both have an ``LSTG1``; ``owner`` records
which subject each came from.

:meth:`cortex.Electrode.to_sub` moves one view onto another subject's surfaces.
It reads the montage the lab already localised into that space -- it does *not*
transform the coordinates. Registering a montage to a template is an offline
step with its own tooling, and inventing an answer here would put contacts
somewhere plausible and wrong.
