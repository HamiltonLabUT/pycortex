// Intracranial electrode markers.
//
// An electrode has no vertex identity, so a stored position cannot survive the
// surface inflating or flattening. What survives is the anchor cortex.electrodes
// computes in python: the triangle a contact sits over, three barycentric
// weights within it, and a normalised cortical depth. This module turns those
// into markers that follow the surface.
//
// The tracking is not new machinery. The picker's axes (facepick.js) and the ROI
// labels (svgoverlay.js) already stay stuck to the surface through a morph by
// listening for the "mix" event and re-deriving their position from posdata.
// Electrodes are that same pattern, which is why no shader and no surface code
// changes to support them.
var electrodes = (function(module) {

    // Per-device-type shape overrides. Empty by default: every contact is a
    // sphere, because that is what an electrode contact is, and shape-coding
    // the device type is a decision the caller should make deliberately rather
    // than inherit. Fill this in for e.g.
    // {grid: "sphere", strip: "cube", seeg: "diamond"}.
    module.SHAPES = {};

    module.DEFAULT_SHAPE = "sphere";

    // One geometry per shape and radius, shared between every contact that wants
    // it. A montage records a handful of distinct diameters at most, so this
    // stays a handful of geometries however many contacts there are.
    module.makeGeometry = function(cache, shape, radius) {
        var key = shape + ":" + radius.toFixed(3);
        if (cache[key] === undefined) {
            if (shape === "cube")
                cache[key] = new THREE.BoxGeometry(radius * 1.7, radius * 1.7, radius * 1.7);
            else if (shape === "diamond")
                cache[key] = new THREE.OctahedronGeometry(radius * 1.35);
            else
                cache[key] = new THREE.SphereGeometry(radius, 12, 8);
        }
        return cache[key];
    };

    module.Electrodes = function(json, posdata, surf) {
        this.surf = surf;
        this.posdata = posdata;
        // Fallback radius, in millimetres. Each contact overrides it with half
        // its own recorded diameter where the montage has one, so markers are
        // drawn at the size the electrodes actually are.
        this.radius = json.radius === undefined ? 1.5 : json.radius;
        this.geometries = {};
        this.subject = json.subject === undefined ? "" : json.subject;
        this.nelec = json.nelec === undefined ? null : json.nelec;
        this._raw = new THREE.Vector3();   // scratch, reused every frame
        // How far, in millimetres, a contact may sit from the depth currently
        // being sampled and still be drawn on a deformed surface. Null shows
        // everything.
        this.depthWindow = json.depth_window === undefined ? 2.0 : json.depth_window;
        this._visible = true;

        this.markers = {left:new THREE.Group(), right:new THREE.Group()};
        this.markers.left.name = "electrodes_left";
        this.markers.right.name = "electrodes_right";
        this.contacts = [];

        // Unlit: the scene carries no lights, so a Lambert or Phong material
        // would render black. Flat colour is also all this phase claims to do --
        // colouring by data is a later step and brings its own shader.
        this.material = new THREE.MeshBasicMaterial({
            color: new THREE.Color(json.color === undefined ? "#ffcc33" : json.color),
        });

        var contacts = json.electrodes === undefined ? [] : json.electrodes;
        for (var i = 0; i < contacts.length; i++) {
            var contact = contacts[i];
            var map = this.posdata[contact.hemi].map;
            // Vertex order is shuffled twice between python and here, and both
            // trips have to be made. The CTM's own spatial sort is applied in
            // python before these indices are sent, so what arrives is already in
            // CTM order; indexMap is three.js's second, local shuffle, applied
            // when it split the buffers into 16-bit index chunks. svgoverlay.js
            // applies only this one because its label anchors are likewise
            // already CTM-ordered.
            //
            // Make only one of the two trips and the markers still land on the
            // surface and still track the morph -- they are simply on the wrong
            // vertices, scattered over the hemisphere. It looks plausible in a
            // screenshot, so it is worth measuring a known within-grid distance
            // rather than trusting the picture.
            var verts = [
                map[contact.verts[0]], map[contact.verts[1]], map[contact.verts[2]],
            ];
            // What the montage says, kept so it can be restored when a view
            // that overrode it is unbound.
            var shape = module.SHAPES[contact.group_type] || module.DEFAULT_SHAPE;
            // One material per contact rather than the shared one, so a contact
            // can take its own colour from data. They all start as copies of
            // the flat colour, so a page with no electrode data looks exactly as
            // it did before -- at the cost of one small material each, which for
            // a few hundred contacts is nothing.
            // Half the recorded contact diameter, or the fallback. `size` is a
            // diameter in millimetres; the geometries take a radius.
            var radius = contact.size ? contact.size / 2 : this.radius;
            var mesh = new THREE.Mesh(
                module.makeGeometry(this.geometries, shape, radius),
                this.material.clone()
            );
            mesh.name = contact.name;
            this.markers[contact.hemi].add(mesh);

            this.contacts.push({
                hemi:       contact.hemi,
                coords:     contact.coords,
                verts:      verts,
                rawVerts:   [contact.verts[0], contact.verts[1], contact.verts[2]],
                weights:    contact.weights,
                shape:      shape,
                radius:     radius,
                baseShape:  shape,
                baseRadius: radius,
                // Position in the montage. Not the same as this contact's index
                // in `contacts`, because the payload drops unplaceable ones --
                // a view's arrays are montage-length and must be read by this.
                index:      contact.index === undefined ? i : contact.index,
                mesh:       mesh,
                name:       contact.name,
                group:      contact.group,
                group_type: contact.group_type,
                anatomy:    contact.anatomy,
                status:     contact.status,
                placement:  contact.placement,
                size:       contact.size,
                depth:      contact.depth,
                depth_mm:   contact.depth_mm,
                thickness:  contact.thickness_mm,
            });
        }

        this._buildLabels();
        this._bindHover();

        this._buildFilters();
        this._bindClick();

        this.ui = (new jsplot.Menu()).add({
            visible: {action:[this, "setVisible"]},
            labels:  {action:[this, "setLabels"]},
            depth_window: {action:[this, "setDepthWindow", 0.0, 20.0]},
            shape: {action:[this, "setShape", ["auto", "sphere", "cube", "diamond"]]},
        });
        if (this._filterFields.length)
            this.ui.addFolder("filter", true, this.filterUI);
    }

    // Re-place every marker for the current inflation and depth. Called on each
    // "mix" event, which the surface dispatches whenever either changes.
    //
    // Two regimes, because an electrode has two positions and only one of them
    // is true at a time. On the anatomical surface the contact's measured
    // TkRegRAS coordinate is exactly right, and it should be drawn there
    // whether or not that puts it on the cortex -- a depth contact belongs
    // inside the brain, and snapping it to a gyrus would be a lie. Once the
    // surface starts to deform, that coordinate means nothing, and the anchor
    // is all that is left. So: true coordinate at surfmix 0, anchored position
    // by the time the first morph target is reached, linear in between.
    module.Electrodes.prototype.setMix = function(evt) {
        var morphs = this.surf.names.length - 1;
        var t = Math.max(0, Math.min(1, evt.mix * morphs));

        for (var i = 0; i < this.contacts.length; i++) {
            var contact = this.contacts[i];

            if (t < 1) {
                // The browser's base surface is the pia in its original
                // coordinates, so a TkRegRAS point drops straight into the
                // same frame with no conversion.
                this._raw.fromArray(contact.coords);
            }
            if (t <= 0) {
                // Anatomical surface: every contact is at its measured position,
                // so there is no "sampled depth" to be near or far from.
                contact.mesh.visible = this._visible && this._passesFilters(contact);

                contact.mesh.position.copy(this._raw);
                continue;
            }

            // Once the surface deforms it is showing one depth through the
            // ribbon -- thickmix 0 is the pia, 1 the white matter -- and a
            // contact that is not at that depth is not on the sheet being
            // drawn. Hide it rather than project it onto a surface it is not
            // near, which is the same reasoning as the placement policy: draw
            // what is there, not what would look tidy.
            contact.mesh.visible = this._visible && this._passesFilters(contact) && this._nearSampledDepth(contact, evt.thickmix);
            if (!contact.mesh.visible)
                continue;

            var vert = mriview.get_position_bary(
                this.posdata[contact.hemi], evt.mix, evt.thickmix,
                contact.verts, contact.weights
            );
            contact.mesh.position.copy(
                t >= 1 ? vert.pos : this._raw.lerp(vert.pos, t)
            );
        }
        this._placeLabels();
    }

    // Draw markers whatever is in front of them. Needed the moment the surface
    // goes translucent: blending does not remove it from the depth buffer, so a
    // contact inside the brain stays culled however see-through the cortex
    // looks. renderOrder puts them after the hull so they are not blended over.
    // Is this contact within the depth window of the surface being sampled?
    // A contact with no known depth -- a subject with no white-matter surface --
    // cannot fail the test, so it is always shown.
    module.Electrodes.prototype._nearSampledDepth = function(contact, thickmix) {
        if (this.depthWindow === null || this.depthWindow === undefined)
            return true;
        if (contact.depth_mm === null || contact.thickness === null)
            return true;
        var sampled = thickmix * contact.thickness;     // mm from the pia
        return Math.abs(contact.depth_mm - sampled) <= this.depthWindow;
    };

    module.Electrodes.prototype.setDepthWindow = function(val) {
        if (val === undefined)
            return this.depthWindow;
        // The top of the slider means "no filtering" rather than a 20 mm window,
        // which would be an arbitrary number pretending to be a limit.
        this.depthWindow = val >= 20 ? null : val;
        this._refresh();
    };

    module.Electrodes.prototype.setXray = function(val) {
        if (val === undefined)
            return !this.material.depthTest;
        this.material.depthTest = !val;
        this.material.needsUpdate = true;
        for (var i = 0; i < this.contacts.length; i++) {
            var mesh = this.contacts[i].mesh;
            mesh.material.depthTest = !val;
            mesh.material.needsUpdate = true;
            mesh.renderOrder = val ? 999 : 0;
        }
    };

    // -- colour from data ---------------------------------------------------
    //
    // The colour is looked up on the CPU rather than sampled in a shader. The
    // viewer already loads every colormap as an <img>, so one drawImage into an
    // offscreen canvas gives the same 256 (or 256x256) texels the shader would
    // read, and for a few hundred contacts indexing that in JavaScript costs
    // nothing measurable. It also keeps the markers on the plain unlit material
    // they already use, instead of a ShaderMaterial that would have to
    // reimplement the depth-window fade and the x-ray mode.
    //
    // What matters is that the answer tracks the same three things the surface's
    // colours do -- the colormap, the vmin/vmax sliders and the movie frame --
    // and it does, because all three are read from the active DataView here and
    // this runs again whenever any of them changes.

    var _cmapCache = {};

    function _cmapPixels(texture) {
        // A colormap texture's <img>, rasterised once and kept. Keyed on the
        // image's own src, since the same colormap is one shared texture.
        if (texture === undefined || texture === null || !texture.image)
            return null;
        var img = texture.image;
        if (!img.complete || !img.width)
            return null;
        var key = img.currentSrc || img.src;
        if (_cmapCache[key] !== undefined)
            return _cmapCache[key];

        var canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        var ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        var out;
        try {
            out = {
                data: ctx.getImageData(0, 0, img.width, img.height).data,
                width: img.width,
                height: img.height,
            };
        } catch (e) {
            out = null;      // a tainted canvas; fall back to the flat colour
        }
        _cmapCache[key] = out;
        return out;
    }

    // Bind the dataview whose values colour these markers. Null unbinds, which
    // is what a volumetric or surface dataset does -- those say nothing about
    // electrodes, so the markers go back to their flat colour.
    module.Electrodes.prototype.setDataView = function(dataview) {
        this.dataview = (dataview && dataview.electrode) ? dataview : null;
        this._applyViewMarkers();
        this.setValues();
    };

    // Size and shape from the bound view, falling back to the montage.
    //
    // The montage records each contact's physical diameter, which is the right
    // default and the same for every dataset drawn on it. A view's vectors are
    // for when the data should drive the marker instead. Unbinding a view -- or
    // binding one that carries nothing -- restores the montage, which is why
    // every contact keeps its baseRadius and baseShape.
    module.Electrodes.prototype._applyViewMarkers = function() {
        var attrs = this.dataview ? (this.dataview.attrs || {}) : {};
        var sizes = attrs.marker_size || null;
        var shapes = attrs.marker_shape || null;

        if (sizes !== null && this.nelec !== null && sizes.length !== this.nelec)
            console.warn("marker_size has " + sizes.length + " entries but the "
                       + "montage has " + this.nelec + " contacts");

        for (var i = 0; i < this.contacts.length; i++) {
            var c = this.contacts[i];
            // By montage index, never by position: the payload drops
            // unplaceable contacts, so the two disagree the moment any montage
            // has one.
            var size = sizes === null ? null : sizes[c.index];
            var radius = (size === null || size === undefined) ? c.baseRadius : size / 2;
            var shape = this._shapeOverride
                || (shapes === null ? null : shapes[c.index])
                || c.baseShape;

            c.radius = radius;
            c.shape = shape;
            c.mesh.geometry = module.makeGeometry(this.geometries, shape, radius);
            if (c.label !== undefined)
                c.label.scale.set(radius * 4 * c.labelAspect, radius * 4, 1);
        }
        this._placeLabels();
        this._refresh();
    };

    // Recolour every contact from the bound dataview's current frame. Cheap
    // enough to call on any change rather than working out which changed.
    // module.Electrodes.prototype.setValues = function() {
    //     var view = this.dataview;
    //     var values = view === null ? null : view.electrodeValues();
    //     var pixels = view === null ? null : _cmapPixels(view.cmap[0].value);

    //     if (values === null || pixels === null) {
    //         for (var i = 0; i < this.contacts.length; i++)
    //             this.contacts[i].mesh.material.color.copy(this.material.color);
    //         this.surf.dispatchEvent({type:"update"});
    //         return;
    //     }

    //     var vmin = view.vmin[0].value[0], vmax = view.vmax[0].value[0];
    //     var span = vmax - vmin;
    //     // A 1-D colormap is one row; a 2-D one is a square, and a scalar view
    //     // read against it should walk its diagonal-free bottom row, which is
    //     // what the shader does for a single channel.
    //     var row = (pixels.height - 1) * pixels.width * 4;
    //     var last = pixels.width - 1;

    //     for (var i = 0; i < this.contacts.length; i++) {
    //         var mat = this.contacts[i].mesh.material;
    //         var v = i < values.length ? values[i] : NaN;
    //         if (isNaN(v)) {
    //             // No value for this contact -- a NaN in the data, or a montage
    //             // longer than the array. Grey says "not measured" rather than
    //             // borrowing whatever colour zero happens to have.
    //             mat.color.setRGB(0.5, 0.5, 0.5);
    //             continue;
    //         }
    //         var frac = span === 0 ? 0.5 : (v - vmin) / span;
    //         frac = frac < 0 ? 0 : (frac > 1 ? 1 : frac);
    //         var o = row + Math.round(frac * last) * 4;
    //         mat.color.setRGB(
    //             pixels.data[o] / 255, pixels.data[o + 1] / 255, pixels.data[o + 2] / 255
    //         );
    //     }
    //     this.surf.dispatchEvent({type:"update"});
    // };
    module.Electrodes.prototype.setValues = function() {
        var view = this.dataview;
        var values = view === null ? null : view.electrodeValues(undefined, 0);
        var pixels = view === null ? null : _cmapPixels(view.cmap[0].value);
        // The second channel, for a 2D view. Null for a scalar one, and then
        // the colormap is read along a single row exactly as the shader reads
        // it with `vec2(x, 0.)`.
        var values2 = view === null ? null : view.electrodeValues(undefined, 1);

        if (values === null || pixels === null) {
            for (var i = 0; i < this.contacts.length; i++)
                this.contacts[i].mesh.material.color.copy(this.material.color);
            this.surf.dispatchEvent({type:"update"});
            return;
        }

        var vmin = view.vmin[0].value[0], vmax = view.vmax[0].value[0];
        var span = vmax - vmin;
        // One pair of bounds per axis of the colormap, not per data array:
        // `value` is [dim1, dim2] and the vertical slider writes its second
        // entry, which is why both come off vmin[0].
        var vmin2 = view.vmin[0].value[1], vmax2 = view.vmax[0].value[1];
        var span2 = vmax2 - vmin2;
        var lastx = pixels.width - 1, lasty = pixels.height - 1;

        for (var i = 0; i < this.contacts.length; i++) {
            var mat = this.contacts[i].mesh.material;
            var v = i < values.length ? values[i] : NaN;
            var v2 = values2 === null ? 0
                : (i < values2.length ? values2[i] : NaN);
            if (isNaN(v) || isNaN(v2)) {
                // No value for this contact -- a NaN in either channel, or a
                // montage longer than the array. Grey says "not measured"
                // rather than borrowing whatever colour zero happens to have,
                // and matches the alpha=0 the matplotlib path gives a NaN in
                // either dimension.
                mat.color.setRGB(0.5, 0.5, 0.5);
                continue;
            }
            var frac = span === 0 ? 0.5 : (v - vmin) / span;
            frac = frac < 0 ? 0 : (frac > 1 ? 1 : frac);
            var frac2 = values2 === null ? 0
                : (span2 === 0 ? 0.5 : (v2 - vmin2) / span2);
            frac2 = frac2 < 0 ? 0 : (frac2 > 1 ? 1 : frac2);
            // `frac2` is a texture coordinate, which runs up the image, while
            // the rows of pixel data run down it -- the colormaps are uploaded
            // with flipY set (mriview.js), so the top row is the *high* end of
            // the second axis. Hence `1 - frac2`, which is the same flip
            // `Dataview2D._to_raw` applies for the matplotlib path. A scalar
            // view has frac2 = 0 and so lands on the bottom row.
            var row = Math.round((1 - frac2) * lasty) * pixels.width;
            var o = (row + Math.round(frac * lastx)) * 4;
            mat.color.setRGB(
                pixels.data[o] / 255, pixels.data[o + 1] / 255, pixels.data[o + 2] / 255
            );
        }
        this.surf.dispatchEvent({type:"update"});
    };

    // -- channel-name labels ------------------------------------------------
    //
    // A camera-facing sprite per contact, carrying its channel name. Ported from
    // makeLabelSprite in a hand-written three.js viewer of Liberty's, with two
    // substitutions this three.js (r69) forces: THREE.CanvasTexture does not
    // exist yet, so the canvas goes through a plain THREE.Texture with
    // needsUpdate set, and renderOrder does not exist either -- but r69 draws
    // sprites in their own pass after the opaque geometry, so depthTest:false
    // alone already puts labels on top.
    module.makeLabelSprite = function(text, height) {
        var canvas = document.createElement("canvas");
        var ctx = canvas.getContext("2d");
        var fontSize = 48;
        ctx.font = fontSize + "px sans-serif";
        canvas.width = Math.ceil(ctx.measureText(text).width) + 20;
        canvas.height = fontSize + 20;

        // Resizing the canvas resets the context, so the font is set twice.
        ctx.font = fontSize + "px sans-serif";
        ctx.fillStyle = "rgba(20, 20, 20, 0.72)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(text, 10, canvas.height / 2);

        var texture = new THREE.Texture(canvas);
        texture.minFilter = THREE.LinearFilter;
        texture.needsUpdate = true;

        var sprite = new THREE.Sprite(new THREE.SpriteMaterial({
            map: texture, transparent: true, depthTest: false,
        }));
        sprite.scale.set(height * canvas.width / canvas.height, height, 1);
        return sprite;
    };

    module.Electrodes.prototype._buildLabels = function() {
        this.labels = [];
        for (var i = 0; i < this.contacts.length; i++) {
            var contact = this.contacts[i];
            // Scaled off the marker radius so labels stay legible whatever the
            // brain is measured in -- pycortex surfaces are millimetres, the
            // viewer this came from used metres.
            var label = module.makeLabelSprite(contact.name, contact.radius * 4);
            // The sprite's texture is size-independent, so a later radius change
            // only has to rescale it -- keep the aspect to do that with.
            contact.labelAspect = label.scale.x / label.scale.y;
            label.visible = false;
            contact.label = label;
            this.labels.push(label);
            this.markers[contact.hemi].add(label);
        }
    };

    // -- hover ---------------------------------------------------------------
    //
    // Raycast against the marker meshes on mousemove and show that one
    // contact's label. The viewer's own picker cannot help here: it resolves a
    // *vertex* by rendering the surface to an offscreen buffer and reading back
    // an encoded index, and electrodes are separate meshes that never appear in
    // that buffer.
    //
    // Reuses the label sprites rather than adding a DOM tooltip, so a hovered
    // name looks identical to a pinned one and the two can coexist: with the
    // labels toggle off you get one at a time, with it on the hovered one is
    // already shown.
    module.Electrodes.prototype._bindHover = function() {
        this._ray = new THREE.Raycaster();
        this._hovered = null;

        $("#brain").on("mousemove.electrodes", function(evt) {
            var el = $("#brain"), off = el.offset();
            var hit = this._pickNDC(
                ((evt.pageX - off.left) / el.width()) * 2 - 1,
                -((evt.pageY - off.top) / el.height()) * 2 + 1
            );
            if (hit === this._hovered)
                return;                     // nothing changed; don't redraw
            this._hovered = hit;
            this._placeLabels();
            if (typeof viewer !== "undefined" && viewer.schedule !== undefined)
                viewer.schedule();
        }.bind(this));
    };

    // Takes normalised device coordinates rather than an event, so the hit test
    // can be driven from anywhere -- a mouse handler, or a test that knows
    // where a contact projects to.
    module.Electrodes.prototype._pickNDC = function(x, y) {
        if (!this._visible || typeof viewer === "undefined" || viewer.camera === undefined)
            return null;

        // r69 has no Raycaster.setFromCamera, so unproject a point on the far
        // plane and aim the ray at it by hand.
        var target = new THREE.Vector3(x, y, 0.5);
        target.unproject(viewer.camera);
        this._ray.set(viewer.camera.position,
                      target.sub(viewer.camera.position).normalize());

        var meshes = [];
        for (var i = 0; i < this.contacts.length; i++)
            if (this.contacts[i].mesh.visible)
                meshes.push(this.contacts[i].mesh);

        var hits = this._ray.intersectObjects(meshes);
        if (!hits.length)
            return null;
        for (var i = 0; i < this.contacts.length; i++)
            if (this.contacts[i].mesh === hits[0].object)
                return this.contacts[i];
        return null;
    };

    // Where a contact lands on screen, in the same normalised coordinates
    // _pickNDC takes. Exists so the hit test can be exercised without
    // synthesising DOM events.
    module.Electrodes.prototype.projectContact = function(i) {
        var mesh = this.contacts[i].mesh;
        mesh.updateMatrixWorld();
        var p = new THREE.Vector3().setFromMatrixPosition(mesh.matrixWorld);
        p.project(viewer.camera);
        return [p.x, p.y];
    };

    // The name under those coordinates, or "" -- a plain value, so it survives
    // the trip across the python bridge.
    module.Electrodes.prototype.pickNameAt = function(x, y) {
        var hit = this._pickNDC(x, y);
        return hit === null ? "" : hit.name;
    };

    // Set the hover from normalised coordinates, as a mousemove would.
    module.Electrodes.prototype.hoverAt = function(x, y) {
        this._hovered = this._pickNDC(x, y);
        this._placeLabels();
        return this.hovered();
    };

    module.Electrodes.prototype.hovered = function() {
        return this._hovered === null ? "" : this._hovered.name;
    };

    // -- filtering -----------------------------------------------------------
    //
    // Dropdowns are built from the values this montage actually contains, so a
    // file with no anatomy column gets no anatomy filter rather than an empty
    // one. Filters compose: each is null for "no restriction", and a contact has
    // to pass all of them.
    // -- click-through metadata ----------------------------------------------
    //
    // Bound on mouseup rather than click, and only when the pointer has barely
    // moved: dragging to rotate the brain ends on whatever mesh it started from
    // and would otherwise open the panel every time you turned the head.
    module.Electrodes.prototype._bindClick = function() {
        var brain = $("#brain");
        brain.on("mousedown.electrodes", function(evt) {
            this._downAt = {x: evt.pageX, y: evt.pageY};
        }.bind(this));
        brain.on("mouseup.electrodes", function(evt) {
            var d = this._downAt;
            this._downAt = null;
            if (!d || Math.abs(evt.pageX - d.x) > 4 || Math.abs(evt.pageY - d.y) > 4)
                return;                      // that was a drag, not a click
            var el = $("#brain"), off = el.offset();
            this.select(this._pickNDC(
                ((evt.pageX - off.left) / el.width()) * 2 - 1,
                -((evt.pageY - off.top) / el.height()) * 2 + 1
            ));
        }.bind(this));
    };

    module.Electrodes.prototype.select = function(contact) {
        this._selected = contact || null;
        var panel = $("#electrode_info");
        if (!panel.length)
            return this.selected();
        if (this._selected === null) {
            panel.css("display", "none");
            return "";
        }

        // Only the fields this montage actually carries. A row reading
        // "anatomy: --" is worse than no row: it looks like missing data rather
        // than like a column the file never had.
        var rows = [["electrode", contact.name]];
        if (this.subject) rows.push(["participant", this.subject]);
        var optional = [["group", contact.group], ["type", contact.group_type],
                        ["anatomy", contact.anatomy], ["status", contact.status]];
        for (var i = 0; i < optional.length; i++)
            if (optional[i][1]) rows.push(optional[i]);
        if (contact.depth_mm !== null && contact.depth_mm !== undefined)
            rows.push(["depth", contact.depth_mm.toFixed(1) + " mm from pia"]);
        if (contact.placement)
            rows.push(["placement", contact.placement.replace(/_/g, " ")]);
        if (contact.value !== null && contact.value !== undefined)
            rows.push(["value", (+contact.value).toPrecision(4)]);

        var html = "";
        for (var i = 0; i < rows.length; i++)
            html += "<div class='erow'><span class='ekey'>" + rows[i][0] +
                    "</span><span class='eval'>" + rows[i][1] + "</span></div>";
        panel.html(html).css("display", "block");
        return contact.name;
    };

    module.Electrodes.prototype.selected = function() {
        return this._selected ? this._selected.name : "";
    };

    // Select from normalised coordinates, as a click would. Keeps the panel
    // testable without synthesising DOM events.
    module.Electrodes.prototype.selectAt = function(x, y) {
        return this.select(this._pickNDC(x, y));
    };

    module.Electrodes.prototype._distinct = function(field) {
        var seen = {}, out = [];
        for (var i = 0; i < this.contacts.length; i++) {
            var v = this.contacts[i][field];
            if (v && seen[v] === undefined) {
                seen[v] = true;
                out.push(v);
            }
        }
        return out.sort();
    };

    module.Electrodes.prototype._buildFilters = function() {
        this.filters = {group: null, group_type: null, anatomy: null, status: null};
        this.filterUI = new jsplot.Menu();
        this._filterFields = [];

        var fields = [["group", "group"], ["group_type", "type"],
                      ["anatomy", "anatomy"], ["status", "status"]];
        for (var i = 0; i < fields.length; i++) {
            var field = fields[i][0], label = fields[i][1];
            // The setter exists for every field, whatever this montage
            // contains, so the API does not change shape with the data. Only
            // the *dropdown* is conditional -- one distinct value is nothing to
            // choose between, and an empty menu is clutter.
            this._makeFilterSetter(field);
            var values = this._distinct(field);
            if (values.length < 2)
                continue;
            var desc = {};
            desc[label] = {action: [this, "filter_" + field, ["all"].concat(values)]};
            this.filterUI.add(desc);
            this._filterFields.push(field);
        }
    };

    // dat.GUI binds one control to one named method, so each field needs its
    // own setter. Generated rather than written out four times.
    module.Electrodes.prototype._makeFilterSetter = function(field) {
        this["filter_" + field] = function(val) {
            if (val === undefined)
                return this.filters[field] === null ? "all" : this.filters[field];
            this.filters[field] = (val === "all") ? null : val;
            this._refresh();
        }.bind(this);
    };

    module.Electrodes.prototype._passesFilters = function(contact) {
        for (var field in this.filters) {
            var want = this.filters[field];
            if (want !== null && contact[field] !== want)
                return false;
        }
        return true;
    };

    // How many contacts are currently drawn. A plain number, so it survives the
    // trip to python and a test can assert on it.
    module.Electrodes.prototype.countVisible = function() {
        var n = 0;
        for (var i = 0; i < this.contacts.length; i++)
            if (this.contacts[i].mesh.visible) n++;
        return n;
    };

    // -- appearance ----------------------------------------------------------

    // A global override on top of the bound view's shapes. "auto" clears it and
    // hands control back -- without that the menu would win permanently after
    // one click and a view's own shapes would be unrecoverable.
    module.Electrodes.prototype.setShape = function(val) {
        if (val === undefined)
            return this._shapeOverride || "auto";
        this._shapeOverride = (val === "auto") ? null : val;
        this._applyViewMarkers();
    };

    module.Electrodes.prototype.setLabels = function(val) {
        if (val === undefined)
            return !!this._labelsOn;
        this._labelsOn = val;
        this._placeLabels();
    };

    // Labels sit just above their marker, and are only shown for markers that
    // are themselves visible -- otherwise a depth-filtered contact would leave
    // its name floating over nothing.
    module.Electrodes.prototype._placeLabels = function() {
        if (this.labels === undefined)
            return;
        for (var i = 0; i < this.contacts.length; i++) {
            var contact = this.contacts[i];
            contact.label.position.copy(contact.mesh.position);
            contact.label.position.z += contact.radius * 2.5;
            contact.label.visible = contact.mesh.visible
                && (!!this._labelsOn || contact === this._hovered);
        }
    };

    module.Electrodes.prototype.setVisible = function(val) {
        if (val === undefined)
            return this._visible;
        this._visible = val;
        this.markers.left.visible = val;
        this.markers.right.visible = val;
        this.surf.dispatchEvent({type:"update"});
    }

    // Re-place the markers at the surface's current state, for a change that did
    // not come from a "mix" event and so carries no state of its own.
    module.Electrodes.prototype._refresh = function() {
        this.setMix({
            mix: this.surf.uniforms.surfmix.value,
            thickmix: this.surf.uniforms.thickmix.value,
        });
        this.surf.dispatchEvent({type:"update"});
    }

    // Every contact, for a caller that wants to inspect what was drawn.
    module.Electrodes.prototype.describe = function() {
        return this.contacts.map(function(contact) {
            return {
                name: contact.name, group: contact.group,
                group_type: contact.group_type, hemi: contact.hemi,
                depth: contact.depth, shape: contact.shape,
                radius: contact.radius, index: contact.index,
                visible: contact.mesh.visible,
                color: contact.mesh.material.color.getHex(),
                verts: contact.verts, weights: contact.weights,
                position: contact.mesh.position.toArray(),
            };
        });
    }

    return module;
}(electrodes || {}));
