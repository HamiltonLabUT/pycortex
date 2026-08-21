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

    module.makeGeometries = function(radius) {
        return {
            sphere:  new THREE.SphereGeometry(radius, 12, 8),
            cube:    new THREE.BoxGeometry(radius * 1.7, radius * 1.7, radius * 1.7),
            diamond: new THREE.OctahedronGeometry(radius * 1.35),
        };
    }

    module.Electrodes = function(json, posdata, surf) {
        this.surf = surf;
        this.posdata = posdata;
        this.radius = json.radius === undefined ? 1.5 : json.radius;
        this.lift = json.lift === undefined ? 1.0 : json.lift;
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

        this.geometries = module.makeGeometries(this.radius);
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
            var shape = module.SHAPES[contact.group_type] || module.DEFAULT_SHAPE;
            var mesh = new THREE.Mesh(this.geometries[shape], this.material);
            mesh.name = contact.name;
            this.markers[contact.hemi].add(mesh);

            this.contacts.push({
                hemi:       contact.hemi,
                coords:     contact.coords,
                verts:      verts,
                rawVerts:   [contact.verts[0], contact.verts[1], contact.verts[2]],
                weights:    contact.weights,
                shape:      shape,
                mesh:       mesh,
                name:       contact.name,
                group:      contact.group,
                group_type: contact.group_type,
                depth:      contact.depth,
                depth_mm:   contact.depth_mm,
                thickness:  contact.thickness_mm,
            });
        }

        this.ui = (new jsplot.Menu()).add({
            visible: {action:[this, "setVisible"]},
            radius:  {action:[this, "setRadius", 0.5, 6.0]},
            lift:    {action:[this, "setLift", 0.0, 4.0]},
            depth_window: {action:[this, "setDepthWindow", 0.0, 20.0]},
        });
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
                contact.mesh.visible = this._visible;
                contact.mesh.position.copy(this._raw);
                continue;
            }

            // Once the surface deforms it is showing one depth through the
            // ribbon -- thickmix 0 is the pia, 1 the white matter -- and a
            // contact that is not at that depth is not on the sheet being
            // drawn. Hide it rather than project it onto a surface it is not
            // near, which is the same reasoning as the placement policy: draw
            // what is there, not what would look tidy.
            contact.mesh.visible = this._visible && this._nearSampledDepth(contact, evt.thickmix);
            if (!contact.mesh.visible)
                continue;

            var vert = mriview.get_position_bary(
                this.posdata[contact.hemi], evt.mix, evt.thickmix,
                contact.verts, contact.weights
            );
            // Stand the marker off along the local normal, so it reads as sitting
            // on the cortex rather than half sunk into it.
            vert.pos.add(
                vert.norm.normalize().multiplyScalar(this.radius * this.lift)
            );
            contact.mesh.position.copy(
                t >= 1 ? vert.pos : this._raw.lerp(vert.pos, t)
            );
        }
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
        for (var i = 0; i < this.contacts.length; i++)
            this.contacts[i].mesh.renderOrder = val ? 999 : 0;
    };

    module.Electrodes.prototype.setVisible = function(val) {
        if (val === undefined)
            return this._visible;
        this._visible = val;
        this.markers.left.visible = val;
        this.markers.right.visible = val;
        this.surf.dispatchEvent({type:"update"});
    }

    module.Electrodes.prototype.setRadius = function(val) {
        if (val === undefined)
            return this.radius;
        this.radius = val;
        var geometries = module.makeGeometries(val);
        for (var i = 0; i < this.contacts.length; i++)
            this.contacts[i].mesh.geometry = geometries[this.contacts[i].shape];
        for (var shape in this.geometries)
            this.geometries[shape].dispose();
        this.geometries = geometries;
        this._refresh();
    }

    module.Electrodes.prototype.setLift = function(val) {
        if (val === undefined)
            return this.lift;
        this.lift = val;
        this._refresh();
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
                visible: contact.mesh.visible,
                verts: contact.verts, weights: contact.weights,
                position: contact.mesh.position.toArray(),
            };
        });
    }

    return module;
}(electrodes || {}));
