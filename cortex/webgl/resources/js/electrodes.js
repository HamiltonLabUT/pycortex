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

    // Shape carries the device, so a reader can tell a grid from a depth
    // electrode without consulting a legend -- and the two mean quite different
    // things about how far to trust the position drawn. Deliberately the same
    // vocabulary quickflat's add_electrodes uses: circle, square, diamond.
    module.SHAPES = {
        grid:  "sphere",
        strip: "cube",
        seeg:  "diamond",
        depth: "diamond",
    };
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
                verts:      verts,
                rawVerts:   [contact.verts[0], contact.verts[1], contact.verts[2]],
                weights:    contact.weights,
                shape:      shape,
                mesh:       mesh,
                name:       contact.name,
                group:      contact.group,
                group_type: contact.group_type,
                depth:      contact.depth,
            });
        }

        this.ui = (new jsplot.Menu()).add({
            visible: {action:[this, "setVisible"]},
            radius:  {action:[this, "setRadius", 0.5, 6.0]},
            lift:    {action:[this, "setLift", 0.0, 4.0]},
        });
    }

    // Re-place every marker for the current inflation and depth. Called on each
    // "mix" event, which the surface dispatches whenever either changes.
    module.Electrodes.prototype.setMix = function(evt) {
        for (var i = 0; i < this.contacts.length; i++) {
            var contact = this.contacts[i];
            var vert = mriview.get_position_bary(
                this.posdata[contact.hemi], evt.mix, evt.thickmix,
                contact.verts, contact.weights
            );
            // Stand the marker off along the local normal, so it reads as sitting
            // on the cortex rather than half sunk into it.
            vert.pos.add(
                vert.norm.normalize().multiplyScalar(this.radius * this.lift)
            );
            contact.mesh.position.copy(vert.pos);
        }
    }

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
                verts: contact.verts, weights: contact.weights,
                position: contact.mesh.position.toArray(),
            };
        });
    }

    return module;
}(electrodes || {}));
