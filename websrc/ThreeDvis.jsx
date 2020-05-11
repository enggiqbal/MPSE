import React, { Component } from "react";
import ReactDOM from "react-dom";
import * as THREE from "three";
//import OrbitControls  from 'three-orbit-controls' 
var OrbitControls = require('three-orbit-controls')(THREE)
import { points } from "/Users/iqbal/projects/MPSE-web/MPSE/precomputed/circlesquire/coordinates.js";


//import {points} from "/Users/iqbal/projects/MPSE-web/MPSE/precomputed/123/coordinates.js";


export class ThreeDvis extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            points: points,
            backgroundcolor: 'black'
        }
    }



    componentDidMount() {
        let points = this.state.points;
        var scene = new THREE.Scene();
        var camera = new THREE.PerspectiveCamera(1, window.innerWidth / window.innerHeight, 1, 100000);
        camera.position.set(0, 0, 35);
        camera.lookAt(scene.position);
        var renderer = new THREE.WebGLRenderer();
        this.mount.appendChild(renderer.domElement);

        scene.background = new THREE.Color(this.state.backgroundcolor);
        var controls = new OrbitControls(camera, renderer.domElement);

        for (var i = 0; i < points.length; i++) {
            var xyz = points[i];
            var dotGeometry = new THREE.Geometry();
            dotGeometry.vertices.push(new THREE.Vector3(xyz[0], xyz[1], xyz[2]));
            var dotMaterial = new THREE.PointsMaterial({ color: 'red', size: 1, sizeAttenuation: false });
            var dot = new THREE.Points(dotGeometry, dotMaterial);
            scene.add(dot);
        }
        var worldAxis = new THREE.AxesHelper(5);
        scene.add(worldAxis);
        (function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        })();
    }


    render() {
        return (
            <div className="row top-buffer">
                <div className="col-md-12" ref={ref => (this.mount = ref)} />
            </div>
        )
    }
} 