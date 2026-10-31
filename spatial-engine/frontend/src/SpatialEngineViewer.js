import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { RoamController } from './RoamController.js';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

// Patch Three.js BufferGeometry and Mesh prototypes for O(log N) BVH raycasting
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

export class SpatialEngineViewer {
  constructor(containerDOM, apiBaseUrl, sceneId) {
    this.container = containerDOM;
    this.apiBaseUrl = apiBaseUrl;
    this.sceneId = sceneId;

    this.threeScene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(
      65, window.innerWidth / window.innerHeight, 0.05, 150.0
    );
    this.camera.position.set(0, 1.5, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);
    this.threeScene.background = new THREE.Color(0x0a0f1a);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.pointCloud = null;
    this.roam = new RoamController(this);
    this.sandbox = null;

    this.raycaster = new THREE.Raycaster();
    this.raycaster.firstHitOnly = true;

    this.collisionMesh = null;
    this.splatViewer = null;
    this.activePathMesh = null;
    this.poiMarkersGroup = new THREE.Group();
    this.threeScene.add(this.poiMarkersGroup);

    this.flightCurve = null;
    this.flightProgress = 0.0;
    this.flightSpeedMetersPerSec = 1.65;
    this.flightTotalLength = 1.0;
    this.isFlying = false;
    this.clock = new THREE.Clock();

    this._bindEvents();
  }

  async _exists(url) {
    try {
      const r = await fetch(url, { method: 'HEAD' });
      return r.ok;
    } catch { return false; }
  }

  async loadDigitalTwin() {
    const art = `${this.apiBaseUrl}/scenes/${this.sceneId}/artifacts`;
    const splatUrl = (await this._exists(`${art}/visual_splats.ksplat`))
      ? `${art}/visual_splats.ksplat` : `${art}/visual_splats.ply`;
    const meshUrl = `${art}/collision_mesh.glb`;
    const pointsUrl = `${art}/points.ply`;

    // Navigation backend is optional: viewing must work without it.
    this.navReady = false;
    try {
      const r = await fetch(`${this.apiBaseUrl}/api/v1/scenes/${this.sceneId}/initialize`, { method: 'POST' });
      this.navReady = r.ok;
    } catch { /* static hosting */ }

    this.threeScene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dl = new THREE.DirectionalLight(0xffffff, 0.8);
    dl.position.set(5, 10, 5);
    this.threeScene.add(dl);

    // Collision mesh (invisible, used for BVH raycasts)
    let meshLoaded = false;
    try {
      const gltf = await new GLTFLoader().loadAsync(meshUrl);
      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          child.geometry.computeBoundsTree();
          child.material = new THREE.MeshBasicMaterial({ visible: false });
          this.collisionMesh = child;
        }
      });
      this.threeScene.add(gltf.scene);
      meshLoaded = true;
    } catch (err) { console.warn('collision mesh unavailable:', err.message); }

    // Visual layer: 3DGS if present, else coloured point cloud
    let splatLoaded = false, pointsLoaded = false;
    if (await this._exists(splatUrl)) {
      try {
        this.splatViewer = new GaussianSplats3D.DropInViewer({
          gpuAcceleratedSort: false, sharedMemoryForWorkers: false,
        });
        await this.splatViewer.addSplatScene(splatUrl, { splatAlphaRemovalThreshold: 5, showLoadingUI: false });
        this.threeScene.add(this.splatViewer);
        splatLoaded = true;
      } catch (err) { console.warn('splats failed:', err.message); this.splatViewer = null; }
    }
    if (!splatLoaded && await this._exists(pointsUrl)) {
      try {
        const geo = await new PLYLoader().loadAsync(pointsUrl);
        const hasColor = !!geo.getAttribute('color');
        this.pointCloud = new THREE.Points(geo, new THREE.PointsMaterial({
          size: 0.06, vertexColors: hasColor, color: hasColor ? 0xffffff : 0x64b4ff, sizeAttenuation: true,
        }));
        this.threeScene.add(this.pointCloud);
        pointsLoaded = true;
        geo.computeBoundingBox();
        const c = geo.boundingBox.getCenter(new THREE.Vector3());
        const size = geo.boundingBox.getSize(new THREE.Vector3()).length();
        this.camera.position.set(c.x, c.y + size * 0.5, c.z + size * 0.8);
        this.controls.target.copy(c);
      } catch (err) { console.warn('points failed:', err.message); }
    }
    if (splatLoaded && this.collisionMesh) {
      // splats have no cheap bounds: frame the camera from the collision mesh instead
      this.collisionMesh.geometry.computeBoundingBox();
      const bb = this.collisionMesh.geometry.boundingBox;
      const c = bb.getCenter(new THREE.Vector3());
      const d = bb.getSize(new THREE.Vector3()).length();
      this.camera.position.set(c.x, c.y + d * 0.15, c.z + d * 0.3);
      this.controls.target.copy(c);
    }
    if (!splatLoaded && !pointsLoaded) {
      if (this.collisionMesh) {
        // No photoreal layer: show the real collision mesh instead of fake geometry
        this.collisionMesh.material = new THREE.MeshStandardMaterial({ color: 0x5b7a99, roughness: 0.8, side: THREE.DoubleSide });
        this.collisionMesh.geometry.computeBoundingBox();
        const bb = this.collisionMesh.geometry.boundingBox;
        const c = bb.getCenter(new THREE.Vector3());
        const d = bb.getSize(new THREE.Vector3()).length();
        this.camera.position.set(c.x, c.y + d * 0.6, c.z + d * 0.7);
        this.controls.target.copy(c);
      } else {
        this._addVisibleRoomGeometry();
      }
    }

    this.scaleFactor = 1.0;
    this.calibrated = false;
    if (this.navReady) {
      try {
        const c = await (await fetch(`${this.apiBaseUrl}/api/v1/scenes/${this.sceneId}/calibration`)).json();
        this.scaleFactor = c.scale_factor; this.calibrated = c.calibrated;
      } catch { /* keep defaults */ }
    }
    this.roam.applyLinkPose();
    this._startRenderLoop();
    return { splatLoaded, pointsLoaded, meshLoaded, navReady: this.navReady };
  }

  _addVisibleRoomGeometry() {
    // Floor
    const floorGeo = new THREE.PlaneGeometry(10, 8);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x2a3a4a,
      roughness: 0.9,
      side: THREE.DoubleSide
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0.01;
    this.threeScene.add(floor);

    // Walls
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x3a4a5a,
      roughness: 0.8,
      side: THREE.DoubleSide
    });

    // Back wall
    const backWall = new THREE.Mesh(new THREE.PlaneGeometry(10, 3), wallMat);
    backWall.position.set(5, 1.5, 8);
    this.threeScene.add(backWall);

    // Front wall
    const frontWall = new THREE.Mesh(new THREE.PlaneGeometry(10, 3), wallMat);
    frontWall.position.set(5, 1.5, 0);
    frontWall.rotation.y = Math.PI;
    this.threeScene.add(frontWall);

    // Left wall
    const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(8, 3), wallMat);
    leftWall.position.set(0, 1.5, 4);
    leftWall.rotation.y = Math.PI / 2;
    this.threeScene.add(leftWall);

    // Right wall
    const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(8, 3), wallMat);
    rightWall.position.set(10, 1.5, 4);
    rightWall.rotation.y = -Math.PI / 2;
    this.threeScene.add(rightWall);

    // Add a grid for reference
    const gridHelper = new THREE.GridHelper(10, 10, 0x64b4ff, 0x2a3a4a);
    gridHelper.position.y = 0.02;
    this.threeScene.add(gridHelper);
  }

  async navigateToPoint(targetWorldPoint) {
    const startXYZ = [this.camera.position.x, this.camera.position.y, this.camera.position.z];
    const goalXYZ = [targetWorldPoint.x, targetWorldPoint.y, targetWorldPoint.z];

    const response = await fetch(`${this.apiBaseUrl}/api/v1/navigation/plan-path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: this.sceneId, start_xyz: startXYZ, goal_xyz: goalXYZ }),
    });

    if (!response.ok) {
      const err = await response.json();
      console.error('Pathplanning failed:', err.detail);
      return null;
    }

    const pathData = await response.json();
    this._renderNavigationRibbon(pathData.waypoints_3d);
    this._beginCameraFlight(pathData.waypoints_3d, pathData.total_length_m);
    return pathData;
  }

  _renderNavigationRibbon(waypoints3D) {
    if (this.activePathMesh) {
      this.threeScene.remove(this.activePathMesh);
      this.activePathMesh.geometry.dispose();
      this.activePathMesh.material.dispose();
    }
    if (waypoints3D.length < 2) return;

    const floorPoints = waypoints3D.map((pt) => new THREE.Vector3(pt[0], 0.12, pt[2]));
    const curve = new THREE.CatmullRomCurve3(floorPoints);
    const tubeGeo = new THREE.TubeGeometry(curve, Math.max(32, floorPoints.length * 2), 0.045, 8, false);
    const tubeMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.85 });
    this.activePathMesh = new THREE.Mesh(tubeGeo, tubeMat);
    this.threeScene.add(this.activePathMesh);
  }

  clearPath() {
    if (this.activePathMesh) {
      this.threeScene.remove(this.activePathMesh);
      this.activePathMesh.geometry.dispose();
      this.activePathMesh.material.dispose();
      this.activePathMesh = null;
    }
    this.isFlying = false;
    this.flightCurve = null;
  }

  _beginCameraFlight(waypoints3D, totalLengthMeters) {
    if (waypoints3D.length < 2 || totalLengthMeters < 0.1) return;
    // stop ~0.9 m short of the goal so the camera does not end inside the target object
    const goal = waypoints3D[waypoints3D.length - 1];
    let trimmed = waypoints3D.filter((pt) => Math.hypot(pt[0] - goal[0], pt[2] - goal[2]) > 0.9);
    if (trimmed.length >= 2) {
      waypoints3D = trimmed;
      totalLengthMeters = Math.max(0.1, totalLengthMeters - 0.9);
    }
    const curvePts = waypoints3D.map((pt) => new THREE.Vector3(pt[0], pt[1], pt[2]));
    this.flightCurve = new THREE.CatmullRomCurve3(curvePts);
    this.flightTotalLength = totalLengthMeters;
    this.flightProgress = 0.0;
    this.isFlying = true;
  }

  async searchAndNavigateToPOI(naturalLanguageQuery) {
    const response = await fetch(`${this.apiBaseUrl}/api/v1/semantics/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene_id: this.sceneId, query_text: naturalLanguageQuery, top_k: 3 }),
    });
    if (!response.ok) return [];
    const pois = await response.json();

    while (this.poiMarkersGroup.children.length > 0) {
      this.poiMarkersGroup.remove(this.poiMarkersGroup.children[0]);
    }

    pois.forEach((poi, idx) => {
      const minVec = new THREE.Vector3(...poi.bbox_min_xyz);
      const maxVec = new THREE.Vector3(...poi.bbox_max_xyz);
      const box3 = new THREE.Box3(minVec, maxVec);
      const helper = new THREE.Box3Helper(box3, idx === 0 ? 0xf59e0b : 0x3b82f6);
      this.poiMarkersGroup.add(helper);
    });

    if (pois.length > 0) {
      const bestPoi = pois[0];
      const targetVec = new THREE.Vector3(...bestPoi.centroid_xyz);
      await this.navigateToPoint(targetVec);
    }
    return pois;
  }

  _bindEvents() {
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
    });

    this.renderer.domElement.addEventListener('dblclick', (event) => {
      if (!this.collisionMesh) return;
      const mouse = new THREE.Vector2(
        (event.clientX / window.innerWidth) * 2 - 1,
        -(event.clientY / window.innerHeight) * 2 + 1
      );
      this.raycaster.setFromCamera(mouse, this.camera);
      const hits = this.raycaster.intersectObject(this.collisionMesh, true);
      if (hits.length > 0) {
        this.navigateToPoint(hits[0].point);
      }
    });
  }

  _startRenderLoop() {
    const animate = () => {
      requestAnimationFrame(animate);
      const dt = this.clock.getDelta();

      if (this.isFlying && this.flightCurve) {
        this.flightProgress += (this.flightSpeedMetersPerSec * dt) / this.flightTotalLength;
        if (this.flightProgress >= 1.0) { this.flightProgress = 1.0; this.isFlying = false; }
        this.camera.position.copy(this.flightCurve.getPointAt(this.flightProgress));
        this.controls.target.lerp(this.flightCurve.getPointAt(Math.min(1.0, this.flightProgress + 0.04)), 0.18);
      }
      if (this.roam.walking) {
        this.roam.update(dt);
      } else {
        this.controls.update();
      if (this.sandbox) this.sandbox.update();
      }

      this.renderer.render(this.threeScene, this.camera);
    };
    animate();
  }
}
