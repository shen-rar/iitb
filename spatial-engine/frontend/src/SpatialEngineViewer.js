import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from 'three-mesh-bvh';
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

  async loadDigitalTwin() {
    const splatUrl = `${this.apiBaseUrl}/scenes/${this.sceneId}/artifacts/visual_splats.ksplat`;
    const meshUrl = `${this.apiBaseUrl}/scenes/${this.sceneId}/artifacts/collision_mesh.glb`;

    // Initialize the scene first
    const response = await fetch(`${this.apiBaseUrl}/api/v1/scenes/${this.sceneId}/initialize`, {
      method: 'POST',
    });
    if (!response.ok) {
      throw new Error('Failed to initialize scene');
    }

    const gltfLoader = new GLTFLoader();
    const gltf = await gltfLoader.loadAsync(meshUrl);

    gltf.scene.traverse((child) => {
      if (child.isMesh) {
        child.geometry.computeBoundsTree();
        child.material = new THREE.MeshBasicMaterial({
          color: 0x00ffcc, wireframe: true, visible: false,
        });
        this.collisionMesh = child;
      }
    });
    this.threeScene.add(gltf.scene);

    // Add visible room geometry for demo
    this._addVisibleRoomGeometry();

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.threeScene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 10, 5);
    this.threeScene.add(directionalLight);

    // Try to load splats if they exist, but don't fail if they don't
    let splatLoaded = false;
    try {
      this.splatViewer = new GaussianSplats3D.Viewer({
        selfDrivenMode: false,
        renderer: this.renderer,
        camera: this.camera,
        threeScene: this.threeScene,
        useBuiltInControls: true,
        gpuAcceleratedSort: true,
        sharedMemoryForWorkers: false,
      });

      await this.splatViewer.addSplatScene(splatUrl, {
        splatAlphaRemovalThreshold: 5,
        showLoadingUI: false,
        position: [0, 0, 0],
        rotation: [0, 0, 0, 1],
        scale: [1, 1, 1],
      });
      splatLoaded = true;
    } catch (err) {
      console.warn('Gaussian splats not available, using basic geometry:', err.message);
      this.splatViewer = null;
    }

    this._startRenderLoop();
    return { splatLoaded };
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
        if (this.flightProgress >= 1.0) {
          this.flightProgress = 1.0;
          this.isFlying = false;
        }
        const currentPos = this.flightCurve.getPointAt(this.flightProgress);
        const lookAheadT = Math.min(1.0, this.flightProgress + 0.04);
        const lookTarget = this.flightCurve.getPointAt(lookAheadT);
        this.camera.position.copy(currentPos);
        if (this.splatViewer && this.splatViewer.controls) {
          this.splatViewer.controls.target.lerp(lookTarget, 0.18);
        }
      }

      if (this.splatViewer) {
        this.splatViewer.update();
        this.splatViewer.render();
      }
    };
    animate();
  }
}