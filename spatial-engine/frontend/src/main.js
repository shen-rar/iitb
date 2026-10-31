import { SpatialEngineViewer } from './SpatialEngineViewer.js';
import { PoiSearchUI } from './PoiSearchUI.js';
import { FieldHUD } from './FieldHUD.js';
import { Sandbox } from './Sandbox.js';

// Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';
const SCENE_ID = new URLSearchParams(window.location.search).get('scene') || 'demo_office';

async function main() {
  const $ = (id) => document.getElementById(id);

  const viewer = new SpatialEngineViewer($('canvas-container'), API_BASE_URL, SCENE_ID);
  window.__viewer = viewer; // handy for debugging in the console
  const roam = viewer.roam;
  const hud = new FieldHUD(viewer, API_BASE_URL, SCENE_ID);
  const sandbox = new Sandbox(viewer, hud, roam);
  viewer.sandbox = sandbox;

  // Measurement readout and calibration (first measurement asks for a real length once)
  const readout = $('readout');
  const showDistance = (units) => {
    readout.style.display = 'block';
    const m = units * viewer.scaleFactor;
    readout.textContent = viewer.calibrated
      ? `Distance: ${m.toFixed(2)} m (calibrated)`
      : `Distance: ${m.toFixed(2)} m (estimated: calibrate with a known length)`;
  };
  roam.onMeasure = async (units) => {
    showDistance(units);
    if (viewer.calibrated || !viewer.navReady) return;
    const real = parseFloat(window.prompt('Calibrate: real length of this segment in metres (blank to skip)', ''));
    if (!(real > 0)) return;
    const r = await fetch(`${API_BASE_URL}/api/v1/scenes/${SCENE_ID}/calibration`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ measured_units: units, real_metres: real }),
    });
    if (r.ok) {
      const c = await r.json();
      viewer.scaleFactor = c.scale_factor; viewer.calibrated = true;
      showDistance(units);
    }
  };

  // Toolbar
  $('btn-walk').onclick = () => roam.toggleWalk();
  $('btn-plan').onclick = () => roam.togglePlan();
  $('btn-points').onclick = () => roam.toggleVisual();
  $('btn-measure').onclick = () => {
    roam.toggleMeasure();
    readout.style.display = roam.measuring ? 'block' : 'none';
    if (roam.measuring) readout.textContent = 'Measure: click two points';
  };
  const copyLink = async (url, btn) => {
    try { await navigator.clipboard.writeText(url); btn.textContent = 'LINK COPIED'; }
    catch { window.prompt('Copy link', url); }
    setTimeout(() => { btn.textContent = 'COPY LINK'; }, 1500);
  };
  $('btn-link').onclick = () => copyLink(roam.shareLink(), $('btn-link'));
  window.addEventListener('roam-mode', (e) => {
    $('btn-walk').classList.toggle('on', e.detail.walking);
    $('btn-plan').classList.toggle('on', e.detail.plan);
    $('btn-measure').classList.toggle('on', e.detail.measuring);
    if (!e.detail.measuring) readout.style.display = 'none';
  });

  // Connection / layer status (status bar)
  const setDot = (id, textId, text, warn = false) => {
    $(id).classList.toggle('warning', warn);
    $(textId).textContent = text;
  };
  setDot('conn-status', 'conn-text', 'Viewer ready');
  setDot('mesh-status', 'mesh-text', 'Mesh: loading…', true);
  setDot('splat-status', 'splat-text', 'Layer: loading…', true);
  hud.setSubtitle('Loading scene…');

  try {
    const result = await viewer.loadDigitalTwin();

    setDot('mesh-status', 'mesh-text', result.meshLoaded ? 'Mesh: BVH ready' : 'Mesh: not available', !result.meshLoaded);
    const layer = result.splatLoaded ? 'Splats: rendering' : result.pointsLoaded ? 'Point cloud: rendering' : 'Layer: none';
    setDot('splat-status', 'splat-text', layer, !(result.splatLoaded || result.pointsLoaded));

    const live = result.splatLoaded || result.pointsLoaded;
    hud.setStamp(live, 'mesh only');
    hud.setSubtitle(`${SCENE_ID} · ${result.splatLoaded ? 'gaussian splats' : result.pointsLoaded ? 'point cloud' : 'collision mesh'} · navigation ${result.navReady ? 'on' : 'off'}`);
    hud.showReference();
    hud.setupGate(roam);
    if (await hud.loadMap()) hud.startLoop();

    // Plan clicks set the route end; sandbox restores props from link or this browser
    hud.onMapClick = (x, z) => sandbox.routeTo(x, 0, z);
    sandbox.restore();
    sandbox.onRoute = (path) => {
      $('nav-info').classList.add('active');
      $('nav-distance').textContent = `${path.total_length_m.toFixed(2)} m`;
      $('nav-clearance').textContent = `${path.min_obstacle_clearance_m.toFixed(2)} m`;
      $('nav-waypoints').textContent = `${path.waypoints_3d.length}`;
    };
    $('clear-path-btn').addEventListener('click', () => { hud.route = []; $('legend-route').style.display = 'none'; });
  } catch (err) {
    console.error('Failed to load digital twin:', err);
    setDot('conn-status', 'conn-text', 'Error');
    hud.setStamp(false, 'load failed');
    hud.setSubtitle(`Failed to load "${SCENE_ID}". Check the backend at ${API_BASE_URL || 'this origin'}.`);
  }

  // Poi search keeps its own panel logic
  new PoiSearchUI(viewer);
}

main();
