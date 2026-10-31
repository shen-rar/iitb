import { SpatialEngineViewer } from './SpatialEngineViewer.js';
import { PoiSearchUI } from './PoiSearchUI.js';

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const SCENE_ID = new URLSearchParams(window.location.search).get('scene') || 'demo_office';

async function main() {
  const container = document.getElementById('canvas-container');
  const connStatus = document.getElementById('conn-status');
  const connText = document.getElementById('conn-text');
  const meshStatus = document.getElementById('mesh-status');
  const meshText = document.getElementById('mesh-text');
  const splatStatus = document.getElementById('splat-status');
  const splatText = document.getElementById('splat-text');
  const sceneInfo = document.getElementById('scene-info');

  // Initialize the Spatial Engine Viewer
  const viewer = new SpatialEngineViewer(container, API_BASE_URL, SCENE_ID);

  // Initialize the POI Search UI
  const searchUI = new PoiSearchUI(viewer);

  // Status updates
  connText.textContent = 'Connected';
  connStatus.style.background = '#22c55e';

  try {
    meshText.textContent = 'Mesh: Loading...';
    meshStatus.classList.add('warning');
    splatText.textContent = 'Splats: Loading...';
    splatStatus.classList.add('warning');

    const result = await viewer.loadDigitalTwin();

    meshText.textContent = 'Mesh: BVH Ready';
    meshStatus.classList.remove('warning');

    if (result.splatLoaded) {
      splatText.textContent = 'Splats: Rendering';
      splatStatus.classList.remove('warning');
    } else {
      splatText.textContent = 'Splats: Not Available (using basic geometry)';
      splatStatus.classList.remove('warning');
      splatStatus.style.background = '#f59e0b';
    }

    sceneInfo.innerHTML = `
      <div style="font-size:12px;line-height:1.8;">
        <div><strong>Scene:</strong> ${SCENE_ID}</div>
        <div><strong>Renderer:</strong> 3DGS + BVH Mesh</div>
        <div><strong>Coordinate:</strong> RUB (Y-Up)</div>
        <div><strong>Navigation:</strong> Theta* + B-Spline</div>
      </div>
    `;
  } catch (err) {
    console.error('Failed to load digital twin:', err);
    connStatus.style.background = '#ef4444';
    connText.textContent = 'Error';
    meshText.textContent = 'Mesh: Failed';
    splatText.textContent = 'Splats: Failed';
    sceneInfo.innerHTML = `
      <div style="color:#f87171;font-size:12px;">
        Failed to load scene. Ensure the backend is running at ${API_BASE_URL}
        and scene "${SCENE_ID}" has been processed.
      </div>
    `;
  }
}

main();