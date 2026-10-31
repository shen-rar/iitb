# Spatial Engine - Namma Space Round 1

Video -> gravity-aligned, metric 3D scene -> web viewer with navigation and POI search.

## Pipeline (SOP)
1. **Ingest** - `backend/ingestion/frame_selector.py`: blur rejection (Laplacian variance) + optical-flow parallax gating -> keyframes.
2. **SfM** - `backend/reconstruction/pycolmap_pipeline.py`: COLMAP (BSD) via `pycolmap`: SIFT features, sequential matching, incremental mapping.
3. **Align** - `gravity_aligner.py`: RANSAC floor plane -> Y-up, floor at Y=0; metric scale from assumed camera height (1.45 m) or ARKit poses if supplied.
4. **Export** - `points.ply` (visual layer), `collision_mesh.glb` (invisible BVH layer), `cameras.json`, `scene_report.json`. A 3DGS `visual_splats.ksplat` placed in `artifacts/` is picked up automatically by the viewer.
5. **Serve** - FastAPI (nav grid + Theta* path planning + POI search) and a Three.js viewer.

All dependencies are open source / commercially usable (no SuperPoint/SuperGlue).

## Setup
```bash
pip install numpy scipy opencv-python-headless trimesh fastapi "uvicorn[standard]" pydantic pycolmap pytest httpx
cd frontend && npm install && cd ..
```

## One-command build
```bash
python build_scene.py my_video.mp4 my_scene              # geometry + splats (GPU)
python build_scene.py my_video.mp4 my_scene --no-splats  # geometry only, no GPU needed
```
Then start the backend and frontend as below and open `/?scene=my_scene`.

## Run
```bash
python create_demo_scene.py                       # sample scene (data/scenes/demo_office)
python run_pipeline.py my_video.mp4 my_scene      # reconstruct your own video
python -m uvicorn backend.main:app --port 8000    # backend
cd frontend && npm run dev                        # viewer: http://localhost:5173/?scene=my_scene
```
On Windows set `PYTHONUTF8=1` before running tests. Tests: `python -m pytest tests -v`.

## Viewer (Field System)
Layout modelled on the renderify field viewer: plan map, real-frame panel, enter-the-space gate and a sandbox.

- **Plan map (bottom right / top right):** live map from the backend's per-cell status. Open, blocked (obstacle or too close to a wall), unseen (no floor observed), capture path, route and placed props. Click the map to route to that spot.
- **Sandbox (G):** place cone, barrier, crate or sign (keys 1-4), click to place, R rotates, Delete removes the prop under the reticle, T routes to the last prop. Placement is checked against the map: blocked and unseen cells refuse with a reason. Layouts persist per scene in this browser and "Copy link" shares them in the URL (`layout=`).
- **Real frame:** a captured frame saved by the build as `artifacts/reference.jpg`.
- **Status stamp:** LIVE when splats or a point cloud are loaded, OFFLINE otherwise.

| Action | Control |
|---|---|
| Orbit / pan / zoom | mouse |
| Walk first person (wall collision) | **F**, then WASD + mouse, Shift sprints, Esc exits |
| Auto-navigate to a spot | double-click the floor |
| POI search + fly-to | type a label, e.g. "espresso machine" |
| Plan (top-down) view | **M** |
| Toggle points / mesh | **P** |
| Measure distance | **U**, click two points; the first measurement asks for a real length to calibrate the scale |
| Share a viewpoint | "Copy link" (adds `cam=x,y,z`) |

Set `VITE_API_BASE_URL` for a remote backend. Viewing works without the backend; navigation and search need it.

## Training splats (GPU)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
python train_splats.py <colmap_model_dir> <image_dir> <scene_id> --iters 2500
```
Writes `artifacts/visual_splats.ply`, which the viewer prefers over the point cloud. The trainer is pure PyTorch (no compiler needed); use the reference gsplat/3DGS on Linux for best quality.

## Sample data
`tests/data/library_sample/` holds 30 frames of a public indoor video (CC BY-SA 4.0, see ATTRIBUTION.txt).

## Known limits
- Scale is assumed (camera height) until calibrated. Calibration stores a factor (`artifacts/scale_calibration.json`) applied to measurements only; it does not rescale the geometry.
- Sparse SfM point cloud only; train 3DGS (e.g. gsplat, Apache-2.0) on a GPU for photoreal output.
- POI labels are supplied in `poi_database.json`; CLIP embedding fusion exists in `semantics/` but is not wired to an auto-labeller.
