import base64
import json
from dataclasses import replace
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.navigation.pathplanner import IndoorSpatialPathPlanner
from backend.semantics.spatial_poi_indexer import VoxelSemanticField, SpatialPOI

app = FastAPI(
    title="Intelligent 3D Spatial Engine API",
    version="1.0.0",
    description="Dual-Representation 3DGS + Collision Mesh Spatial Twin Backend",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_ROOT = Path("./data/scenes")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/scenes", StaticFiles(directory=str(DATA_ROOT)), name="scenes")

# In-memory spatial caches per loaded scene
LOADED_PLANNERS: dict[str, IndoorSpatialPathPlanner] = {}
LOADED_SEMANTICS: dict[str, VoxelSemanticField] = {}
LOADED_POIS: dict[str, List[SpatialPOI]] = {}


class PathRequest(BaseModel):
    scene_id: str
    start_xyz: List[float] = Field(..., min_length=3, max_length=3)
    goal_xyz: List[float] = Field(..., min_length=3, max_length=3)


class CalibrationRequest(BaseModel):
    measured_units: float = Field(..., gt=0)   # distance measured in the viewer (scene units)
    real_metres: float = Field(..., gt=0)      # true length of the same segment in metres


class SemanticSearchRequest(BaseModel):
    scene_id: str
    query_text: str
    query_embedding: Optional[List[float]] = None
    similarity_threshold: float = 0.25
    top_k: int = 5


@app.post("/api/v1/scenes/{scene_id}/initialize")
def initialize_scene_engine(scene_id: str):
    scene_dir = DATA_ROOT / scene_id / "artifacts"
    mesh_path = scene_dir / "collision_mesh.glb"
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail=f"Collision mesh not found at {mesh_path}")

    loaded = trimesh.load(str(mesh_path), force="mesh")
    planner = IndoorSpatialPathPlanner(
        cell_size_m=0.05, agent_radius_m=0.28, step_height_m=0.16, agent_height_m=1.80
    )
    planner.build_navgrid_from_mesh(loaded)
    LOADED_PLANNERS[scene_id] = planner

    sem_field = VoxelSemanticField(voxel_size_m=0.06)
    LOADED_SEMANTICS[scene_id] = sem_field

    poi_file = scene_dir / "poi_database.json"
    LOADED_POIS[scene_id] = []
    if poi_file.exists():
        LOADED_POIS[scene_id] = [
            SpatialPOI(embedding=[], confidence=1.0, **{k: v for k, v in p.items() if k not in ("embedding", "confidence")})
            for p in json.loads(poi_file.read_text(encoding="utf-8"))
        ]

    return {
        "status": "initialized",
        "scene_id": scene_id,
        "grid_dimensions": planner.grid_shape,
        "walkable_area_m2": round(
            float(np.sum(planner.walkable_mask)) * (planner.cell_size_m ** 2), 2
        ),
    }


@app.post("/api/v1/navigation/plan-path")
def compute_optimal_path(req: PathRequest):
    planner = LOADED_PLANNERS.get(req.scene_id)
    if planner is None:
        raise HTTPException(
            status_code=400, detail=f"Scene '{req.scene_id}' not initialized."
        )

    try:
        result = planner.plan_path(
            start_xyz=(req.start_xyz[0], req.start_xyz[1], req.start_xyz[2]),
            goal_xyz=(req.goal_xyz[0], req.goal_xyz[1], req.goal_xyz[2]),
        )
        return {
            "scene_id": req.scene_id,
            "waypoints_3d": result.waypoints_3d,
            "raw_control_points_3d": result.raw_control_points_3d,
            "total_length_m": result.total_length_m,
            "min_obstacle_clearance_m": result.min_obstacle_clearance_m,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/v1/scenes/{scene_id}/map")
def get_map(scene_id: str):
    planner = LOADED_PLANNERS.get(scene_id)
    if planner is None:
        raise HTTPException(status_code=400, detail=f"Scene '{scene_id}' not initialized.")
    status = planner.status_grid
    cap = []
    cam_file = DATA_ROOT / scene_id / "artifacts" / "cameras.json"
    if cam_file.exists():
        for c in json.loads(cam_file.read_text(encoding="utf-8"))["centers"]:
            ix, iz = planner.world_to_grid(c[0], c[2])
            cap.append([ix, iz])
    return {
        "nx": int(status.shape[0]), "nz": int(status.shape[1]),
        "origin_xz": [float(planner.origin_xz[0]), float(planner.origin_xz[1])],
        "cell_size_m": planner.cell_size_m,
        # column-major (ix outer, iz inner), one byte per cell, base64 encoded
        "status_b64": base64.b64encode(status.tobytes(order="C")).decode("ascii"),
        "capture_cells": cap,
    }


@app.get("/api/v1/scenes/{scene_id}/calibration")
def get_calibration(scene_id: str):
    f = DATA_ROOT / scene_id / "artifacts" / "scale_calibration.json"
    if not f.exists():
        return {"calibrated": False, "scale_factor": 1.0}
    return {"calibrated": True, **json.loads(f.read_text(encoding="utf-8"))}


@app.post("/api/v1/scenes/{scene_id}/calibration")
def set_calibration(scene_id: str, req: CalibrationRequest):
    scene_dir = DATA_ROOT / scene_id
    if not (scene_dir / "artifacts").exists():
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    # factor converts scene units to metres; applied to measurements, not to geometry
    data = {"scale_factor": req.real_metres / req.measured_units,
            "measured_units": req.measured_units, "real_metres": req.real_metres}
    (scene_dir / "artifacts" / "scale_calibration.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"calibrated": True, **data}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/semantics/search", response_model=List[SpatialPOI])
def search_spatial_pois(req: SemanticSearchRequest):
    sem_field = LOADED_SEMANTICS.get(req.scene_id)
    if sem_field is None:
        raise HTTPException(
            status_code=400, detail=f"Semantic index for '{req.scene_id}' not initialized."
        )

    # Embedding query against the fused voxel field (CLIP pipeline), if the client supplies one.
    if req.query_embedding is not None:
        return sem_field.query_open_vocabulary(
            text_embedding=np.array(req.query_embedding, dtype=np.float32),
            query_label=req.query_text,
            similarity_threshold=req.similarity_threshold,
            top_k=req.top_k,
        )

    # Text query: match against the scene's labelled POI database (deterministic, no fake embeddings).
    tokens = [t for t in req.query_text.lower().split() if t]
    scored = []
    for poi in LOADED_POIS.get(req.scene_id, []):
        label = poi.label.lower()
        hits = sum(t in label for t in tokens)
        if hits:
            scored.append((hits / len(tokens), poi))
    scored.sort(key=lambda x: -x[0])
    return [replace(p, confidence=round(sc, 3)) for sc, p in scored[: req.top_k]]
