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


class PathRequest(BaseModel):
    scene_id: str
    start_xyz: List[float] = Field(..., min_length=3, max_length=3)
    goal_xyz: List[float] = Field(..., min_length=3, max_length=3)


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


@app.post("/api/v1/semantics/search", response_model=List[SpatialPOI])
def search_spatial_pois(req: SemanticSearchRequest):
    sem_field = LOADED_SEMANTICS.get(req.scene_id)
    if sem_field is None:
        raise HTTPException(
            status_code=400, detail=f"Semantic index for '{req.scene_id}' not initialized."
        )

    if req.query_embedding is not None:
        q_emb = np.array(req.query_embedding, dtype=np.float32)
    else:
        # Deterministic fallback hash embedding if client requests text-only testing
        rng = np.random.default_rng(abs(hash(req.query_text.lower().strip())) % (2**32))
        q_emb = rng.standard_normal(512).astype(np.float32)

    return sem_field.query_open_vocabulary(
        text_embedding=q_emb,
        query_label=req.query_text,
        similarity_threshold=req.similarity_threshold,
        top_k=req.top_k,
    )