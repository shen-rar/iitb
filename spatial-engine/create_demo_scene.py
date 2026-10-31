#!/usr/bin/env python
"""Create a minimal demo office scene with a collision mesh."""

import trimesh
import numpy as np
from pathlib import Path

# Create output directory
scene_dir = Path("./data/scenes/demo_office/artifacts")
scene_dir.mkdir(parents=True, exist_ok=True)

# Create a simple office room as a mesh
# Box dimensions: 10m x 8m x 3m (length x width x height)
vertices = np.array([
    # Floor corners
    [0, 0, 0],
    [10, 0, 0],
    [10, 8, 0],
    [0, 8, 0],
    # Ceiling corners
    [0, 0, 3],
    [10, 0, 3],
    [10, 8, 3],
    [0, 8, 3],
])

# Define faces (triangles) for a box
faces = np.array([
    # Floor (2 triangles)
    [0, 1, 2], [0, 2, 3],
    # Ceiling (2 triangles)
    [4, 6, 5], [4, 7, 6],
    # Front wall (2 triangles)
    [0, 5, 1], [0, 4, 5],
    # Back wall (2 triangles)
    [2, 7, 3], [2, 6, 7],
    # Left wall (2 triangles)
    [0, 3, 7], [0, 7, 4],
    # Right wall (2 triangles)
    [1, 5, 6], [1, 6, 2],
])

# Create mesh
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Export as GLB (binary glTF)
output_path = scene_dir / "collision_mesh.glb"
mesh.export(str(output_path))
print(f"Created demo scene collision mesh at {output_path}")
print(f"  - Room dimensions: 10m x 8m x 3m")
print(f"  - Mesh vertices: {len(vertices)}")
print(f"  - Mesh faces: {len(faces)}")
