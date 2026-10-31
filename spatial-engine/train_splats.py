#!/usr/bin/env python
"""python train_splats.py <colmap_model_dir> <image_dir> <scene_id> [--iters 2500] [--width 320]

Trains gaussians in the gravity-aligned metric frame and writes artifacts/visual_splats.ply.
"""
import argparse
from pathlib import Path

import pycolmap
import trimesh

from backend.config import DATA_ROOT
from backend.reconstruction.gravity_aligner import MetricGravityAligner
from backend.reconstruction.pycolmap_pipeline import extract_geometry
from backend.reconstruction.splat_trainer import load_colmap_views, train


def run(model_dir, image_dir, scene_id, iters=2500, width=320, device="cuda"):
    rec = pycolmap.Reconstruction(str(model_dir))
    pts, cols, cen, rots = extract_geometry(rec)
    sim3, _ = MetricGravityAligner(1.45).compute_canonical_sim3(trimesh.Trimesh(vertices=pts), cen, rots)
    views, K = load_colmap_views(rec, Path(image_dir), sim3, width)
    xyz = (sim3[:3, :3] @ pts.T).T + sim3[:3, 3]
    out = DATA_ROOT / scene_id / "artifacts" / "visual_splats.ply"
    print("views", len(views), "points", len(xyz), "K", K, flush=True)
    psnr = train(views, K, xyz, cols, out, iters=iters, device=device, log=lambda m: print(m, flush=True))
    print("wrote", out, flush=True)
    return psnr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images")
    ap.add_argument("scene_id")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run(a.model, a.images, a.scene_id, a.iters, a.width, a.device)
