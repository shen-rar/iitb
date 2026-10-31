"""
Compact 3D Gaussian Splatting trainer in pure PyTorch (no CUDA extensions to compile).

Tile-based differentiable rasteriser: per-tile gaussians are depth sorted and alpha composited.
Colour is SH degree 0. Trains directly in the gravity-aligned metric frame produced by
MetricGravityAligner so the .ply drops into the viewer next to points.ply / collision_mesh.glb.
Slower and lower quality than the reference CUDA implementation; it runs anywhere torch does.
"""
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

TILE = 16
SH_C0 = 0.28209479177387814


def load_colmap_views(rec, image_dir: Path, sim3: np.ndarray, width: int):
    """Undistort + downscale images; return views (image, w2c in aligned frame) and intrinsics."""
    S = sim3[:3, :3]
    s = float(np.cbrt(abs(np.linalg.det(S))))
    Rg = S / s
    views = []
    K = None
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        cam = rec.cameras[img.camera_id]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        k1 = cam.params[3] if len(cam.params) > 3 else 0.0
        raw = cv2.imread(str(image_dir / img.name))
        if raw is None:
            continue
        Km = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        und = cv2.undistort(raw, Km, np.array([k1, 0, 0, 0], dtype=np.float64))
        scale = width / und.shape[1]
        und = cv2.resize(und, (width, int(round(und.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        K = (f * scale, f * scale, cx * scale, cy * scale)
        c2w = img.cam_from_world().inverse().matrix()  # 3x4, colmap frame
        R_c2w = Rg @ c2w[:, :3]
        center = S @ c2w[:, 3] + sim3[:3, 3]
        w2c = np.eye(4)
        w2c[:3, :3] = R_c2w.T
        w2c[:3, 3] = -R_c2w.T @ center
        views.append({
            "img": torch.from_numpy(cv2.cvtColor(und, cv2.COLOR_BGR2RGB)).float() / 255.0,
            "w2c": torch.from_numpy(w2c).float(),
        })
    return views, K


def quat_to_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


class Splats(torch.nn.Module):
    def __init__(self, xyz, rgb, init_scale):
        super().__init__()
        n = len(xyz)
        self.means = torch.nn.Parameter(xyz.clone())
        self.log_scales = torch.nn.Parameter(torch.log(init_scale.clamp(min=1e-3))[:, None].repeat(1, 3))
        q = torch.zeros(n, 4, device=xyz.device)
        q[:, 0] = 1
        self.quats = torch.nn.Parameter(q)
        self.opac = torch.nn.Parameter(torch.full((n,), -1.0, device=xyz.device))
        self.colors = torch.nn.Parameter(torch.logit(rgb.clamp(0.02, 0.98)))

    def n(self):
        return self.means.shape[0]


def render(sp: Splats, w2c, K, H, W, bg=0.0, max_per_tile=1024):
    fx, fy, cx, cy = K
    p_cam = sp.means @ w2c[:3, :3].T + w2c[:3, 3]
    z = p_cam[:, 2]
    zc = z.clamp(min=0.05)
    u = fx * p_cam[:, 0] / zc + cx
    v = fy * p_cam[:, 1] / zc + cy
    R = quat_to_rot(sp.quats)
    S = torch.exp(sp.log_scales)
    M = R * S[:, None, :]
    cov3 = M @ M.transpose(1, 2)
    Rw = w2c[:3, :3]
    cov_cam = Rw @ cov3 @ Rw.T
    J = torch.zeros(sp.n(), 2, 3, device=z.device)
    J[:, 0, 0] = fx / zc
    J[:, 0, 2] = -fx * p_cam[:, 0] / zc ** 2
    J[:, 1, 1] = fy / zc
    J[:, 1, 2] = -fy * p_cam[:, 1] / zc ** 2
    cov2 = J @ cov_cam @ J.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = (a * c - b * b).clamp(min=1e-6)
    ia, ib, ic = c / det, -b / det, a / det
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    rad = 3.0 * torch.sqrt(lam)
    opac = torch.sigmoid(sp.opac)
    col = torch.sigmoid(sp.colors)

    valid = (z > 0.05) & (u + rad > 0) & (u - rad < W) & (v + rad > 0) & (v - rad < H) & (rad < 0.6 * W)
    order = torch.argsort(torch.where(valid, z, torch.full_like(z, 1e9)))[: int(valid.sum())]
    u, v, rad = u[order], v[order], rad[order]
    ia, ib, ic, opac, col = ia[order], ib[order], ic[order], opac[order], col[order]

    dev = z.device
    ty, tx = math.ceil(H / TILE), math.ceil(W / TILE)
    n = u.shape[0]
    if n == 0:
        return torch.full((H, W, 3), bg, device=dev)
    x0 = torch.arange(tx, device=dev) * TILE
    y0 = torch.arange(ty, device=dev) * TILE
    ov_x = (u[None, :] + rad[None, :] >= x0[:, None]) & (u[None, :] - rad[None, :] <= (x0[:, None] + TILE))
    ov_y = (v[None, :] + rad[None, :] >= y0[:, None]) & (v[None, :] - rad[None, :] <= (y0[:, None] + TILE))

    # All tiles in one batched pass (chunked to bound memory) instead of a Python loop per tile.
    # Gaussians are depth sorted, so a stable sort of "not in tile" keeps each tile's first K
    # hits front to back.
    py, px = torch.meshgrid(torch.arange(TILE, device=dev), torch.arange(TILE, device=dev), indexing="ij")
    py, px = py.reshape(-1).float() + 0.5, px.reshape(-1).float() + 0.5
    tile_y = y0[:, None].expand(ty, tx).reshape(-1).float()
    tile_x = x0[None, :].expand(ty, tx).reshape(-1).float()
    out = []
    for s in range(0, ty * tx, 64):
        tj = torch.arange(s, min(s + 64, ty * tx), device=dev)
        mask = ov_y[tj // tx] & ov_x[tj % tx]                                    # [t, n]
        # pad only to the busiest tile in this chunk; a low cap truncates tiles where large
        # near-camera gaussians fill the slots, which shows up as blocky tile edges
        k = max(1, min(max_per_tile, int(mask.sum(1).max())))
        idx = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)[:, :k]  # [t, k]
        hit = mask.gather(1, idx).float()
        X = tile_x[tj, None] + px[None, :]                                         # [t, P]
        Y = tile_y[tj, None] + py[None, :]
        args = (u, v, ia, ib, ic, opac, col, idx, hit, X, Y, bg)
        if torch.is_grad_enabled():
            # recompute each chunk in backward instead of keeping every [t, P, k] intermediate:
            # memory stays bounded by one chunk, which avoids spilling out of GPU memory
            out.append(checkpoint(_shade_tiles, *args, use_reentrant=False))
        else:
            out.append(_shade_tiles(*args))
    img = torch.cat(out, 0).reshape(ty, tx, TILE, TILE, 3).permute(0, 2, 1, 3, 4).reshape(ty * TILE, tx * TILE, 3)
    return img[:H, :W]


def _shade_tiles(u, v, ia, ib, ic, opac, col, idx, hit, X, Y, bg):
    """Alpha-composite the depth-sorted gaussians idx [t, k] over pixel centres X, Y [t, P]."""
    dx = X[:, :, None] - u[idx][:, None, :]                                       # [t, P, k]
    dy = Y[:, :, None] - v[idx][:, None, :]
    pw = -0.5 * (ia[idx][:, None] * dx * dx + 2 * ib[idx][:, None] * dx * dy + ic[idx][:, None] * dy * dy)
    alpha = (opac[idx][:, None] * torch.exp(pw.clamp(max=0))).clamp(max=0.99) * hit[:, None]
    alpha = torch.where(alpha < 1 / 255, torch.zeros_like(alpha), alpha)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1 - alpha[..., :-1]], -1), -1)
    wgt = alpha * T
    return torch.einsum("tpk,tkc->tpc", wgt, col[idx]) + (1 - wgt.sum(-1, keepdim=True)) * bg


def save_ply(sp: Splats, path: Path):
    m = sp.means.detach().cpu().numpy()
    dc = ((torch.sigmoid(sp.colors) - 0.5) / SH_C0).detach().cpu().numpy()
    op = sp.opac.detach().cpu().numpy()[:, None]
    sc = sp.log_scales.detach().cpu().numpy()
    q = torch.nn.functional.normalize(sp.quats, dim=-1).detach().cpu().numpy()
    n = len(m)
    data = np.concatenate([m, np.zeros((n, 3)), dc, op, sc, q], 1).astype("<f4")
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    hdr += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(data.tobytes())


def _densify(sp, gacc, gcnt, size_thresh):
    """For the 10% of gaussians with the largest mean positional gradient: clone the small ones,
    split the large ones into two smaller children sampled inside the parent (as in 3DGS).
    Transparent gaussians are pruned. size_thresh is relative to the scene, so large rooms
    densify too."""
    avg = gacc / gcnt.clamp(min=1)
    sel = avg > torch.quantile(avg, 0.9)
    with torch.no_grad():
        scales = torch.exp(sp.log_scales)
        big = scales.max(1).values > size_thresh
        clone, split = sel & ~big, sel & big
        # prune transparent and oversized gaussians: huge blobs smear across near views
        oversized = scales.max(1).values > 10 * size_thresh
        keep = (torch.sigmoid(sp.opac) > 0.02) & ~split & ~oversized
        rot = quat_to_rot(sp.quats[split])
        for name in ("means", "log_scales", "quats", "opac", "colors"):
            p = getattr(sp, name).detach()
            parts = [p[keep], p[clone]]
            if name == "means":
                parts[1] = p[clone] + torch.randn_like(p[clone]) * 0.01
                for _ in range(2):
                    offset = (rot @ (torch.randn_like(scales[split]) * scales[split])[..., None]).squeeze(-1)
                    parts.append(p[split] + offset)
            elif name == "log_scales":
                parts += [p[split] - math.log(1.6)] * 2
            else:
                parts += [p[split]] * 2
            setattr(sp, name, torch.nn.Parameter(torch.cat(parts)))


def train(views, K, xyz, rgb, out_path: Path, iters=2500, device="cuda", log=print, max_gauss=60000):
    dev = torch.device(device)
    H, W = views[0]["img"].shape[:2]
    pts = torch.from_numpy(xyz).float().to(dev)
    cols = torch.from_numpy(rgb).float().to(dev) / 255.0
    # initial scale = mean distance to 3 nearest neighbours; a KD-tree avoids an N x N matrix
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(xyz).query(xyz, k=4)  # first neighbour is the point itself
    knn = torch.from_numpy(dist[:, 1:].mean(1)).float().to(dev)
    sp = Splats(pts, cols, knn).to(dev)
    size_thresh = 0.01 * float(np.linalg.norm(np.ptp(xyz, axis=0)))  # 1% of the scene diagonal
    lrs = {"means": 8e-4, "log_scales": 5e-3, "quats": 1e-3, "opac": 5e-2, "colors": 2.5e-2}

    def make_opt():
        return torch.optim.Adam([{"params": [getattr(sp, k)], "lr": v} for k, v in lrs.items()])

    opt = make_opt()
    hold = list(range(0, len(views), 8))
    tr = [i for i in range(len(views)) if i not in hold]
    gacc = torch.zeros(sp.n(), device=dev)
    gcnt = torch.zeros(sp.n(), device=dev)
    t0 = time.time()
    for it in range(1, iters + 1):
        vw = views[tr[np.random.randint(len(tr))]]
        img = render(sp, vw["w2c"].to(dev), K, H, W)
        gt = vw["img"].to(dev)
        loss = 0.8 * (img - gt).abs().mean() + 0.8 * ((img - gt) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if sp.means.grad is not None and it < iters * 0.6:
            g = sp.means.grad.norm(dim=-1)
            gacc += g
            gcnt += (g > 0).float()
        opt.step()
        if 200 <= it < iters * 0.6 and it % 100 == 0 and sp.n() < max_gauss:
            _densify(sp, gacc, gcnt, size_thresh)
            opt = make_opt()
            gacc = torch.zeros(sp.n(), device=dev)
            gcnt = torch.zeros(sp.n(), device=dev)
        if it % 100 == 0:
            log(f"it {it}/{iters} loss {loss.item():.4f} gaussians {sp.n()} {time.time() - t0:.0f}s")
    with torch.no_grad():
        ps = []
        for i in hold:
            im = render(sp, views[i]["w2c"].to(dev), K, H, W)
            mse = ((im - views[i]["img"].to(dev)) ** 2).mean().item()
            ps.append(-10 * math.log10(max(mse, 1e-8)))
    psnr = float(np.mean(ps)) if ps else float("nan")
    log(f"held-out PSNR {psnr:.2f} dB over {len(ps)} views")
    save_ply(sp, out_path)
    return psnr
