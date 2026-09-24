# Spider v3: hairy tarantula-style spider -> spider-v3.glb
# Units: meters. Origin between the feet. +Y up, eyes face +Z (toward camera).
import numpy as np, trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals
rng = np.random.default_rng(7)

def mat(c, emissive=None, rough=0.9, metal=0.0):
    m = PBRMaterial(baseColorFactor=c, roughnessFactor=rough, metallicFactor=metal)
    if emissive is not None: m.emissiveFactor = emissive
    return m
MATS = {
  "body":    mat([16, 11, 9, 255]),
  "bristle": mat([70, 48, 34, 255]),          # lighter hairs catch the light
  "band":    mat([150, 95, 50, 255]),         # tarantula knee bands
  "fang":    mat([8, 6, 6, 255], rough=0.25), # glossy black fangs
  "eye":     mat([0, 0, 0, 255], emissive=[1.0, 0.0, 0.0], rough=0.3),
}
bins = {k: [] for k in MATS}

def seg(p0, p1, r, sections=10):
    return trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=sections)
def ball(p, r, sub=2, scale=None):
    b = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    b.apply_scale(scale if scale is not None else r); b.apply_translation(p); return b
def perp(d):
    a = np.array([0, 1, 0]) if abs(d[1]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u); return u, np.cross(d, u)

def hair_segment(p0, p1, r, n, length=(0.003, 0.006)):
    d = p1 - p0; L = np.linalg.norm(d); d = d / L; u, v = perp(d)
    for _ in range(n):
        t = rng.uniform(0.05, 0.95); th = rng.uniform(0, 2*np.pi)
        out = np.cos(th)*u + np.sin(th)*v
        base = p0 + d*L*t + out*r*0.9
        tip = base + (out*0.8 + d*0.5 + rng.normal(0, 0.15, 3)) * rng.uniform(*length)
        bins["bristle"].append(seg(base, tip, 0.00035, sections=3))

def hair_ellipsoid(center, radii, n, length=(0.004, 0.008), skip_front=False):
    for _ in range(n):
        dvec = rng.normal(size=3); dvec /= np.linalg.norm(dvec)
        if skip_front and dvec[2] > 0.5 and dvec[1] > -0.2: continue  # keep eyes clear
        p = center + dvec*radii
        nrm = dvec/np.array(radii); nrm /= np.linalg.norm(nrm)
        tip = p + (nrm + np.array([0, 0, -0.4]) + rng.normal(0, 0.2, 3)) * rng.uniform(*length)
        bins["bristle"].append(seg(p, tip, 0.0004, sections=3))

H = 0.045
# abdomen and cephalothorax
ab_c, ab_r = np.array([0, H+0.010, -0.042]), np.array([0.030, 0.026, 0.037])
ce_c, ce_r = np.array([0, H+0.002, 0.004]), np.array([0.022, 0.015, 0.024])
bins["body"] += [ball(ab_c, None, 3, ab_r), ball(ce_c, None, 3, ce_r)]
hair_ellipsoid(ab_c, ab_r, 650)
hair_ellipsoid(ce_c, ce_r, 220, length=(0.002, 0.004), skip_front=True)

# eyes: two big forward, six small around them
eyes = [(-0.0055, 0.011, 0.024, 0.0042), (0.0055, 0.011, 0.024, 0.0042),
        (-0.0120, 0.009, 0.020, 0.0022), (0.0120, 0.009, 0.020, 0.0022),
        (-0.0090, 0.014, 0.019, 0.0018), (0.0090, 0.014, 0.019, 0.0018),
        (-0.0030, 0.015, 0.021, 0.0016), (0.0030, 0.015, 0.021, 0.0016)]
for x, y, z, r in eyes:
    bins["eye"].append(ball(np.array([x, H+y, z]), r, 2))

# chelicerae (hairy jaw bulbs) with curved glossy fangs
for s in (-1, 1):
    top = np.array([s*0.0065, H-0.001, 0.024]); bot = np.array([s*0.0060, H-0.013, 0.030])
    bins["body"].append(seg(top, bot, 0.0048, 12)); bins["body"].append(ball(bot, 0.0048, 2))
    hair_segment(top, bot, 0.0048, 40, length=(0.002, 0.004))
    pts = [bot + np.array([0, -0.003, 0.002])]
    for k, step in enumerate([(-s*0.0015, -0.004, 0.002), (-s*0.0020, -0.003, -0.001), (-s*0.0015, -0.001, -0.003)]):
        pts.append(pts[-1] + np.array(step))
    for k in range(3):
        bins["fang"].append(seg(pts[k], pts[k+1], [0.0016, 0.0011, 0.0006][k], 8))

# pedipalps: short hairy feelers beside the fangs
for s in (-1, 1):
    a = np.array([s*0.012, H-0.002, 0.020]); b = a + np.array([s*0.008, 0.010, 0.012]); c = b + np.array([s*0.004, -0.018, 0.010])
    for p0, p1, r in ((a, b, 0.0022), (b, c, 0.0019)):
        bins["body"].append(seg(p0, p1, r)); hair_segment(p0, p1, r, 35)
    bins["body"].append(ball(b, 0.0024, 1))

# legs: 4 per side, 3 segments, arched knee, banded joints, hairy
angles = np.radians([55, 20, -15, -50])
for s in (-1, 1):
    for k, a in enumerate(angles):
        hip = np.array([s*0.014, H, 0.004 + 0.008*np.sin(a)])
        dv = np.array([s*np.cos(a), 0, np.sin(a)]); reach = 0.088 if k in (0, 3) else 0.078
        knee  = hip + dv*reach*0.36 + np.array([0, 0.036, 0])
        ankle = hip + dv*reach*0.80 + np.array([0, -0.008, 0])
        foot  = hip + dv*reach*0.97 + np.array([0, -H, 0])
        for p0, p1, r, n in ((hip, knee, 0.0034, 90), (knee, ankle, 0.0029, 90), (ankle, foot, 0.0020, 45)):
            bins["body"].append(seg(p0, p1, r)); hair_segment(p0, p1, r, n)
        for p, r in ((knee, 0.0038), (ankle, 0.0031)):
            bins["band"].append(ball(p, r, 2))

scene = trimesh.Scene()
for name, meshes in bins.items():
    m = trimesh.util.concatenate(meshes); m.visual = TextureVisuals(material=MATS[name])
    scene.add_geometry(m, node_name=name, geom_name=name)
scene.export("spider-v3.glb")
b = scene.bounds
print("bounds:", np.round(b, 3), "tris:", sum(len(g.faces) for g in scene.geometry.values()))
