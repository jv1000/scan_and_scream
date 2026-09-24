# Spider v4: hairy tarantula, RIGGED for animation -> spider-v4.glb
# Same look as v3, but split into named nodes the page can move:
#   SpiderRig (root) > Torso > Abdomen, FangL, FangR ; SpiderRig > LegL0..LegL3, LegR0..LegR3
# Units: meters. Origin between the feet. +Y up, eyes face +Z (toward camera).
# Each leg node sits at its hip; in leg-local space +X points out along the leg,
# so rotating a leg about its local Z lifts the foot.
import numpy as np, trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals
rng = np.random.default_rng(7)

def mat(c, emissive=None, rough=0.9):
    m = PBRMaterial(baseColorFactor=c, roughnessFactor=rough, metallicFactor=0.0)
    if emissive is not None: m.emissiveFactor = emissive
    return m
MATS = {"body": mat([16, 11, 9, 255]), "bristle": mat([70, 48, 34, 255]),
        "band": mat([150, 95, 50, 255]), "fang": mat([8, 6, 6, 255], rough=0.25),
        "eye": mat([0, 0, 0, 255], emissive=[1.0, 0.0, 0.0], rough=0.3)}

def seg(p0, p1, r, sections=10):
    return trimesh.creation.cylinder(radius=r, segment=[np.asarray(p0, float), np.asarray(p1, float)], sections=sections)
def ball(p, r, sub=2, scale=None):
    b = trimesh.creation.icosphere(subdivisions=sub, radius=1.0)
    b.apply_scale(scale if scale is not None else r); b.apply_translation(p); return b
def perp(d):
    a = np.array([0, 1, 0]) if abs(d[1]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(d, a); u /= np.linalg.norm(u); return u, np.cross(d, u)
def hair_segment(bins, p0, p1, r, n, length=(0.003, 0.006)):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0; L = np.linalg.norm(d); d = d / L; u, v = perp(d)
    for _ in range(n):
        t = rng.uniform(0.05, 0.95); th = rng.uniform(0, 2*np.pi); out = np.cos(th)*u + np.sin(th)*v
        base = p0 + d*L*t + out*r*0.9
        tip = base + (out*0.8 + d*0.5 + rng.normal(0, 0.15, 3)) * rng.uniform(*length)
        bins["bristle"].append(seg(base, tip, 0.00035, 3))
def hair_ellipsoid(bins, radii, n, length=(0.004, 0.008), skip_front=False):
    for _ in range(n):
        dv = rng.normal(size=3); dv /= np.linalg.norm(dv)
        if skip_front and dv[2] > 0.5 and dv[1] > -0.2: continue
        p = dv*radii; nrm = dv/np.array(radii); nrm /= np.linalg.norm(nrm)
        tip = p + (nrm + np.array([0, 0, -0.4]) + rng.normal(0, 0.2, 3)) * rng.uniform(*length)
        bins["bristle"].append(seg(p, tip, 0.0004, 3))
def newbins(): return {k: [] for k in MATS}

scene = trimesh.Scene()
def T(p=(0, 0, 0), yaw=0.0):
    M = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0]); M[:3, 3] = p; return M
def node(name, parent, matrix):
    scene.graph.update(frame_from=parent, frame_to=name, matrix=matrix)
def attach(name, bins):
    for k, meshes in bins.items():
        if not meshes: continue
        m = trimesh.util.concatenate(meshes); m.visual = TextureVisuals(material=MATS[k])
        scene.add_geometry(m, node_name=f"{name}_{k}", geom_name=f"{name}_{k}", parent_node_name=name)

H = 0.045
node("SpiderRig", scene.graph.base_frame, T())

# Torso pivot = cephalothorax center
ce_c = np.array([0, H+0.002, 0.004]); ce_r = np.array([0.022, 0.015, 0.024])
node("Torso", "SpiderRig", T(ce_c))
b = newbins()
b["body"].append(ball([0, 0, 0], None, 3, ce_r)); hair_ellipsoid(b, ce_r, 220, (0.002, 0.004), skip_front=True)
for x, y, z, r in [(-0.0055, 0.009, 0.020, 0.0042), (0.0055, 0.009, 0.020, 0.0042),
                   (-0.0120, 0.007, 0.016, 0.0022), (0.0120, 0.007, 0.016, 0.0022),
                   (-0.0090, 0.012, 0.015, 0.0018), (0.0090, 0.012, 0.015, 0.0018),
                   (-0.0030, 0.013, 0.017, 0.0016), (0.0030, 0.013, 0.017, 0.0016)]:
    b["eye"].append(ball([x, y, z], r, 2))
fang_roots = {}
for s, nm in ((-1, "FangL"), (1, "FangR")):
    top = np.array([s*0.0065, -0.003, 0.020]); bot = np.array([s*0.0060, -0.015, 0.026])
    b["body"] += [seg(top, bot, 0.0048, 12), ball(bot, 0.0048, 2)]
    hair_segment(b, top, bot, 0.0048, 40, (0.002, 0.004))
    fang_roots[nm] = (s, bot)
    a = np.array([s*0.012, -0.004, 0.016]); m = a + [s*0.008, 0.010, 0.012]; c = m + [s*0.004, -0.018, 0.010]
    for p0, p1, r in ((a, m, 0.0022), (m, c, 0.0019)):
        b["body"].append(seg(p0, p1, r)); hair_segment(b, p0, p1, r, 35)
    b["body"].append(ball(m, 0.0024, 1))
attach("Torso", b)

# Fangs: pivot at the bottom of each jaw bulb
for nm, (s, bot) in fang_roots.items():
    node(nm, "Torso", T(bot))
    fb = newbins(); pts = [np.array([0, -0.003, 0.002])]
    for step in [(-s*0.0015, -0.004, 0.002), (-s*0.0020, -0.003, -0.001), (-s*0.0015, -0.001, -0.003)]:
        pts.append(pts[-1] + np.array(step))
    for k in range(3): fb["fang"].append(seg(pts[k], pts[k+1], [0.0016, 0.0011, 0.0006][k], 8))
    attach(nm, fb)

# Abdomen: pivot at its center, child of Torso so it rides along
ab_c = np.array([0, H+0.010, -0.042]); ab_r = np.array([0.030, 0.026, 0.037])
node("Abdomen", "Torso", T(ab_c - ce_c))
ab = newbins(); ab["body"].append(ball([0, 0, 0], None, 3, ab_r)); hair_ellipsoid(ab, ab_r, 650)
attach("Abdomen", ab)

# Legs: node at hip, yawed so local +X runs along the leg
angles = np.radians([55, 20, -15, -50])
for s, side in ((-1, "L"), (1, "R")):
    for k, a in enumerate(angles):
        hip = np.array([s*0.014, H, 0.004 + 0.008*np.sin(a)])
        yaw = np.arctan2(-np.sin(a), s*np.cos(a))
        name = f"Leg{side}{k}"; node(name, "SpiderRig", T(hip, yaw))
        reach = 0.088 if k in (0, 3) else 0.078
        knee = np.array([reach*0.36, 0.036, 0]); ankle = np.array([reach*0.80, -0.008, 0]); foot = np.array([reach*0.97, -H, 0])
        lb = newbins()
        for p0, p1, r, n in ((np.zeros(3), knee, 0.0034, 90), (knee, ankle, 0.0029, 90), (ankle, foot, 0.0020, 45)):
            lb["body"].append(seg(p0, p1, r)); hair_segment(lb, p0, p1, r, n)
        lb["band"] += [ball(knee, 0.0038, 2), ball(ankle, 0.0031, 2)]
        attach(name, lb)

scene.export("spider-v4.glb")
bb = scene.bounds
print("bounds:", np.round(bb, 3), "tris:", sum(len(g.faces) for g in scene.geometry.values()))
