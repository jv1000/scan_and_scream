# Builds a low-poly cartoon-creepy spider as spider.glb
# Units: meters. Origin: between the feet. +Y up, eyes face +Z (toward camera).
import numpy as np, trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals

def mat(color, emissive=None, rough=0.85):
    m = PBRMaterial(baseColorFactor=color, roughnessFactor=rough, metallicFactor=0.0)
    if emissive is not None:
        m.emissiveFactor = emissive
    return m

BODY = mat([12, 9, 9, 255], rough=0.95)
LEG  = mat([14, 10, 10, 255], rough=0.95)
EYE  = mat([0, 0, 0, 255], emissive=[1.0, 0.0, 0.0], rough=0.3)

parts = []
def add(mesh, material, name):
    mesh.visual = TextureVisuals(material=material)
    parts.append((name, mesh))

H = 0.045  # body height above feet
# abdomen (behind, -Z), stretched ellipsoid
ab = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
ab.apply_scale([0.030, 0.026, 0.038]); ab.apply_translation([0, H + 0.008, -0.040])
add(ab, BODY, "abdomen")
# cephalothorax (front)
ce = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
ce.apply_scale([0.021, 0.017, 0.023]); ce.apply_translation([0, H, 0.004])
add(ce, BODY, "cephalothorax")

# eyes: two big glowing, two small glowing
for i, (x, y, z, r) in enumerate([(-0.007, H+0.010, 0.024, 0.0050), (0.007, H+0.010, 0.024, 0.0050),
                                  (-0.013, H+0.007, 0.019, 0.0030), (0.013, H+0.007, 0.019, 0.0030)]):
    e = trimesh.creation.icosphere(subdivisions=2, radius=r); e.apply_translation([x, y, z])
    add(e, EYE, f"eye_{i}")

# legs: 4 per side, knee up and out, foot down at y=0
def seg(p0, p1, r):
    return trimesh.creation.cylinder(radius=r, segment=[p0, p1], sections=8)

angles = np.radians([55, 20, -15, -50])  # fore-to-aft spread
for side in (-1, 1):
    for k, a in enumerate(angles):
        hip   = np.array([side*0.012, H, 0.004 + 0.006*np.sin(a)])
        dirv  = np.array([side*np.cos(a), 0, np.sin(a)])
        reach = 0.085 if k in (0, 3) else 0.075
        knee  = hip + dirv*reach*0.35 + np.array([0, 0.035, 0])   # high arched knee
        ankle = hip + dirv*reach*0.80 + np.array([0, -0.010, 0])
        foot  = hip + dirv*reach*0.95 + np.array([0, -H, 0])        # tip curls down
        add(seg(hip, knee, 0.0024), LEG, f"leg_{side}_{k}_femur")
        add(seg(knee, ankle, 0.0019), LEG, f"leg_{side}_{k}_tibia")
        add(seg(ankle, foot, 0.0012), LEG, f"leg_{side}_{k}_tarsus")
        for n, p, r in (("knee", knee, 0.0030), ("ankle", ankle, 0.0022)):
            j = trimesh.creation.icosphere(subdivisions=1, radius=r); j.apply_translation(p)
            add(j, LEG, f"{n}_{side}_{k}")

scene = trimesh.Scene()
for name, m in parts:
    scene.add_geometry(m, node_name=name, geom_name=name)
scene.export("spider.glb")
b = scene.bounds
print("bounds (m):", np.round(b, 3), "size:", np.round(b[1]-b[0], 3))
