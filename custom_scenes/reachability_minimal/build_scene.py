from pathlib import Path

import mujoco


output = Path("scene_generated.xml")


spec = mujoco.MjSpec()

spec.modelname = "reachability_minimal"


# ------------------------------------------------
# floor
# ------------------------------------------------

floor = spec.worldbody.add_geom()

floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [5, 5, 0.1]
floor.rgba = [0.8, 0.8, 0.8, 1]


# ------------------------------------------------
# table
# ------------------------------------------------

table = spec.worldbody.add_body(
    name="table",
    pos=[0, 0, 0],
)

table_top = table.add_geom()

table_top.name = "table_top"
table_top.type = mujoco.mjtGeom.mjGEOM_BOX
table_top.pos = [0, 0, 0.75]
table_top.size = [0.8, 0.6, 0.05]
table_top.rgba = [0.55, 0.35, 0.2, 1]


# ------------------------------------------------
# target
# ------------------------------------------------

target = spec.worldbody.add_body(
    name="target",
    pos=[0.45, 0, 0.90],
)

target.add_joint(
    name="target_freejoint",
    type=mujoco.mjtJoint.mjJNT_FREE,
)

target_geom = target.add_geom()

target_geom.name = "target_geom"
target_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
target_geom.size = [0.04, 0.08, 0]
target_geom.mass = 0.2
target_geom.rgba = [0.9, 0.2, 0.2, 1]


# ------------------------------------------------
# sanity check
# ------------------------------------------------

model = spec.compile()

print("Bodies:", model.nbody)
print("Geoms:", model.ngeom)


# ------------------------------------------------
# save
# ------------------------------------------------

output.write_text(spec.to_xml())

print("saved:", output)