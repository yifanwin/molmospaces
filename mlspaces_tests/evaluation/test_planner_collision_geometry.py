import mujoco
import numpy as np

from molmo_spaces.utils.planner_collision_geometry import body_collision_boxes


def test_separate_collision_geoms_preserve_furniture_cavity():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="counter" pos="2 3 0">
      <geom type="box" pos="-1 0 0" size=".1 1 .5"/>
      <geom type="box" pos="1 0 0" size=".1 1 .5"/>
      <geom type="box" pos="0 1 0" size="1 .1 .5"/>
      <geom type="box" size="1 1 .5" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    transform = np.eye(4)
    transform[:3, 3] = [-2, -3, 0]
    boxes = body_collision_boxes(model, data, model.body('counter').id, transform)
    assert len(boxes) == 3
    for _, pose, dims in boxes:
        assert not np.all(np.abs(pose[:3, 3]) <= dims / 2)
    np.testing.assert_allclose(boxes[0][1][:3, 3], [-1, 0, 0])
    np.testing.assert_allclose(boxes[0][2], [.2, 2, 1])
