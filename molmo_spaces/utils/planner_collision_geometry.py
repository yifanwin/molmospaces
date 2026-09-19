"""从 MuJoCo 碰撞几何生成保留家具空腔的局部包围盒。"""

import numpy as np

from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms


def body_collision_boxes(model, data, body_id, world_to_base):
    """返回 (geom_id, 基座系位姿, 完整尺寸)，不包含仅用于渲染的几何。"""
    boxes = []
    for geom_id in descendant_geoms(model, body_id, visible_only=False):
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        local_center, half_extent = model.geom_aabb[geom_id].reshape(2, 3)
        if not np.all(np.isfinite(half_extent)) or np.any(half_extent <= 0):
            raise ValueError(f"Unsupported collision geometry bounds: geom {geom_id}")
        pose = np.eye(4)
        pose[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
        pose[:3, 3] = data.geom_xpos[geom_id] + pose[:3, :3] @ local_center
        boxes.append((geom_id, world_to_base @ pose, 2 * half_extent))
    return boxes
