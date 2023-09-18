"""Scene visualization utilities."""

import numpy as np
from dm_control import mjcf, mujoco

def render_scene(physics: mjcf.Physics) -> np.ndarray:
  camera = mujoco.MovableCamera(physics, height=480, width=480)
  camera.set_pose([0.0, 0, 0.0], 2.5, 180, -30)
  return camera.render()
