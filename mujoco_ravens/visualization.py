"""Scene visualization utilities."""

import numpy as np
from dm_control import mjcf, mujoco
import PIL.Image

def render_scene(physics: mjcf.Physics) -> None:
  camera = mujoco.MovableCamera(physics, height=480, width=480)
  camera.set_pose([0.0, 0, 0.0], 2.5, 180, -30)
  image_arr = camera.render()
  image = PIL.Image.fromarray(image_arr)
  image.show()
