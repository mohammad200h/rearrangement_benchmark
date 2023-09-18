"""Scene visualization utilities."""

from dm_control import mjcf, mujoco
import PIL.Image


def render_scene(physics: mjcf.Physics, x=0.0, y=0.0, z=0.0, roll=2.5, pitch=180, yaw=-30) -> None:
    """Render the scene using a movable camera."""
    camera = mujoco.MovableCamera(physics, height=480, width=480)
    camera.set_pose([x, y, z], roll, pitch, yaw)
    image_arr = camera.render()
    image = PIL.Image.fromarray(image_arr)
    image.show()
