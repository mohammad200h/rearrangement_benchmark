"""Scene visualization utilities."""
from typing import Tuple

from dm_robotics.moma.prop import Camera
from dm_robotics.moma.sensors.camera_sensor import CameraConfig, get_sensor_bundle
from dm_control import composer, mjcf, mujoco
import PIL.Image


def _make_fixed_camera(
    name: str,
    pos: Tuple = (0.0, 0.0, 0.0),
    quat: Tuple = (0.0, 0.0, 0.0, 1.0),
    height: int = 640,
    width: int = 480,
    fovy: float = 90.0,
) -> None:
    """Create fixed camera."""
    mjcf_root = mjcf.element.RootElement(model=name)
    prop_root = mjcf_root.worldbody.add(
        "body",
        name=f"{name}_root",
    )
    camera = prop_root.add(
        "camera",
        name=name,
        mode="fixed",
        pos=pos,
        quat=quat,
        fovy=fovy,
    )

    return mjcf_root, camera


class FixedCamera(Camera):
    """Fixed camera."""

    def _build(
        self,
        name: str,
        pos: str = "0 0 0",
        quat: str = "0 0 0 1",
        height: int = 640,
        width: int = 480,
        fovy: float = 90.0,
    ) -> None:
        """Build the camera."""
        # make the mjcf element
        mjcf_root, camera = _make_fixed_camera(
            name,
            pos,
            quat,
            height,
            width,
            fovy,
        )

        # build the camera
        super()._build(
            name=name,
            mjcf_root=mjcf_root,
            camera_element=name,
            prop_root=f"{name}_root",
            width=width,
            height=height,
            fovy=fovy,
        )
        del camera


def add_camera(
    arena: composer.Arena,
    name: str,
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    height: int = 480,
    width: int = 480,
    fovy: float = 90.0,
) -> composer.Entity:
    """Add a camera to the arena."""
    # create fixed camera
    camera = FixedCamera(
        name=name,
        pos=pos,
        quat=quat,
        height=height,
        width=width,
        fovy=fovy,
    )

    # attach to arena
    arena.mjcf_model.attach(camera.mjcf_model)

    # create camera sensors
    camera_config = CameraConfig(
        width=width, height=height, fovy=fovy, has_rgb=True, has_depth=True, has_segmentation=True, render_shadows=False
    )

    # TODO: investigate, strangely find_all and find result in different results
    cameras = camera_prop = arena.mjcf_model.find_all("camera")
    for camera in cameras:
        if camera.name == name:
            camera_prop = camera
            break

    pose_sensor, image_sensor = get_sensor_bundle(
        camera_prop,
        camera_config,
        name,
    )

    return camera, [pose_sensor, image_sensor]


def render_scene(physics: mjcf.Physics, x=0.0, y=0.0, z=0.0, roll=2.5, pitch=180, yaw=-30) -> None:
    """Render the scene using a movable camera."""
    camera = mujoco.MovableCamera(physics, height=480, width=480)
    camera.set_pose([x, y, z], roll, pitch, yaw)
    image_arr = camera.render()
    image = PIL.Image.fromarray(image_arr)
    image.show()
