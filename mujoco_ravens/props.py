"""A script for defining props."""

import random
from typing import Dict, Sequence, Tuple, List
from dataclasses import dataclass, field

from dm_robotics.moma.prop import Prop
from dm_robotics.moma.sensors import prop_pose_sensor
from dm_control import composer, mjcf

import numpy as np

# Prop Models

# Block


def _make_block_model(
    name, x_len, y_len, z_len, pos=(0.0, 0.0, 0.0), rgba=(1, 0, 0, 1), solimp=(0.95, 0.995, 0.001), solref=(0.002, 0.7)
):
    """Make a block model: the mjcf element, and site."""
    mjcf_root = mjcf.element.RootElement(model=name)
    prop_root = mjcf_root.worldbody.add("body", name="prop_root")
    box = prop_root.add(
        "geom",
        name="body",
        type="box",
        pos=pos,
        size=(x_len, y_len, z_len),
        solref=solref,
        solimp=solimp,
        condim=1,
        rgba=rgba,
    )
    site = prop_root.add(
        "site",
        name="box_centre",
        type="sphere",
        rgba=(0.1, 0.1, 0.1, 0.8),
        size=(0.005,),
        pos=(0, 0, 0),
        euler=(0, 0, 0),
    )  # Was (np.pi, 0, np.pi / 2)
    del box

    return mjcf_root, site


class Block(Prop):
    """A block prop."""

    def _build(  # pylint:disable=arguments-renamed
        self,
        rgba: List,
        name: str = "box",
        x_len=0.04,
        y_len=0.04,
        z_len=0.04,
        pos=(0.0, 0.0, 0.0),
    ) -> None:
        mjcf_root, site = _make_block_model(name, x_len, y_len, z_len, rgba=rgba, pos=pos)
        super()._build(name, mjcf_root, "prop_root")
        del site


# Cylinder


def _make_cylinder_model(
    rgba: List,
    name: str = "cylinder",
    radius: float = 0.025,
    half_height: float = 0.1,
):
    mjcf_root = mjcf.element.RootElement(model=name)
    prop_root = mjcf_root.worldbody.add("body", name="prop_root")
    cylinder = prop_root.add(
        "geom",
        name=name,
        type="cylinder",
        pos=[0, 0, 0],
        size=[radius, half_height],
        rgba=rgba,
    )

    return mjcf_root, cylinder


class Cylinder(Prop):
    """A cylinder prop."""

    def _build(
        self,
        rgba: List,
        name: str = "cylinder",
        radius: float = 0.025,
        half_height: float = 0.1,
    ) -> None:
        """Build the prop."""
        mjcf_root, cylinder = _make_cylinder_model(
            name=name,
            radius=radius,
            half_height=half_height,
            rgba=rgba,
        )
        super()._build(name, mjcf_root, "prop_root")
        del cylinder


# Sphere


def _make_sphere_model(
    rgba: List,
    name: str = "sphere",
    radius: float = 0.025,
):
    mjcf_root = mjcf.element.RootElement(model=name)
    prop_root = mjcf_root.worldbody.add("body", name="prop_root")
    sphere = prop_root.add(
        "geom",
        name=name,
        type="sphere",
        pos=[0, 0, 0],
        size=[radius],
        rgba=rgba,
    )

    return mjcf_root, sphere


class Sphere(Prop):
    """A sphere prop."""

    def _build(
        self,
        rgba: List,
        name: str = "sphere",
        radius: float = 0.5,
    ) -> None:
        """Build the prop."""
        mjcf_root, sphere = _make_sphere_model(
            name=name,
            radius=radius,
            rgba=rgba,
        )
        super()._build(name, mjcf_root, "prop_root")
        del sphere


# Colours


@dataclass
class Colours:
    """Colour values for the objects in the scene."""

    colour_map: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "red": (1.0, 0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0, 1.0),
            "blue": (0.0, 0.0, 1.0, 1.0),
            "yellow": (1.0, 1.0, 0.0, 1.0),
            "cyan": (0.0, 1.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0, 1.0),
            "white": (1.0, 1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0, 1.0),
            "grey": (0.5, 0.5, 0.5, 1.0),
            "orange": (1.0, 0.5, 0.0, 1.0),
            "purple": (0.5, 0.0, 0.5, 1.0),
            "brown": (0.5, 0.25, 0.0, 1.0),
            "pink": (1.0, 0.75, 0.8, 1.0),
        }
    )

    @property
    def colour_names(self) -> Sequence[str]:
        """Return the names of the colours."""
        return list(self.colour_map.keys())

    def sample_colour(
        self,
    ) -> Tuple[float, float, float, float]:
        """Sample a random colour."""
        return self.colour_map[random.choice(self.colour_names)]


# Scene Samplers

# TODO: read these params from a config file
colours = Colours()
MIN_OBJECT_SIZE = 0.02
MAX_OBJECT_SIZE = 0.05


def _add_block(
    arena: composer.Arena,
    name: str = "block",
    x_len: float = 0.1,
    y_len: float = 0.1,
    z_len: float = 0.1,
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    sample: bool = False,
    seed: int = 0,
) -> composer.Entity:
    if sample:
        # sample block dimensions
        x_len = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        y_len = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        z_len = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        # sample block color
        rgba = colours.sample_colour()

    # create block and add to arena
    block = Block(name=name, x_len=x_len, y_len=y_len, z_len=z_len, rgba=rgba)
    frame = arena.add_free_entity(block)
    block.set_freejoint(frame.freejoint)

    return block


def _add_cylinder(
    arena: composer.Arena,
    name: str = "cylinder",
    radius: float = 0.01,
    half_height: float = 0.01,
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    sample: bool = False,
) -> composer.Entity:
    if sample:
        # sample cylinder dimensions
        radius = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        half_height = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        # sample cylinder color
        rgba = colours.sample_colour()

    # create cylinder and add to arena
    cylinder = Cylinder(name=name, radius=radius, half_height=half_height, rgba=rgba)

    frame = arena.add_free_entity(cylinder)
    cylinder.set_freejoint(frame.freejoint)
    return cylinder


def _add_sphere(
    arena: composer.Arena,
    name: str = "sphere",
    radius: float = 0.01,
    rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    sample: bool = False,
    seed: int = 0,
) -> composer.Entity:
    if sample:
        # sample sphere dimensions
        radius = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        # sample sphere color
        rgba = colours.sample_colour()

    # create sphere and add to arena
    sphere = Sphere(radius=radius, rgba=rgba)
    frame = arena.add_free_entity(sphere)
    sphere.set_freejoint(frame.freejoint)
    return sphere


def _add_object(area: composer.Arena, name: str, sample: bool = False) -> composer.Entity:
    """Add an object to the arena based on object_type."""
    if name == "block":
        return _add_block(area, sample=sample)
    elif name == "cylinder":
        return _add_cylinder(area, sample=sample)
    elif name == "sphere":
        return _add_sphere(area, sample=sample)
    else:
        raise ValueError(f"Unknown object type {name}")


def add_objects(arena: composer.Arena, objects: List[str], max_objects: int) -> List[composer.Entity]:
    """Add objects to the arena."""
    extra_sensors = []
    props = []

    # randomly sample num_objects of each object type
    num_objects = np.random.randint(1, max_objects, size=len(objects))

    for object_type, amount in zip(objects, num_objects):
        for i in range(amount):
            obj = _add_object(arena, object_type, sample=True)
            props.append(obj)
            extra_sensors.append(prop_pose_sensor.PropPoseSensor(obj, name=f"{object_type}_{i}"))

    return props, extra_sensors
