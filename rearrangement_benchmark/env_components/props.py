"""A script for defining props."""

import random
from typing import Dict, Sequence, Tuple, List

from dm_robotics.moma.prop import Prop
from dm_robotics.moma.sensors.prop_pose_sensor import PropPoseSensor
from dm_control import composer, mjcf

import numpy as np

COLOURS= {
    "red": (1.0, 0.0, 0.0, 1.0),
    "green": (0.0, 1.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0, 1.0),
    "grey": (0.5, 0.5, 0.5, 1.0),
}


class Rectangle(Prop):
    """Prop with a rectangular shape."""

    @staticmethod
    def _make(
        name:str,
        pos: Tuple[float, float, float]=(0.0, 0.0, 0.0),
        x_len: float = 0.1,
        y_len: float = 0.1,
        z_len: float = 0.1,
        rgba: Tuple[float, float, float,float]=(1, 0, 0, 1),
        solimp: Tuple[float, float, float]=(0.95, 0.995, 0.001),
        solref: Tuple[float, float, float]=(0.002, 0.7)
    ):
        """Make a block model: the mjcf element, and site."""
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        box = prop_root.add(
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=(x_len, y_len, z_len),
            solref=solref,
            solimp=solimp,
            condim=3,
            rgba=rgba,
            #mass = 10,
            friction = "10 10 10"
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

    
    def _build(  # pylint:disable=arguments-renamed
        self,
        rgba: List,
        name: str = "box",
        x_len: float = 0.1,
        y_len: float = 0.1,
        z_len: float = 0.1,
        pos=(0.0, 0.0, 0.0)
    ) -> None:
        mjcf_root, site = Rectangle._make(name,
                                          x_len=x_len,
                                          y_len=y_len,
                                          z_len=z_len,
                                          rgba=rgba,
                                          pos=pos)
        super()._build(name, mjcf_root, "prop_root")
        del site

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "red_rectangle_1",
        color: str = "red",
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,     
        x_len: float = 0.04,
        y_len: float = 0.04,
        z_len: float = 0.04,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        is_cube: bool = False,
        color_noise: float = 0.1,
    ) -> composer.Entity:
        """Add a block to the arena."""
        if sample_size:
            # sample block dimensions
            if is_cube:
                size = 3*[np.random.uniform(min_object_size, max_object_size)]
            else:
                size = np.random.uniform(min_object_size, max_object_size, size=3)

            x_len, y_len, z_len = size[0], size[1], size[2]
                
        if sample_colour:
            # sample block color
            rgba = COLOURS[color]
            # add noise
            rgba = [ c + np.random.uniform(-color_noise, color_noise) for c in rgba]
            rgba[3] = 1.0
            
        # create block and add to arena
        rectangle = Rectangle(name=name,
                              x_len=x_len,
                              y_len=y_len,
                              z_len=z_len,
                              rgba=rgba)
        frame = arena.add_free_entity(rectangle)
        rectangle.set_freejoint(frame.freejoint)

        return rectangle
    

class Cylinder(Prop):
    """A cylinder prop."""

    def _make(rgba: List,
            name: str = "cylinder",
            radius: float = 0.025,
            half_height: float = 0.1):
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        cylinder = prop_root.add("geom",
                                 name=name,
                                 type="cylinder",
                                 pos=(0, 0, 0),
                                 size=(radius, half_height),
                                 rgba=rgba,
                                 #mass=50
                                 )

        return mjcf_root, cylinder

    def _build(self,
               rgba: List,
               name: str = "cylinder",
               radius: float = 0.025,
               half_height: float = 0.1) -> None:
        """Build the prop."""
        mjcf_root, cylinder = Cylinder._make(name=name,
                                             radius=radius,
                                             half_height=half_height,
                                             rgba=rgba)
        super()._build(name, mjcf_root, "prop_root")
        del cylinder

    
    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "red_cylinder_1",
        color: str = "red",
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,     
        radius: float = 0.025,
        half_height: float = 0.1,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        color_noise: float = 0.1,
    ) -> composer.Entity:
        if sample_size:
            # sample block dimensions
            size = np.random.uniform(min_object_size, max_object_size, size=2)

            radius, half_height = size[0], size[1]

        if sample_colour:
            # sample block color
            rgba = COLOURS[color]
            # add noise
            rgba = [ c + np.random.uniform(-color_noise, color_noise) for c in rgba]
            rgba[3] = 1.0
        
        cylinder = Cylinder(name=name,
                            radius=radius,
                            half_height=half_height,
                            rgba=rgba)

        frame = arena.add_free_entity(cylinder)
        cylinder.set_freejoint(frame.freejoint)
        return cylinder


class Sphere(Prop):
    """A sphere prop."""

    @staticmethod
    def _make(rgba: List,
            name: str = "sphere",
            radius: float = 0.025):
        mjcf_root = mjcf.element.RootElement(model=name)
        prop_root = mjcf_root.worldbody.add("body", name="prop_root")
        sphere = prop_root.add("geom",
                               name=name,
                               type="sphere",
                               pos=(0, 0, 0),
                               size=(radius,),
                               rgba=rgba,
                               #mass=50,
                               )

        return mjcf_root, sphere

    def _build(self, rgba: List, name: str = "sphere", radius: float = 0.5) -> None:
        """Build the prop."""
        mjcf_root, sphere = Sphere._make(name=name, radius=radius, rgba=rgba)
        super()._build(name, mjcf_root, "prop_root")
        del sphere

    @staticmethod
    def _add(
        arena: composer.Arena,
        name: str = "red_sphere_1",
        color: str = "red",
        min_object_size: float = 0.02,
        max_object_size: float = 0.05,          
        radius: float = 0.025,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample_size: bool = False,
        sample_colour: bool = False,
        color_noise: float = 0.1,
    ) -> composer.Entity:
        if sample_size:
            # sample block dimensions
            radius = np.random.uniform(min_object_size, max_object_size)
        if sample_colour:
            # sample block color
            rgba = COLOURS[color]
            # add noise
            rgba = [ c + np.random.uniform(-color_noise, color_noise) for c in rgba]
            rgba[3] = 1.0

        # create sphere and add to arena
        sphere = Sphere(name=name, radius=radius, rgba=rgba)
        frame = arena.add_free_entity(sphere)
        sphere.set_freejoint(frame.freejoint)
        return sphere


def add_object(area: composer.Arena,
               name: str,
               shape: str,
               color: str,
               min_object_size: float,
               max_object_size: float,
               sample_size: bool = False,
               sample_colour: bool = False,
               color_noise: float=0.1) -> composer.Entity:
    """Add an object to the arena based on the shape and color."""
    if shape == "cube":
        return Rectangle._add(area,
                              name,
                              color,
                              min_object_size,
                              max_object_size,                             
                              is_cube=True,
                              sample_size=sample_size,
                              sample_colour=sample_colour,
                              color_noise=color_noise)
    elif shape == "rectangle":
        return Rectangle._add(area,
                              name,
                              color,
                              min_object_size,
                              max_object_size,           
                              sample_size=sample_size,
                              sample_colour=sample_colour,
                              color_noise=color_noise)
    elif shape == "cylinder":
        return Cylinder._add(area,
                             name,
                             color,
                             min_object_size,
                             max_object_size,          
                             sample_size=sample_size,
                             sample_colour=sample_colour,
                             color_noise=color_noise)
    elif shape == "sphere":
        return Sphere._add(area,
                           name,
                           color,
                           min_object_size,
                           max_object_size,          
                           sample_size=sample_size,
                           sample_colour=sample_colour,
                           color_noise=color_noise)
    else:
        raise ValueError(f"Unknown shape {shape}")

def add_objects(
    arena: composer.Arena,
    shapes: List[str], 
    colours: List[str],
    min_object_size: float,
    max_object_size: float,
    min_objects: int,
    max_objects: int,
    sample_size: bool = True,
    sample_colour: bool = True,
    color_noise: float = 0.1,
) -> List[composer.Entity]:
    """Add objects to the arena."""

    assert all(colour in COLOURS.keys() for colour in colours), "Unknown colour"

    extra_sensors = []
    props = []

    num_objects = np.random.randint(min_objects, max_objects)
    for i in range(num_objects):

        shape = random.choice(shapes)
        colour = random.choice(colours)
        name = f"{colour}_{shape}_{i}"
        obj = add_object(arena,
                         name,
                         shape,
                         colour,
                         min_object_size,
                         max_object_size,
                         sample_size,
                         sample_colour,
                         color_noise)
        
        props.append(obj)
        extra_sensors.append(PropPoseSensor(obj, name=name))

    return props, extra_sensors
