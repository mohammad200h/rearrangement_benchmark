"""A script for defining props."""


from dm_robotics.moma.prop import Prop
from dm_control import mjcf




def make_block_model(name,
                      width,
                      height,
                      depth,
                      rgba=(1, 0, 0, 1),
                      solimp=(0.95, 0.995, 0.001),
                      solref=(0.002, 0.7)):
  """Makes a plug model: the mjcf element, and reward sites."""

  mjcf_root = mjcf.element.RootElement(model=name)
  prop_root = mjcf_root.worldbody.add('body', name='prop_root')
  box = prop_root.add(
      'geom',
      name='body',
      type='box',
      pos=(0, 0, 0),
      size=(width / 2., height / 2., depth / 2.),
      mass=0.050,
      solref=solref,
      solimp=solimp,
      condim=1,
      rgba=rgba)
  site = prop_root.add(
      'site',
      name='box_centre',
      type='sphere',
      rgba=(0.1, 0.1, 0.1, 0.8),
      size=(0.002,),
      pos=(0, 0, 0),
      euler=(0, 0, 0))  # Was (np.pi, 0, np.pi / 2)
  del box

  return mjcf_root, site


class Block(Prop):
  """A block prop."""

  def _build(  # pylint:disable=arguments-renamed
      self,
      name: str = 'box',
      width=0.04,
      height=0.04,
      depth=0.04,
      rgba: list = [1, 0, 0, 1],
      ) -> None:
    mjcf_root, site = make_block_model(name, width, height, depth, rgba=rgba)
    super()._build(name, mjcf_root, 'prop_root')
    del site


def make_cylinder_model(
        name: str = "cylinder",
        radius: float = 0.025,
        half_height: float = 0.1,
        rgba: list = [1, 0, 0, 1],
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
        name: str = "cylinder",
        radius: float = 0.025,
        half_height: float = 0.1,
        rgba: list = [1, 0, 0, 1],
    ) -> None:
        """Build the prop."""
        mjcf_root, cylinder = make_cylinder_model(
                name=name,
                radius=radius,
                half_height=half_height,
                rgba=rgba,
                )
        super()._build(name, mjcf_root, "prop_root")
        del cylinder


def make_sphere_model(
        name: str = "sphere",
        radius: float = 0.025,
        rgba: list = [1, 0, 0, 1],
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
        name: str = "sphere",
        radius: float = 0.5,
        rgba: list = [1, 0, 0, 1],
    ) -> None:
        """Build the prop."""
        mjcf_root, sphere = make_sphere_model(
                name=name,
                radius=radius,
                rgba=rgba,
                )
        super()._build(name, mjcf_root, "prop_root")
        del sphere

