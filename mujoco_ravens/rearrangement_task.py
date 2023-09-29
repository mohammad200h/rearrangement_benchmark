"""A high-level task API for rearrangement tasks that leverage motion planning."""

import PIL
import numpy as np
from task import construct_task_env

from dm_robotics.transformations.transformations import mat_to_quat


class RearrangementTask:
    """A high-level task API for rearrangement tasks that leverage motion planning."""

    def __init__(self, config):
        """Initialize the rearrangement task."""
        if config is None:
            self._task_env = construct_task_env()
        else:
            self._task_env = construct_task_env(config)

        self.effectors = self._task_env.task.effectors
        self.FILTER_GEOM_WORDS = ["panda", "table", "ground"]

        # Automatically reset the task environment on initialization.
        self._task_env.reset()

    def testing_dude(self):
        """Testing dude."""
        print("testing dude")
        print(self._task_env.physics.data.geom_xpos)

    def render(self):
        """Render the task environment."""
        pixels = self._task_env.physics.render()
        PIL.Image.fromarray(pixels).show()

    def print_mjcf(self):
        """Print the MJCF model."""
        print(self._task_env.physics.data.time)
        print(self._task_env.physics.named.data.geom_xpos)
        print(self._task_env.physics.named.model.geom_rgba)
        parent_action_spec = self._task_env.task.effectors_action_spec(
            physics=self._task_env.physics, effectors=self.effectors
        )
        min_action = parent_action_spec.minimum
        noop_action = np.ones(parent_action_spec.shape, dtype=parent_action_spec.dtype) * min_action
        for _ in range(10):
            self._task_env.step(noop_action)
            print(self._task_env.physics.data.time)
            print(self._task_env.physics.named.data.geom_xpos)

    def close(self):
        """Close the task environment."""
        self._task_env.close()

    def pick(self, object_id):
        """
        Pick up an object.

        This method leverages MoveIt to plan and execute a pick motion.
        ROS 2 control manages sending commands to hardware interfaces.
        """
        raise NotImplementedError

    @property
    def domain_model(self) -> dict:
        """
        Gets domain model.

        The domain model is a dictionary of objects and their properties.
        """
        # get manipulation object names
        object_names = [
            self._task_env.physics.model.id2name(i, "geom") for i in range(self._task_env.physics.model.ngeom)
        ]
        object_names = [obj for obj in object_names if all(keyword not in obj for keyword in self.FILTER_GEOM_WORDS)]

        # get object information
        object_positions = self._task_env.physics.named.data.geom_xpos[object_names]
        object_orientations = self._task_env.physics.named.data.geom_xmat[object_names]
        object_orientations = [mat_to_quat(mat.reshape((3, 3))) for mat in object_orientations]
        object_rgba = self._task_env.physics.named.model.geom_rgba[object_names]

        print(object_positions)
        print(object_orientations)
        print(object_rgba)

        # map object coordinates to se2

        return dict

    @property
    def problem_instance(self):
        """
        Gets problem instance.

        The problem instance is a dictionary of objects and their locations.
        """
        raise NotImplementedError


if __name__ == "__main__":
    task = RearrangementTask(None)
    # task.testing_dude()
    # task.render()
    # task.print_mjcf()
    task.domain_model
    task.close()
