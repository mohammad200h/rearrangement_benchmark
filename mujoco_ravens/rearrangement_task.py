"""A high-level task API for rearrangement tasks that leverage motion planning."""
import time

import PIL
import numpy as np
from task import construct_task_env

from dm_robotics.transformations.transformations import mat_to_quat
from ros2_start_docker import start_control_server


class RearrangementTask:
    """A high-level task API for rearrangement tasks that leverage motion planning."""

    def __init__(self, config):
        """Initialize the rearrangement task."""
        if config is None:
            self._task_env, config = construct_task_env()
        else:
            self._task_env, _ = construct_task_env(config)

        self.config = config
        self.shapes = self.config.props.shapes

        # start control and motion planning software
        if self.config.task.use_simulation:
            start_control_server()
            time.sleep(5)
        else:
            raise NotImplementedError

        # Automatically reset the task environment on initialization.
        self._task_env.reset()

    def __del__(self):
        """Close the task environment on instance deletion."""
        self._task_env.close()

    def pick(self, object_id):
        """
        Pick up an object.

        This method leverages MoveIt to plan and execute a pick motion.
        ROS 2 control manages sending commands to hardware interfaces.
        """
        raise NotImplementedError

    def place(self, object_id, position, orientation):
        """
        Place an object.

        This method leverages MoveIt to plan and execute a place motion.
        ROS 2 control manages sending commands to hardware interfaces.
        """
        raise NotImplementedError

    @property
    def props(self) -> dict:
        """
        Gets domain model.

        The domain model is a dictionary of objects and their properties.
        """
        # get prop object names
        prop_names = [
            self._task_env.physics.model.id2name(i, "geom")
            for i in range(self._task_env.physics.model.ngeom)
            if any(keyword in self._task_env.physics.model.id2name(i, "geom") for keyword in self.shapes)
        ]
        prop_ids = [self._task_env.physics.model.name2id(name, "geom") for name in prop_names]

        # get object information
        prop_positions = self._task_env.physics.named.data.geom_xpos[prop_names]
        prop_orientations = self._task_env.physics.named.data.geom_xmat[prop_names]
        prop_orientations = [mat_to_quat(mat.reshape((3, 3))) for mat in prop_orientations]
        prop_rgba = self._task_env.physics.named.model.geom_rgba[prop_names]
        prop_names = [name.split("/")[0] for name in prop_names]

        # get object bounding box information
        def get_bbox(prop_id, segmentation_map):
            """Get the bounding box of an object (PASCAL VOC)."""
            prop_coords = np.argwhere(segmentation_map[:, :, 0] == prop_id)
            bbox_corners = np.array(
                [
                    np.min(prop_coords[:, 0]),
                    np.min(prop_coords[:, 1]),
                    np.max(prop_coords[:, 0]),
                    np.max(prop_coords[:, 1]),
                ]
            )

            return bbox_corners

        # TODO: consider vectorizing this
        segmentation_map = self._task_env.physics.render(segmentation=True)
        prop_bbox = []
        for idx in prop_ids:
            bbox = get_bbox(idx, segmentation_map)
            prop_bbox.append(bbox)

        # create a dictionary with all the data
        prop_info = {
            prop_names[i]: {
                "position": prop_positions[i],
                "orientation": prop_orientations[i],
                "rgba": prop_rgba[i],
                "bbox": prop_bbox[i],
            }
            for i in range(len(prop_names))
        }

        return prop_info

    def render(self):
        """Render the task environment."""
        pixels = self._task_env.physics.render()
        PIL.Image.fromarray(pixels).show()


if __name__ == "__main__":
    task = RearrangementTask(None)
    # task.testing_dude()
    # task.render()
    # task.print_mjcf()
    task.render()
    print(task.props)
