"""A high-level task API for rearrangement tasks that leverage motion planning."""


class RearrangementTask:
    """A high-level task API for rearrangement tasks that leverage motion planning."""

    def __init__(self):
        """Initialize the rearrangement task."""
        pass

    def pick(self, object_id):
        """
        Pick up an object.

        This method leverages MoveIt to plan and execute a pick motion.
        ROS 2 control manages sending commands to hardware interfaces.
        """
        raise NotImplementedError

    @property
    def domain_model(self):
        """
        Gets domain model.

        The domain model is a dictionary of objects and their properties.
        """
        raise NotImplementedError

    @property
    def problem_instance(self):
        """
        Gets problem instance.

        The problem instance is a dictionary of objects and their locations.
        """
        raise NotImplementedError
