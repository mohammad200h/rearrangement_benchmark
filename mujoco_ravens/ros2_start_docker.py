"""A script to test ROS 2 functionality."""
import os
import subprocess


def start_control_server():
    """Start the control server."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "control")
    subprocess.Popen(["docker", "compose", "-f", "docker-compose-control.yaml", "up"], cwd=docker_compose_dir)


def shutdown_control_server():
    """Shutdown the control server."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "control")
    subprocess.call(["docker", "compose", "-f", "docker-compose-control.yaml", "down"], cwd=docker_compose_dir)


def start_motion_planning_prerequisites():
    """Start the motion planning prerequisites."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "motion_planning")
    subprocess.Popen(["docker", "compose", "-f", "docker-compose-motion-planning.yaml", "up"], cwd=docker_compose_dir)


def shutdown_motion_planning_prerequisites():
    """Shutdown the motion planning prerequisites."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "motion_planning")
    subprocess.call(["docker", "compose", "-f", "docker-compose-motion-planning.yaml", "down"], cwd=docker_compose_dir)


if __name__ == "__main__":
    start_control_server()
