"""A script to test ROS 2 functionality."""
import os
import subprocess


def start_control_server():
    """Start the control server."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "control")
    subprocess.Popen(["docker", "compose", "-f", "docker-compose-control.yaml", "up"], cwd=docker_compose_dir)


def stop_control_server():
    """Stop the control server."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "control")
    subprocess.call(["docker", "compose", "-f", "docker-compose-control.yaml", "down"], cwd=docker_compose_dir)


# def start_motion_planning_service():

if __name__ == "__main__":
    start_control_server()
