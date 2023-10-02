"""A script to test ROS 2 functionality."""
import os
import sys
import subprocess


def start_control_server():
    """Start the control server."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    docker_compose_dir = os.path.join(file_dir, "..", ".docker", "control")
    success = subprocess.call(["docker", "compose", "-f", "docker-compose-control.yaml", "up"], cwd=docker_compose_dir)

    if success:
        print("Control server started successfully.")
        sys.exit(0)
    else:
        print("Control server failed to start.")
        sys.exit(1)


# def start_motion_planning_service():

if __name__ == "__main__":
    start_control_server()
