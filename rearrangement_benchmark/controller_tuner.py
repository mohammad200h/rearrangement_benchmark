"""A high-level API for tuning controllers."""

import numpy as np
import pandas as pd
from mujoco import viewer
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
from dm_robotics.transformations import transformations as tr
from dm_robotics.transformations.transformations import mat_to_quat, quat_to_mat, quat_to_euler

import PIL
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rearrangement_benchmark.task import construct_task_env

from simple_pid import PID

URDF_PATH = "./models/arms/robot.urdf"

class ControllerTuner(object):
    """A high-level API for tuning controllers."""

    def __init__(self, cfg=None, pid_controllers=None):
        """Initializes a rearrangement task."""
        # if a config is provided overwrite the default config
        if cfg is not None:
            self._sim, self.config = construct_task_env(cfg)
        else:
            self._sim, self.config = construct_task_env()
        
        self.viewer = None
        self.ee_chain = Chain.from_urdf_file(URDF_PATH, base_elements=["panda_link0"]) # TODO: read from config
        self.joint_angles = None 
        self.obs = None

        # define pid controllers 
        if pid_controllers is None:
            self.pid_controllers = {}
            self.pid_controllers["joint_1"] = PID(100.0, 0.0, 0.0, setpoint=0)
            self.pid_controllers["joint_2"] = PID(0.0, 0.0, 0.0, setpoint=-np.pi/4)
            self.pid_controllers["joint_3"] = PID(0.0, 0.0, 0.0, setpoint=0)
            self.pid_controllers["joint_4"] = PID(0.0, 0.0, 0.0, setpoint=-3*np.pi/4)
            self.pid_controllers["joint_5"] = PID(0.0, 0.0, 0.0, setpoint=0)
            self.pid_controllers["joint_6"] = PID(0.0, 0.0, 0.0, setpoint=np.pi/2)
            self.pid_controllers["joint_7"] = PID(0.0, 0.0, 0.0, setpoint=np.pi/4)
        else:
            self.pid_controllers = pid_controllers

        # set pid time step
        for pid in self.pid_controllers.values():
            pid.sample_time = 0.001

    def __del__(self):
        """Cleans up the task."""
        self.viewer.close()
        self._sim.close()

    def update_internal_vars(self, obs):
        """Updates internal variables based on the observation."""
        self.joint_angles = obs[3]["franka_emika_panda_joint_pos"]
        self.obs = obs
    
    def reset(self):
        """Resets the task."""
        if self.viewer is not None:
            self.viewer.close()
        obs = self._sim.reset()
        self.update_internal_vars(obs)
        self.viewer = viewer.launch_passive(self._sim.physics.model._model, self._sim.physics.data._data)
        return obs

    def position_control(self, target_pos, control_iters = 100):
        # set PID setpoints 
        self.pid_controllers["joint_1"].setpoint = target_pos[0]
        self.pid_controllers["joint_2"].setpoint = target_pos[1]
        self.pid_controllers["joint_3"].setpoint = target_pos[2]
        self.pid_controllers["joint_4"].setpoint = target_pos[3]
        self.pid_controllers["joint_5"].setpoint = target_pos[4]
        self.pid_controllers["joint_6"].setpoint = target_pos[5]
        self.pid_controllers["joint_7"].setpoint = target_pos[6]

        # reset the sim
        obs = self.reset()
        self.update_internal_vars(obs)

        # store joint angles and velocities
        joint_pos_data = pd.DataFrame(columns=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"])
        joint_vel_data = pd.DataFrame(columns=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"])

        # start interactive plots
        itr = 0
        while True:
            # calcuate joint torques with pid controllers
            joint_torques = np.zeros(8)
            joint_torques[-1] = -255.0
            for i, joint in enumerate(self.pid_controllers.keys()):
                joint_torques[i] = self.pid_controllers[joint](self.joint_angles[i])
            
            obs = self._sim.step(joint_torques)
            self.update_internal_vars(obs)
            self.viewer.sync()
            itr += 1
            
            # retrieve joint angles and velocities
            joint_angles = self.obs[3]["franka_emika_panda_joint_pos"]     
            joint_velocities = self.obs[3]["franka_emika_panda_joint_vel"]
            joint_pos_data = pd.concat([joint_pos_data, pd.DataFrame([joint_angles], columns=joint_pos_data.columns)], ignore_index=True) 
            joint_vel_data = pd.concat([joint_vel_data, pd.DataFrame([joint_velocities], columns=joint_vel_data.columns)], ignore_index=True)
            

            if itr % control_iters == 0:
                # plot joint angles and velocities
                fig_pos = make_subplots(rows=3, cols=3, vertical_spacing=0.02)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_1"], name="joint_1"), row=1, col=1)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_2"], name="joint_2"), row=1, col=2)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_3"], name="joint_3"), row=1, col=3)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_4"], name="joint_4"), row=2, col=1)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_5"], name="joint_5"), row=2, col=2)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_6"], name="joint_6"), row=2, col=3)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_7"], name="joint_7"), row=3, col=1)
                fig_pos.update_layout(
                        height=600, 
                        width=800, 
                        title_text="Joint Angles",
                        yaxis_title="Joint Angle (rad)",
                        )
                
                if command_type == "position":
                    print("target_pos: ", target_pos)
                    # add constant reference line
                    fig_pos.add_hline(y=target_pos[0], line_dash="dash", row=1, col=1)
                    fig_pos.add_hline(y=target_pos[1], line_dash="dash", row=1, col=2)
                    fig_pos.add_hline(y=target_pos[2], line_dash="dash", row=1, col=3)
                    fig_pos.add_hline(y=target_pos[3], line_dash="dash", row=2, col=1)
                    fig_pos.add_hline(y=target_pos[4], line_dash="dash", row=2, col=2)
                    fig_pos.add_hline(y=target_pos[5], line_dash="dash", row=2, col=3)
                    fig_pos.add_hline(y=target_pos[6], line_dash="dash", row=3, col=1)


                fig_vel = make_subplots(rows=3, cols=3, vertical_spacing=0.02)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_1"], name="joint_1"), row=1, col=1)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_2"], name="joint_2"), row=1, col=2)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_3"], name="joint_3"), row=1, col=3)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_4"], name="joint_4"), row=2, col=1)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_5"], name="joint_5"), row=2, col=2)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_6"], name="joint_6"), row=2, col=3)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_7"], name="joint_7"), row=3, col=1)
                fig_vel.update_layout(
                        height=600, 
                        width=800, 
                        title_text="Joint Velocities",
                        yaxis_title="Joint Velocity (rad/s)",
                        )
                
                return (fig_pos, fig_vel)

    def actuator_command(self, command, control_iters = 100, command_type="velocity"):
        """Sets a command for the actuators and prints debug output."""
        # reset the sim
        self.obs = self.reset()

        # store joint angles and velocities
        joint_pos_data = pd.DataFrame(columns=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"])
        joint_vel_data = pd.DataFrame(columns=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"])

        # start interactive plots
        itr = 0
        while True:
            obs = self._sim.step(command)
            self.update_internal_vars(obs)
            self.viewer.sync()
            itr += 1

            # retrieve joint angles and velocities
            joint_angles = self.obs[3]["franka_emika_panda_joint_pos"]     
            joint_velocities = self.obs[3]["franka_emika_panda_joint_vel"]
            joint_pos_data = pd.concat([joint_pos_data, pd.DataFrame([joint_angles], columns=joint_pos_data.columns)], ignore_index=True) 
            joint_vel_data = pd.concat([joint_vel_data, pd.DataFrame([joint_velocities], columns=joint_vel_data.columns)], ignore_index=True)
            
            if itr % control_iters == 0:
                # plot joint angles and velocities
                fig_pos = make_subplots(rows=3, cols=3, vertical_spacing=0.02)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_1"], name="joint_1"), row=1, col=1)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_2"], name="joint_2"), row=1, col=2)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_3"], name="joint_3"), row=1, col=3)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_4"], name="joint_4"), row=2, col=1)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_5"], name="joint_5"), row=2, col=2)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_6"], name="joint_6"), row=2, col=3)
                fig_pos.add_trace(go.Scatter(y=joint_pos_data["joint_7"], name="joint_7"), row=3, col=1)
                fig_pos.update_layout(
                        height=600, 
                        width=800, 
                        title_text="Joint Angles",
                        yaxis_title="Joint Angle (rad)",
                        )
                
                if command_type == "position":
                    # add constant reference line
                    fig_pos.add_hline(y=command[0], line_dash="dash", row=1, col=1)
                    fig_pos.add_hline(y=command[1], line_dash="dash", row=1, col=2)
                    fig_pos.add_hline(y=command[2], line_dash="dash", row=1, col=3)
                    fig_pos.add_hline(y=command[3], line_dash="dash", row=2, col=1)
                    fig_pos.add_hline(y=command[4], line_dash="dash", row=2, col=2)
                    fig_pos.add_hline(y=command[5], line_dash="dash", row=2, col=3)
                    fig_pos.add_hline(y=command[6], line_dash="dash", row=3, col=1)


                fig_vel = make_subplots(rows=3, cols=3, vertical_spacing=0.02)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_1"], name="joint_1"), row=1, col=1)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_2"], name="joint_2"), row=1, col=2)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_3"], name="joint_3"), row=1, col=3)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_4"], name="joint_4"), row=2, col=1)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_5"], name="joint_5"), row=2, col=2)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_6"], name="joint_6"), row=2, col=3)
                fig_vel.add_trace(go.Scatter(y=joint_vel_data["joint_7"], name="joint_7"), row=3, col=1)
                fig_vel.update_layout(
                        height=600, 
                        width=800, 
                        title_text="Joint Velocities",
                        yaxis_title="Joint Velocity (rad/s)",
                        )

                if command_type == "velocity":
                    # add constant reference lines
                    fig_vel.add_hline(y=command[0], line_dash="dash", line_color="red", row=1, col=1)
                    fig_vel.add_hline(y=command[1], line_dash="dash", line_color="red", row=1, col=2)
                    fig_vel.add_hline(y=command[2], line_dash="dash", line_color="red", row=1, col=3)
                    fig_vel.add_hline(y=command[3], line_dash="dash", line_color="red", row=2, col=1)
                    fig_vel.add_hline(y=command[4], line_dash="dash", line_color="red", row=2, col=2)
                    fig_vel.add_hline(y=command[5], line_dash="dash", line_color="red", row=2, col=3)
                    fig_vel.add_hline(y=command[6], line_dash="dash", line_color="red", row=3, col=1)

                return (fig_pos, fig_vel)
