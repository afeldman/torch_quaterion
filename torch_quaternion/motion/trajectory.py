from typing import List

import torch

from torch_quaternion.dualQuaternion import DualQuaternion
from torch_quaternion.kinematic.forward_kinematics import ForwardKinematics
from torch_quaternion.kinematic.inverse_kinematics import InverseKinematics
from torch_quaternion.kinematic.twist import Twist

class Trajectory:

    """
    A class used to represent a Trajectory for a robotic arm.

    Attributes:
        twists (List[Twist]): List of twists for each joint.
        link_lengths (torch.Tensor): Lengths of the robot arms.
        initial_pose (DualQuaternion): Initial pose of the end effector.
        fk (ForwardKinematics): Forward kinematics solver.
        ik (InverseKinematics): Inverse kinematics solver.
    """

    def __init__(self, twists: List[Twist], link_lengths: torch.Tensor, initial_pose: DualQuaternion):
        """
        Initializes the Trajectory class.
        
        Args:
            twists (List[Twist]): List of twists for each joint.
            link_lengths (torch.Tensor): Lengths of the robot arms.
            initial_pose (DualQuaternion): Initial pose of the end effector.
        """
        self.twists = twists
        self.link_lengths = link_lengths
        self.initial_pose = initial_pose
        self.fk = ForwardKinematics(twists=twists, initial_pose=initial_pose)
        self.ik = InverseKinematics(twists=twists, target_pose=initial_pose)

    def plan_trajectory(self, start_pose: DualQuaternion, end_pose: DualQuaternion, num_points: int) -> List[DualQuaternion]:
        """
        Plans a trajectory between two poses.
        
        Args:
            start_pose (DualQuaternion): Start pose.
            end_pose (DualQuaternion): Target pose.
            num_points (int): Number of intermediate points.
        
        Returns:
            List[DualQuaternion]: List of poses along the trajectory.
        """
        # Lineare Interpolation zwischen den Posen
        trajectory = []
        for t in torch.linspace(0, 1, num_points):
            pose = self.interpolate_poses(start_pose, end_pose, t)
            trajectory.append(pose)
        return trajectory

    def interpolate_poses(self, pose1: DualQuaternion, pose2: DualQuaternion, t: float) -> DualQuaternion:
        """
        Interpolates between two poses.
        
        Args:
            pose1 (DualQuaternion): First pose.
            pose2 (DualQuaternion): Second pose.
            t (float): Interpolation factor (0 <= t <= 1).
        
        Returns:
            DualQuaternion: Interpolated pose.
        """
        return pose1 * (1 - t) + pose2 * t

    def compute_joint_angles(self, trajectory: List[DualQuaternion]) -> List[torch.Tensor]:
        """
        Computes the joint angles for each pose in the trajectory.
        
        Args:
            trajectory (List[DualQuaternion]): List of poses along the trajectory.
        
        Returns:
            List[torch.Tensor]: List of joint angles for each pose.
        """
        joint_angles = []
        initial_angles = torch.zeros(len(self.twists))  # Startwerte für die Gelenkwinkel
        for pose in trajectory:
            self.ik.target_pose = pose
            angles = self.ik.solve(initial_angles)
            joint_angles.append(angles)
            initial_angles = angles  # Verwende die berechneten Winkel als Startwerte für die nächste Pose
        return joint_angles

    def visualize_trajectory(self, trajectory: List[DualQuaternion]):
        """
        Visualizes the trajectory.
        
        Args:
            trajectory (List[DualQuaternion]): List of poses along the trajectory.
        """
        import matplotlib.pyplot as plt

        # Extrahiere die Positionen des Endeffektors
        positions = [pose.translation for pose in trajectory]
        x = [pos[0].item() for pos in positions]
        y = [pos[1].item() for pos in positions]
        z = [pos[2].item() for pos in positions]

        # 3D-Plot der Trajektorie
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, '-o')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajektorie des Endeffektors")
        plt.show()
