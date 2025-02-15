import torch

from torch_quaternion.dualQuaternion import DualQuaternion
from torch_quaternion.kinematic.forward_kinematics import ForwardKinematics

class InverseKinematics:
    def __init__(self, twists: list = None, link_lengths: torch.Tensor = None, target_pose: DualQuaternion = None, target_position: torch.Tensor = None):
        """Initializes the inverse kinematics.
        
        Args:
            twists (list): List of twists for each joint (optional).
            link_lengths (torch.Tensor): Lengths of the robot arms (optional).
            target_pose (DualQuaternion): Target pose of the end effector (optional).
            target_position (torch.Tensor): Target position of the end effector (optional).
        
        Raises:
            ValueError: If neither twists nor link_lengths are provided.
            ValueError: If neither target_pose nor target_position are provided.
        """
        self.twists = twists
        self.link_lengths = link_lengths
        self.target_pose = target_pose
        self.target_position = target_position

        if self.twists is None and self.link_lengths is None:
            raise ValueError("Entweder twists oder link_lengths müssen angegeben werden.")
        if self.target_pose is None and self.target_position is None:
            raise ValueError("Entweder target_pose oder target_position müssen angegeben werden.")

    def solve(self, initial_angles: torch.Tensor, max_iterations=100, tolerance=1e-5) -> torch.Tensor:
        """Solves the inverse kinematics numerically.
        
        Args:
            initial_angles (torch.Tensor): Initial values for the joint angles.
            max_iterations (int): Maximum number of iterations.
            tolerance (float): Tolerance for convergence.
        
        Returns:
            torch.Tensor: The computed joint angles.
        
        The inverse kinematics problem aims to find the joint angles that achieve a desired end effector pose or position. 
        This can be formulated as minimizing the error between the current pose/position and the target pose/position:
        
        .. math::
            \text{error} = \text{target} - \text{current}
        
        The joint angles are updated iteratively to reduce this error, often using gradient descent or other optimization techniques.
        """
        angles = initial_angles.clone()
        for _ in range(max_iterations):
            # Berechne die aktuelle Pose oder Position
            if self.twists is not None:
                fk = ForwardKinematics(self.twists, self.target_pose)
                current_pose = fk.compute_pose(angles)
                error = self.target_pose - current_pose
            else:
                fk = ForwardKinematics(angles, self.link_lengths)
                current_position = fk.compute_end_effector()
                error = self.target_position - current_position

            # Überprüfe die Konvergenz
            if torch.norm(error) < tolerance:
                break

            # Aktualisiere die Winkel (z. B. mit Gradientenabstieg)
            angles += 0.1 * error  # Vereinfachte Aktualisierung
        return angles