import torch

from torch_quaternion.dualQuaternion import DualQuaternion

class ForwardKinematics:
    def __init__(self, twists: list = None, link_lengths: torch.Tensor = None, initial_pose: DualQuaternion = None):
        """Initialisiert die Vorwärtskinematik.
        
        Args:
            twists (list): Liste von Twists für jedes Gelenk (optional).
            link_lengths (torch.Tensor): Längen der Roboterarme (optional).
            initial_pose (DualQuaternion): Startpose des Endeffektors (optional).
        """
        self.twists = twists
        self.link_lengths = link_lengths
        self.initial_pose = initial_pose

        if self.twists is None and self.link_lengths is None:
            raise ValueError("Entweder twists oder link_lengths müssen angegeben werden.")
        if self.twists is not None and self.initial_pose is None:
            raise ValueError("Für twist-basierte Kinematik muss initial_pose angegeben werden.")

    def compute_end_effector(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Berechnet die Position des Endeffektors basierend auf den Gelenkwinkeln.
        
        Args:
            joint_angles (torch.Tensor): Gelenkwinkel in Radiant.
        
        Returns:
            torch.Tensor: Die Endeffektor-Position.
        """
        if self.link_lengths is None:
            raise ValueError("Link-Längen müssen für diese Methode angegeben werden.")

        # Einfache 2D-Kinematik
        x = torch.sum(self.link_lengths * torch.cos(joint_angles))
        y = torch.sum(self.link_lengths * torch.sin(joint_angles))
        return torch.tensor([x, y, 0.0])  # Endeffektor-Position

    def compute_pose(self, joint_angles: torch.Tensor) -> DualQuaternion:
        """Berechnet die Endeffektor-Pose basierend auf den Gelenkwinkeln.
        
        Args:
            joint_angles (torch.Tensor): Gelenkwinkel in Radiant.
        
        Returns:
            DualQuaternion: Die Endeffektor-Pose.
        """
        if self.twists is None:
            raise ValueError("Twists müssen für diese Methode angegeben werden.")

        current_pose = self.initial_pose
        for twist, angle in zip(self.twists, joint_angles):
            # Exponentielle Abbildung des Twists
            exp_twist = twist.exponential_map(angle)
            # Aktualisiere die Pose
            current_pose = exp_twist * current_pose
        return current_pose

    def compute_jacobian(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """Berechnet den Jacobian basierend auf den Gelenkwinkeln.
        
        Args:
            joint_angles (torch.Tensor): Gelenkwinkel in Radiant.
        
        Returns:
            torch.Tensor: Der Jacobian.
        """
        if self.link_lengths is None:
            raise ValueError("Link-Längen müssen für diese Methode angegeben werden.")

        num_joints = joint_angles.shape[0]
        jacobian = torch.zeros((3, num_joints))

        for i in range(num_joints):
            # Berechne die partielle Ableitung der Endeffektor-Position nach jedem Gelenkwinkel
            partial_x = -torch.sum(self.link_lengths[i:] * torch.sin(joint_angles[i:]))
            partial_y = torch.sum(self.link_lengths[i:] * torch.cos(joint_angles[i:]))
            jacobian[0, i] = partial_x
            jacobian[1, i] = partial_y
            jacobian[2, i] = 0.0  # Z-Komponente ist in 2D-Kinematik null

        return jacobian
