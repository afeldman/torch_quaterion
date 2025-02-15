import torch

from torch_quaternion.dualQuaternion import DualQuaternion
from torch_quaternion.quaternion import Quaternion

class Twist(DualQuaternion):
    def __init__(self, angular_velocity: torch.Tensor, linear_velocity: torch.Tensor):
        """Initializes a Twist from angular velocity and linear velocity.
        
        Args:
            angular_velocity (torch.Tensor): Angular velocity (3D vector).
            linear_velocity (torch.Tensor): Linear velocity (3D vector).
        """
        # Realteil: Winkelgeschwindigkeit
        real_part = Quaternion(size=1)
        real_part.quaterion = torch.cat([torch.zeros(1, 1), angular_velocity.unsqueeze(0)], dim=1)

        # Dualteil: Translationsgeschwindigkeit
        dual_part = Quaternion(size=1)
        dual_part.quaterion = torch.cat([torch.zeros(1, 1), linear_velocity.unsqueeze(0)], dim=1)

        super().__init__(real_part, dual_part)

    @property
    def angular_velocity(self) -> torch.Tensor:
        """Returns the angular velocity."""
        return self.real.vector

    @property
    def linear_velocity(self) -> torch.Tensor:
        """Returns the linear velocity."""
        return self.dual.vector

    def exponential_map(self, angle: float) -> DualQuaternion:
        """Computes the exponential map of the twist.
        
        Args:
            angle (float): The angle in radians.
        
        Returns:
            DualQuaternion: The resulting dual quaternion.
        """
        # Realteil: Rotation
        real_part = Quaternion.from_rodrigues(self.angular_velocity, angle)

        # Dualteil: Translation
        dual_part = Quaternion(size=1)
        dual_part.quaterion = torch.cat([
            torch.zeros(1, 1),
            (angle * self.linear_velocity).unsqueeze(0)
        ], dim=1)

        return DualQuaternion(real_part, dual_part)
