import torch

from torch_quaternion.dualQuaternion import DualQuaternion
from torch_quaternion.quaternion import Quaternion

class Wrench(DualQuaternion):
    """
    A class to represent a wrench, which is a combination of force and torque.

    Attributes:
        force (torch.Tensor): The force vector.
        torque (torch.Tensor): The torque vector.

    Methods:
        __init__(force: torch.Tensor, torque: torch.Tensor): Initializes the wrench with force and torque.
        force() -> torch.Tensor: Returns the force vector.
        torque() -> torch.Tensor: Returns the torque vector.
    """

    def __init__(self, force: torch.Tensor, torque: torch.Tensor):
        """
        Initializes a wrench from force and torque.

        Args:
            force (torch.Tensor): Force (3D vector).
            torque (torch.Tensor): Torque (3D vector).
        """
        # Real part: Force
        real_part = Quaternion(size=1)
        real_part.quaternion = torch.cat([torch.zeros(1, 1), force.unsqueeze(0)], dim=1)

        # Dual part: Torque
        dual_part = Quaternion(size=1)
        dual_part.quaternion = torch.cat([torch.zeros(1, 1), torque.unsqueeze(0)], dim=1)

        super().__init__(real_part, dual_part)

    @property
    def force(self) -> torch.Tensor:
        """
        Returns the force vector.

        Returns:
            torch.Tensor: The force vector.
        """
        return self.real.vector

    @property
    def torque(self) -> torch.Tensor:
        """
        Returns the torque vector.

        Returns:
            torch.Tensor: The torque vector.
        """
        return self.dual.vector
