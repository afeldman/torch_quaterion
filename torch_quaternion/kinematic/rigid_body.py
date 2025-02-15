import torch

from torch_quaternion.dualQuaternion import DualQuaternion
from torch_quaternion.quaternion import Quaternion

class RigidBody:
    def __init__(self, rotation: Quaternion, translation: Quaternion):
        """Initializes a RigidBody with a given rotation and translation.
        
        Args:
            rotation (Quaternion): The rotation component.
            translation (Quaternion): The translation component.
        """
        self.transform = DualQuaternion(rotation, translation)

    def apply_transform(self, point: torch.Tensor) -> torch.Tensor:
        """Applies the rigid body transformation to a point.
        
        Args:
            point (torch.Tensor): The point to transform (3D vector).
        
        Returns:
            torch.Tensor: The transformed point (3D vector).
        """
        # Punkt in ein duales Quaternion umwandeln
        point_real = Quaternion(size=1)
        point_real.quaternion = torch.cat([torch.zeros(1, 1), point.unsqueeze(0)], dim=1)
        point_dual = Quaternion(size=1)
        point_dual.quaternion = torch.zeros(1, 4)
        point_dual_quat = DualQuaternion(point_real, point_dual)

        # Transformation anwenden
        transformed_point = self.transform * point_dual_quat * self.transform.conjugate()
        return transformed_point.real.vector  # Ergebnis ist der transformierte Punkt
