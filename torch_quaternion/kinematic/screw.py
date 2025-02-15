import torch

from torch_quaternion.dualQuaternion import DualQuaternion
from torch_quaternion.quaternion import Quaternion


class Screw(DualQuaternion):
    def __init__(self, axis: torch.Tensor, translation: torch.Tensor, pitch: float = 0.0):
        """Initializes a Screw from rotation axis, translation component, and pitch.
        
        Args:
            axis (torch.Tensor): Rotation axis (3D vector).
            translation (torch.Tensor): Translation component (3D vector).
            pitch (float): Pitch of the screw.
        """
        # Normalisiere die Achse
        axis = axis / torch.norm(axis)

        # Realteil: Rotationsachse
        real_part = Quaternion(size=1)
        real_part.quaterion = torch.cat([torch.zeros(1, 1), axis.unsqueeze(0)], dim=1)

        # Dualteil: Translationsanteil + Pitch
        dual_part = Quaternion(size=1)
        dual_part.quaterion = torch.cat([torch.zeros(1, 1), (translation + pitch * axis).unsqueeze(0)], dim=1)

        super().__init__(real_part, dual_part)

    @property
    def axis(self) -> torch.Tensor:
        """Returns the rotation axis."""
        return self.real.vector

    @property
    def translation(self) -> torch.Tensor:
        """Returns the translation component."""
        return self.dual.vector

    @property
    def pitch(self) -> float:
        """Returns the pitch of the screw."""
        return torch.dot(self.axis, self.translation).item()
