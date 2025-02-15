import torch

from torch_quaternion.kinematic.wrench import Wrench


class Dynamics:
    """
    Diese Klasse modelliert die Dynamik eines Roboters basierend auf seiner Masse und seinem Trägheitstensor.
    
    Attributes:
        mass (float): Die Masse des Roboters.
        inertia (torch.Tensor): Der Trägheitstensor des Roboters.
    """
    def __init__(self, mass: float, inertia: torch.Tensor):
        """Initialisiert die Dynamik mit Masse und Trägheitstensor.
        
        Args:
            mass (float): Masse des Roboters.
            inertia (torch.Tensor): Trägheitstensor des Roboters.
        """
        self.mass = mass
        self.inertia = inertia

    def compute_acceleration(self, wrench: Wrench) -> torch.Tensor:
        """Berechnet die Beschleunigung basierend auf einem Wrench.
        
        Args:
            wrench (Wrench): Der Wrench, der auf den Roboter wirkt.
        
        Returns:
            torch.Tensor: Die lineare und Winkelbeschleunigung.
        """
        # Lineare Beschleunigung
        linear_acceleration = wrench.force / self.mass

        # Winkelbeschleunigung
        angular_acceleration = torch.matmul(torch.inverse(self.inertia), wrench.torque)

        return torch.cat([linear_acceleration, angular_acceleration])
