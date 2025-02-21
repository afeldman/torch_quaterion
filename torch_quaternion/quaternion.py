''' quaternion with torch backend '''
from typing import Tuple
import numbers
import torch
import numpy as np

from torch_quaternion.util import classproperty

class Quaternion:
    """
    A class to represent a quaternion with PyTorch backend.

    Attributes:
        gpu (bool): Indicates if GPU is available.
        quaternion (torch.Tensor): The quaternion tensor.

    Methods:
        __init__(size: int = 0, *args, **kwargs): Initializes the quaternion.
        is_unit() -> bool: Checks if the quaternion is a unit quaternion.
        normalize: Normalizes the quaternion to have a unit norm.
        complex() -> np.ndarray: Converts the quaternion to a complex number.
        __len__() -> int: Returns the length of the quaternion.
        __rsub__(p): Subtracts another quaternion or scalar from this quaternion.
        __isub__(p): Subtracts another quaternion or scalar from this quaternion.
        __sub__(p): Subtracts another quaternion or scalar from this quaternion.
        alpha_sub(p, alpha: int = 1): Subtracts another quaternion or scalar with a scaling factor.
        __radd__(p): Adds another quaternion or scalar to this quaternion.
        __iadd__(p): Adds another quaternion or scalar to this quaternion.
        __add__(p): Adds another quaternion or scalar to this quaternion.
        alpha_add(p, alpha: int = 1): Adds another quaternion or scalar with a scaling factor.
        __div__(p): Divides this quaternion by another quaternion or scalar.
        __rdiv__(p): Divides this quaternion by another quaternion or scalar.
        __idiv__(p): Divides this quaternion by another quaternion or scalar.
        __truediv__(p): Divides this quaternion by another quaternion or scalar.
        __rtruediv__(p): Divides this quaternion by another quaternion or scalar.
        __itruediv__(p): Divides this quaternion by another quaternion or scalar.
        __floordiv__(p): Performs floor division on this quaternion by another quaternion or scalar.
        __imul__(p): Multiplies this quaternion by another quaternion or scalar.
        __rmul__(p): Multiplies this quaternion by another quaternion or scalar.
        __mul__(p): Multiplies this quaternion by another quaternion or scalar.
        __matmul__(p): Performs matrix multiplication with another quaternion or scalar.
        __abs__() -> torch.Tensor: Returns the norm of the quaternion.
        sq_sum: Returns the squared sum of the quaternion elements.
        norm: Returns the norm of the quaternion.
        magnitude: Returns the magnitude of the quaternion.
        shape: Returns the shape of the quaternion tensor.
        np: Returns the quaternion as a numpy array.
        elements: Returns the quaternion tensor.
        real: Returns the scalar part of the quaternion.
        imaginary: Returns the vector part of the quaternion.
        vector: Returns the vector part of the quaternion.
        scalar: Returns the scalar part of the quaternion.
        random(counter: int = 1): Generates a random quaternion.
        zero: Returns a quaternion with zero scalar part.
        one: Returns a quaternion with one scalar part.
        x_: Returns a quaternion with x component.
        y_: Returns a quaternion with y component.
        z_: Returns a quaternion with z component.
        w: Returns the w component of the quaternion.
        x: Returns the x component of the quaternion.
        y: Returns the y component of the quaternion.
        z: Returns the z component of the quaternion.
        polar_unit_vector: Returns the polar unit vector of the quaternion.
        polar_angle: Returns the polar angle of the quaternion.
        polar_decomposition: Returns the polar decomposition of the quaternion.
        __str__() -> str: Returns a string representation of the quaternion.
        __repr__() -> str: Returns a string representation of the quaternion.
        __eq__(p) -> bool: Checks if two quaternions are equal.
        __neg__(): Negates the quaternion.
        conjugate(inplace=True): Computes the conjugate of the quaternion.
        __pow__(exponent): Raises the quaternion to a power.
        __ipow__(other): Raises the quaternion to a power.
        __rpow__(other): Raises the quaternion to a power.
        rotate_quaternion(p): Rotates the quaternion by another quaternion.
    """

    gpu = True if torch.cuda.is_available() else False

    def __init__(self, size: int = 0, *args, **kwargs) -> None:
        """
        Initializes the quaternion.

        Args:
            size (int): The size of the quaternion. Defaults to 0.
        """
        if size <= 0:
            size = 1

        one = np.array([1., 0., 0., 0.])

        if self.gpu:
            self.quaternion = torch.from_numpy(one).float().cuda().repeat(
                size, 1)
        else:
            self.quaternion = torch.from_numpy(one).float().cpu().repeat(
                size, 1)

    def is_unit(self) -> bool:
        """
        Checks if the quaternion is a unit quaternion.

        Returns:
            bool: True if the quaternion is a unit quaternion, False otherwise.
        """
        return bool(torch.all(self.norm.bool()))

    @property
    def normalize(self) -> 'Quaternion':
        """
        Normalizes the quaternion to have a unit norm.

        Returns:
            quaternion: The normalized quaternion.
        """
        norm = self.norm
        if not torch.allclose(norm, torch.tensor(1.0)):
            self.quaternion = torch.nn.functional.normalize(self.quaternion,
                                                            p=2,
                                                            dim=1)
        return self

    def complex(self) -> np.ndarray:
        """
        Converts the quaternion to a complex number.

        Returns:
            np.ndarray: The complex representation of the quaternion.
        """
        data = self.quaternion[:, 0:2].cpu().numpy()
        result = np.empty(data.shape[:-1], dtype=complex)
        result.real, result.imag = data[..., 0], data[..., 1]
        return result

    def __len__(self) -> int:
        """
        Returns the length of the quaternion.

        Returns:
            int: The length of the quaternion.
        """
        return int(self.quaternion.shape[0])

    def __rsub__(self, p) -> 'Quaternion':
        return self - p

    def __isub__(self, p) -> 'Quaternion':
        return self - p

    def __sub__(self, p) -> 'Quaternion':
        return self.alpha_sub(p=p)

    def alpha_sub(self, p, alpha: int = 1) -> 'Quaternion':
        """
        Subtracts another quaternion or a scalar from this quaternion.

        Args:
            p (quaternion or numbers.Number): The quaternion or scalar to subtract.
            alpha (numbers.Number): A scaling factor for the subtraction.

        Returns:
            quaternion: This quaternion after subtraction.
        """
        assert isinstance(alpha, numbers.Number), "alpha is not a number"
        assert self.shape == p.shape, f"Shape mismatch: expected {self.shape}, got {p.shape}"

        if isinstance(p, Quaternion):
            self.quaternion = torch.sub(self.quaternion,
                                        p.quaternion,
                                        alpha=alpha)
        elif isinstance(p, numbers.Number):
            self.quaternion = torch.sub(self.quaternion, p, alpha=alpha)
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(p)}")

        return self

    def __radd__(self, p) -> 'Quaternion':
        return self + p

    def __iadd__(self, p) -> 'Quaternion':
        return self + p

    def __add__(self, p) -> 'Quaternion':
        return self.alpha_add(p=p)

    def alpha_add(self, p, alpha: int = 1) -> 'Quaternion':
        """
        Adds another quaternion or a scalar to this quaternion.

        Args:
            p (quaternion or numbers.Number): The quaternion or scalar to add.
            alpha (numbers.Number): A scaling factor for the addition.

        Returns:
            quaternion: This quaternion after addition.
        """
        assert isinstance(alpha, numbers.Number), "alpha is not a number"
        assert self.shape == p.shape, "can not create a random quaternion matrix. Please add a proper integer number"

        if isinstance(p, Quaternion):
            self.quaternion = torch.add(self.quaternion,
                                        p.quaternion,
                                        alpha=alpha)
        elif isinstance(p, numbers.Number):
            self.quaternion = torch.add(self.quaternion, p, alpha=alpha)
        else:
            raise TypeError("value is of wrong type")

        return self

    def __div__(self, p) -> 'Quaternion':
        if isinstance(p, Quaternion):
            self.quaternion = torch.div(self.quaternion, p.quaternion)
        elif isinstance(p, numbers.Number):
            self.quaternion = torch.div(self.quaternion, p)
        else:
            raise TypeError("data type not implemented")

        return self

    def __rdiv__(self, p) -> 'Quaternion':
        return self / p

    def __idiv__(self, p) -> 'Quaternion':
        return self / p

    def __truediv__(self, p) -> 'Quaternion':
        if isinstance(p, Quaternion):
            self.quaternion = torch.true_divide(self.quaternion, p.quaternion)
        elif isinstance(p, numbers.Number):
            self.quaternion = torch.true_divide(self.quaternion, p)
        else:
            raise TypeError("data type not implemented")

        return self

    def __rtruediv__(self, p) -> 'Quaternion':
        return self / p

    def __itruediv__(self, p) -> 'Quaternion':
        return self / p

    def __floordiv__(self, p) -> 'Quaternion':
        if isinstance(p, Quaternion):
            self.quaternion = torch.floor_divide(self.quaternion, p.quaternion)
        elif isinstance(p, numbers.Number):
            self.quaternion = torch.floor_divide(self.quaternion, p)
        else:
            raise TypeError("data type not implemented")

        return self

    def __imul__(self, p) -> 'Quaternion':
        return self * p

    def __rmul__(self, p) -> 'Quaternion':
        return self * p

    def __mul__(self, p) -> 'Quaternion':
        if isinstance(p, Quaternion):
            assert p.shape == self.shape, "incoming vector has to be the same dimension"

            # Compute outer product
            terms = torch.bmm(p.quaternion.view(-1, 4, 1),
                              self.quaternion.view(-1, 1, 4))

            w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3,
                                                                         3]
            x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3,
                                                                         2]
            y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3,
                                                                         1]
            z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3,
                                                                         0]

            self.quaternion = torch.stack((w, x, y, z), dim=1).view(self.shape)

        elif isinstance(p, numbers.Number):
            self.quaternion = torch.mul(self.quaternion, p)
        else:
            raise TypeError("value is of wrong type")

        return self

    def __matmul__(self, p) -> 'Quaternion':
        return torch.matmul(self, p)

    def __abs__(self) -> torch.Tensor:
        return self.norm

    @property
    def sq_sum(self) -> torch.Tensor:
        """
        Returns the squared sum of the quaternion elements.

        Returns:
            torch.Tensor: The squared sum of the quaternion elements.
        """
        return (self.quaternion * self.quaternion).sum(dim=1).float()

    @property
    def norm(self) -> torch.Tensor:
        """
        Returns the norm of the quaternion.

        Returns:
            torch.Tensor: The norm of the quaternion.
        """
        return self.quaternion.norm(p=2, dim=1,
                                    keepdim=True).float().squeeze().detach()

    @property
    def magnitude(self) -> torch.Tensor:
        """
        Returns the magnitude of the quaternion.

        Returns:
            torch.Tensor: The magnitude of the quaternion.
        """
        return self.norm

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the quaternion tensor.

        Returns:
            torch.Size: The shape of the quaternion tensor.
        """
        return self.quaternion.shape

    @property
    def np(self) -> np.ndarray:
        """
        Returns the quaternion as a numpy array.

        Returns:
            np.ndarray: The quaternion as a numpy array.
        """
        return self.quaternion.cpu().numpy()

    @property
    def elements(self) -> torch.Tensor:
        """
        Returns the quaternion tensor.

        Returns:
            torch.Tensor: The quaternion tensor.
        """
        return self.quaternion

    @property
    def real(self) -> torch.Tensor:
        """
        Returns the scalar part of the quaternion.

        Returns:
            torch.Tensor: The scalar part of the quaternion.
        """
        return self.scalar

    @property
    def imaginary(self) -> torch.Tensor:
        """
        Returns the vector part of the quaternion.

        Returns:
            torch.Tensor: The vector part of the quaternion.
        """
        return self.vector

    @property
    def vector(self) -> torch.Tensor:
        """
        Returns the vector part of the quaternion.

        Returns:
            torch.Tensor: The vector part of the quaternion.
        """
        return self.quaternion[:, 1:]

    @property
    def scalar(self) -> torch.Tensor:
        """
        Returns the scalar part of the quaternion.

        Returns:
            torch.Tensor: The scalar part of the quaternion.
        """
        return self.quaternion[:, 0]

    @classmethod
    def random(self, counter: int = 1) -> 'Quaternion':
        """
        Generates a random quaternion.

        Args:
            counter (int): The number of random quaternions to generate. Defaults to 1.

        Returns:
            quaternion: The generated random quaternion.
        """
        assert isinstance(
            counter, int
        ), "can not create a random quaternion matrix. Please add a proper integer number"

        if counter <= 0:
            counter = 1

        quat = Quaternion()

        if self.gpu:
            quat.quaternion = torch.rand(counter, 4).float().cuda()
        else:
            quat.quaternion = torch.rand(counter, 4).float().cpu()

        return quat

    @classproperty
    @classmethod
    def zero(cls) -> 'Quaternion':
        """
        Returns a quaternion with zero scalar part.

        Returns:
            quaternion: The quaternion with zero scalar part.
        """
        quat = Quaternion.one
        quat.quaternion[0, 0] = 0.0
        return quat

    @classproperty
    @classmethod
    def one(cls) -> 'Quaternion':
        """
        Returns a quaternion with one scalar part.

        Returns:
            quaternion: The quaternion with one scalar part.
        """
        quat = Quaternion(1)
        return quat

    @classproperty
    @classmethod
    def x_(cls) -> 'Quaternion':
        """
        Returns a quaternion with x component.

        Returns:
            quaternion: The quaternion with x component.
        """
        quat = Quaternion.zero
        quat.quaternion[0, 1]
        return quat

    @classproperty
    @classmethod
    def y_(cls) -> 'Quaternion':
        """
        Returns a quaternion with y component.

        Returns:
            quaternion: The quaternion with y component.
        """
        quat = Quaternion.zero
        quat.quaternion[0, 2]
        return quat

    @classproperty
    @classmethod
    def z_(cls) -> 'Quaternion':
        """
        Returns a quaternion with z component.

        Returns:
            quaternion: The quaternion with z component.
        """
        quat = Quaternion.zero
        quat.quaternion[0, 3]
        return quat

    @property
    def w(self) -> torch.Tensor:
        """
        Returns the w component of the quaternion.

        Returns:
            torch.Tensor: The w component of the quaternion.
        """
        return self.quaternion[:, 0]

    @property
    def x(self) -> torch.Tensor:
        """
        Returns the x component of the quaternion.

        Returns:
            torch.Tensor: The x component of the quaternion.
        """
        return self.quaternion[:, 1]

    @property
    def y(self) -> torch.Tensor:
        """
        Returns the y component of the quaternion.

        Returns:
            torch.Tensor: The y component of the quaternion.
        """
        return self.quaternion[:, 2]

    @property
    def z(self) -> torch.Tensor:
        """
        Returns the z component of the quaternion.

        Returns:
            torch.Tensor: The z component of the quaternion.
        """
        return self.quaternion[:, 3]

    @property
    def polar_unit_vector(self) -> torch.Tensor:
        """
        Returns the polar unit vector of the quaternion.

        Returns:
            torch.Tensor: The polar unit vector of the quaternion.
        """
        return torch.nn.functional.normalize(self.vector).detach().squeeze()

    @property
    def polar_angle(self) -> torch.Tensor:
        """
        Returns the polar angle of the quaternion.

        Returns:
            torch.Tensor: The polar angle of the quaternion.
        """
        return torch.acos(self.scalar / self.norm.T).detach().squeeze()

    @property
    def polar_decomposition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the polar decomposition of the quaternion.

        Returns:
            tuple: The polar unit vector and polar angle of the quaternion.
        """
        return self.polar_unit_vector, self.polar_angle

    def __str__(self) -> str:
        """
        Returns a string representation of the quaternion.

        Returns:
            str: The string representation of the quaternion.
        """
        return f"size is {self.__len__()}\n\n"

    def __repr__(self) -> str:
        """
        Returns a string representation of the quaternion.

        Returns:
            str: The string representation of the quaternion.
        """
        return f"size is {self.__len__()}\n\n"

    def __eq__(self, p) -> bool:
        """
        Checks if two quaternions are equal.

        Args:
            p (quaternion): The quaternion to compare.

        Returns:
            bool: True if the quaternions are equal, False otherwise.
        """
        return self.quaternion.equal(p.quaternion)

    def __neg__(self) -> 'Quaternion':
        """
        Negates the quaternion.

        Returns:
            quaternion: The negated quaternion.
        """
        return -1. * self

    def conjugate(self, inplace=True) -> 'Quaternion':
        """
        Computes the conjugate of the quaternion.

        Args:
            inplace (bool): If True, modify the quaternion in place. Otherwise, return a new quaternion.

        Returns:
            quaternion: The conjugated quaternion.
        """
        if inplace:
            self.quaternion[..., 1:] *= -1
            return self
        else:
            new_quat = Quaternion(self.shape[0])
            new_quat.quaternion = self.quaternion.clone()
            new_quat.quaternion[..., 1:] *= -1
            return new_quat

    def __pow__(self, exponent) -> 'Quaternion':
        """
        Raises the quaternion to a power.

        Args:
            exponent (float): The exponent to raise the quaternion to.

        Returns:
            quaternion: The quaternion raised to the power.
        """
        exponent = float(exponent)
        norm = self.norm
        n, theta = self.polar_decomposition

        tmp = (norm**exponent)

        self.quaternion[..., 0] = tmp * torch.cos(exponent * theta)
        self.quaternion[..., 1:] = (tmp * n.T * torch.sin(exponent * theta)).T

        return self

    def __ipow__(self, other) -> 'Quaternion':
        return self**other

    def __rpow__(self, other) -> 'Quaternion':
        return other**float(self)

    def rotate_quaternion(self, p) -> 'Quaternion':
        """
        Rotates the quaternion by another quaternion.

        Args:
            p (quaternion): The quaternion to rotate by.

        Returns:
            quaternion: The rotated quaternion.
        """
        assert torch.count_nonzero(
            p.scalar) == 0, "p has to be of form p[...,0]=0"

        if not self.is_unit():
            _ = self.normalize
        return self * p * self.conjugate

    @staticmethod
    def from_euler(roll, pitch, yaw) -> 'Quaternion':
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        quat = Quaternion(size=1)
        quat.quaternion = torch.tensor([[w, x, y, z]])
        return quat
    
    @staticmethod
    def from_rodrigues(axis: torch.Tensor, angle: float) -> 'Quaternion':
        """Konvertiert eine Rodrigues-Darstellung (Achse + Winkel) in ein Quaternion.
        
        Args:
            axis (torch.Tensor): Die Rotationsachse (3D-Vektor).
            angle (float): Der Rotationswinkel in Radiant.
        
        Returns:
            quaternion: Das resultierende Quaternion.
        """
        axis = axis / torch.norm(axis)  # Normalisiere die Achse
        half_angle = angle / 2.0
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle)

        quat = Quaternion(size=1)
        quat.quaternion = torch.cat([w.unsqueeze(0), xyz], dim=0).unsqueeze(0)
        return quat

    def to_rodrigues(self) -> Tuple[torch.Tensor, float]:
        """Konvertiert ein Quaternion in die Rodrigues-Darstellung (Achse + Winkel).
        
        Returns:
            Tuple[torch.Tensor, float]: Die Rotationsachse (3D-Vektor) und der Winkel in Radiant.
        """
        w = self.quaternion[:, 0]
        xyz = self.quaternion[:, 1:]

        angle = 2 * torch.acos(w)
        axis = xyz / torch.norm(xyz, dim=1, keepdim=True)

        return axis.squeeze(), angle.squeeze()

    def to_euler(self) -> Tuple[float, float, float]:
        """Konvertiert ein Quaternion in Euler-Winkel (Roll, Pitch, Yaw).
        
            Returns:
                Tuple[float, float, float]: Roll-, Pitch- und Yaw-Winkel in Radiant.
        """
        w, x, y, z = self.quaternion[0]

        # Roll (x-Achse)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        # Pitch (y-Achse)
        pitch = torch.asin(2 * (w * y - z * x))

        # Yaw (z-Achse)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return roll.item(), pitch.item(), yaw.item()

    def rotate_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Rotiert einen 3D-Vektor mit der aktuellen Quaternion.

        Args:
            vector (torch.Tensor): Der zu rotierende 3D-Vektor.

        Returns:
            torch.Tensor: Der rotierte Vektor.
        """
        assert vector.shape == (3,), "Der Eingabevektor muss eine Länge von 3 haben"

        # Erzeuge ein Quaternion für den Vektor (0, x, y, z)
        v_quat = Quaternion(size=1)
        v_quat.quaternion = torch.cat((torch.tensor([0.0]), vector)).unsqueeze(0)  # (1, 4) Tensor

        # Berechne die Rotation: q * v * q^-1
        q_conjugate = self.conjugate(inplace=False)  # Konjugierte Quaternion
        rotated_vector = self * v_quat * q_conjugate  # Quaternionen-Multiplikation

        return rotated_vector.quaternion[:, 1:]  # Gib nur den Vektoranteil zurück
