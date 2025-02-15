from torch_quaternion.quaternion import Quaternion

class DualQuaternion(Quaternion):
    """
    A class to represent a dual quaternion, which is used for representing rigid body transformations.

    Attributes:
        real (Quaternion): The real part of the dual quaternion (rotation).
        dual (Quaternion): The dual part of the dual quaternion (translation).

    Methods:
        __init__(real_part: Quaternion, dual_part: Quaternion): Initializes the dual quaternion.
        __mul__(other): Multiplies this dual quaternion by another dual quaternion.
        conjugate(): Computes the conjugate of the dual quaternion.
        norm(): Computes the norm of the dual quaternion.
        normalize(): Normalizes the dual quaternion.
        __str__(): Returns a string representation of the dual quaternion.
    """

    def __init__(self, real_part: Quaternion, dual_part: Quaternion):
        """
        Initializes the dual quaternion.

        Args:
            real_part (Quaternion): The real part of the dual quaternion (rotation).
            dual_part (Quaternion): The dual part of the dual quaternion (translation).
        """
        super().__init__(size=real_part.shape[0])  # init base
        self.real = real_part  # real part (rotation)
        self.dual = dual_part  # dual part (translation)

    def __mul__(self, other):
        """
        Multiplies this dual quaternion by another dual quaternion.

        Args:
            other (DualQuaternion): The dual quaternion to multiply with.

        Returns:
            DualQuaternion: The result of the multiplication.
        """
        if isinstance(other, DualQuaternion):
            real = self.real * other.real
            dual = self.real * other.dual + self.dual * other.real
            return DualQuaternion(real, dual)
        else:
            raise TypeError("Unsupported type for multiplication")

    def conjugate(self):
        """
        Computes the conjugate of the dual quaternion.

        Returns:
            DualQuaternion: The conjugated dual quaternion.
        """
        real_conj = self.real.conjugate()
        dual_conj = self.dual.conjugate()
        return DualQuaternion(real_conj, dual_conj)

    def norm(self):
        """
        Computes the norm of the dual quaternion.

        Returns:
            torch.Tensor: The norm of the dual quaternion.
        """
        real_norm = self.real.norm()
        dual_norm = (self.real.conjugate() * self.dual + self.dual.conjugate() * self.real).scalar
        return real_norm + dual_norm

    def normalize(self):
        """
        Normalizes the dual quaternion.

        Returns:
            DualQuaternion: The normalized dual quaternion.
        """
        norm = self.norm()
        self.real = self.real / norm
        self.dual = self.dual / norm
        return self

    def __str__(self):
        """
        Returns a string representation of the dual quaternion.

        Returns:
            str: The string representation of the dual quaternion.
        """
        return f"Real: {self.real}\nDual: {self.dual}"
