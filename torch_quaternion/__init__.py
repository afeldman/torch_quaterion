"""
torch_quaternion
================

A library for quaternion and dual quaternion operations using PyTorch.

Author: Anton Feldmann
Email: anton.feldmann@gmail.com
Version: 0.2.0
"""

__author__ = "Anton Feldmann"
__email__ = "anton.feldmann@gmail.com"
__version__ = (0, 2, 0)

from .quaternion import Quaternion
from .dualQuaternion import DualQuaternion

__all__ = ["Quaternion", "DualQuaternion"]
