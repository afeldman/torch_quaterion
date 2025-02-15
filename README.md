# Torch Quaternion

Torch Quaternion is a library for implementing screw theory and kinematics for robots using quaternions and dual quaternions. This library is designed to work with PyTorch and provides functionalities for forward kinematics, inverse kinematics, and dynamics calculations.

## Features

- **Forward Kinematics**: Calculate the end effector pose based on joint angles.
- **Inverse Kinematics**: Calculate the joint angles for a desired end effector pose.
- **Trajectory Planning**: Plan and visualize trajectories for the end effector.
- **Dynamics**: Calculate forces, moments, and joint loads.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/torch-quaternion.git
   cd torch-quaternion
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. **Calculation of Moments and Forces**
We extend the `Wrench` class to calculate forces and moments acting on the end effector.

#### Extended `Wrench` Class
```python
class Wrench(DualQuaternion):
    def __init__(self, force: torch.Tensor, torque: torch.Tensor):
        """Initializes a Wrench from force and torque.
        
        Args:
            force (torch.Tensor): Force (3D vector).
            torque (torch.Tensor): Torque (3D vector).
        """
        # Real part: Force
        real_part = Quaterion(size=1)
        real_part.quaterion = torch.cat([torch.zeros(1, 1), force.unsqueeze(0)], dim=1)

        # Dual part: Torque
        dual_part = Quaterion(size=1)
        dual_part.quaterion = torch.cat([torch.zeros(1, 1), torque.unsqueeze(0)], dim=1)

        super().__init__(real_part, dual_part)

    @property
    def force(self) -> torch.Tensor:
        """Returns the force."""
        return self.real.vector

    @property
    def torque(self) -> torch.Tensor:
        """Returns the torque."""
        return self.dual.vector
```

---

### 2. **Example: Calculation of Moments and Forces**
Here is an example of how to use the `Wrench` class to calculate the forces and moments acting on the end effector:

```python
# Example: Force and Torque
force = torch.tensor([0.0, 10.0, 0.0])  # 10 N along the y-axis
torque = torch.tensor([0.0, 0.0, 5.0])  # 5 Nm along the z-axis

# Create Wrench
wrench = Wrench(force, torque)

# Display force and torque
print("Force:", wrench.force)
print("Torque:", wrench.torque)
```

---

### 3. **Integration into Kinematics**
You can use the `Wrench` class to account for forces and moments in kinematics. For example, you could calculate the joint loads:

```python
class Dynamics:
    def __init__(self, mass: float, inertia: torch.Tensor):
        """Initializes the dynamics with mass and inertia tensor.
        
        Args:
            mass (float): Mass of the robot.
            inertia (torch.Tensor): Inertia tensor of the robot.
        """
        self.mass = mass
        self.inertia = inertia

    def compute_joint_loads(self, wrench: Wrench, jacobian: torch.Tensor) -> torch.Tensor:
        """Calculates the joint loads based on a Wrench and the Jacobian matrix.
        
        Args:
            wrench (Wrench): The Wrench acting on the end effector.
            jacobian (torch.Tensor): The Jacobian matrix of the robot.
        
        Returns:
            torch.Tensor: The joint loads.
        """
        # Convert Wrench to a 6D vector
        wrench_vector = torch.cat([wrench.force, wrench.torque])

        # Calculate joint loads
        joint_loads = torch.matmul(jacobian.T, wrench_vector)
        return joint_loads
```

---

### 4. **Trajectory Planning**
You can use the `Trajectory` class to plan and visualize trajectories for the end effector.

#### Trajectory Class
```python
class Trajectory:
    def __init__(self, twists: List[Twist], link_lengths: torch.Tensor, initial_pose: DualQuaternion):
        """Initializes the Trajectory class.
        
        Args:
            twists (List[Twist]): List of twists for each joint.
            link_lengths (torch.Tensor): Lengths of the robot arms.
            initial_pose (DualQuaternion): Initial pose of the end effector.
        """
        self.twists = twists
        self.link_lengths = link_lengths
        self.initial_pose = initial_pose
        self.fk = ForwardKinematics(twists=twists, initial_pose=initial_pose)
        self.ik = InverseKinematics(twists=twists, target_pose=initial_pose)

    def plan_trajectory(self, start_pose: DualQuaternion, end_pose: DualQuaternion, num_points: int) -> List[DualQuaternion]:
        """Plans a trajectory between two poses.
        
        Args:
            start_pose (DualQuaternion): Start pose.
            end_pose (DualQuaternion): Target pose.
            num_points (int): Number of intermediate points.
        
        Returns:
            List[DualQuaternion]: List of poses along the trajectory.
        """
        trajectory = []
        for t in torch.linspace(0, 1, num_points):
            pose = self.interpolate_poses(start_pose, end_pose, t)
            trajectory.append(pose)
        return trajectory

    def interpolate_poses(self, pose1: DualQuaternion, pose2: DualQuaternion, t: float) -> DualQuaternion:
        """Interpolates between two poses.
        
        Args:
            pose1 (DualQuaternion): First pose.
            pose2 (DualQuaternion): Second pose.
            t (float): Interpolation factor (0 <= t <= 1).
        
        Returns:
            DualQuaternion: Interpolated pose.
        """
        return pose1 * (1 - t) + pose2 * t

    def compute_joint_angles(self, trajectory: List[DualQuaternion]) -> List[torch.Tensor]:
        """Computes the joint angles for each pose in the trajectory.
        
        Args:
            trajectory (List[DualQuaternion]): List of poses along the trajectory.
        
        Returns:
            List[torch.Tensor]: List of joint angles for each pose.
        """
        joint_angles = []
        initial_angles = torch.zeros(len(self.twists))
        for pose in trajectory:
            self.ik.target_pose = pose
            angles = self.ik.solve(initial_angles)
            joint_angles.append(angles)
            initial_angles = angles
        return joint_angles

    def visualize_trajectory(self, trajectory: List[DualQuaternion]):
        """Visualizes the trajectory.
        
        Args:
            trajectory (List[DualQuaternion]): List of poses along the trajectory.
        """
        import matplotlib.pyplot as plt

        positions = [pose.translation for pose in trajectory]
        x = [pos[0].item() for pos in positions]
        y = [pos[1].item() for pos in positions]
        z = [pos[2].item() for pos in positions]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, '-o')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Trajectory of the End Effector")
        plt.show()
```

---

### 5. **Example: Trajectory Planning**
Here is an example of how to use the `Trajectory` class to plan and visualize a trajectory:

```python
# Example: Trajectory Planning
twists = [
    Twist(torch.tensor([0.0, 0.0, 1.0])),  # Joint 1
    Twist(torch.tensor([0.0, 1.0, 0.0])),  # Joint 2
    # ...
]

link_lengths = torch.tensor([1.0, 1.0])  # Example link lengths
initial_pose = DualQuaternion.identity()

trajectory_planner = Trajectory(twists, link_lengths, initial_pose)

start_pose = DualQuaternion.identity()
end_pose = DualQuaternion.from_translation(torch.tensor([1.0, 1.0, 1.0]))
num_points = 10

trajectory = trajectory_planner.plan_trajectory(start_pose, end_pose, num_points)
trajectory_planner.visualize_trajectory(trajectory)
```

---

### 6. **Documentation in the `README.md**
Here is an example of a `README.md`, which describes the functionalities of your project:

```markdown
# Torch-Quaternion: Screw Theory and Kinematics

This project implements **Screw Theory** and **Kinematics** for robots like the **Fanuc M-10iD/10L** using **quaternions** and **dual quaternions**. It includes:

- **Forward Kinematics**: Calculation of the end effector pose based on joint angles.
- **Inverse Kinematics**: Calculation of the joint angles for a desired end effector pose.
- **Trajectory Planning**: Plan and visualize trajectories for the end effector.
- **Dynamics**: Calculation of forces, moments, and joint loads.

## Classes

### `Quaternion`
A class for representing quaternions with PyTorch as the backend.

### `DualQuaternion`
A class for representing dual quaternions used in screw theory.

### `Twist`
A class for representing twists (rotational and translational velocities).

### `Wrench`
A class for representing wrenches (forces and moments).

### `ForwardKinematics`
A class for calculating forward kinematics.

### `InverseKinematics`
A class for calculating inverse kinematics.

### `Trajectory`
A class for planning and visualizing trajectories.

### `Dynamics`
A class for calculating dynamics, including joint loads.

## Example

```python
# Twists for each joint
twists = [
    Twist(torch.tensor([0.0, 0.0, 1.0])),  # Joint 1
    Twist(torch.tensor([0.0, 1.0, 0.0])),  # Joint 2
    # ...
]

# Calculate forward kinematics
fk = ForwardKinematics(twists=twists, initial_pose=DualQuaternion.one)
joint_angles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
end_effector_pose = fk.compute_pose(joint_angles)

# Calculate inverse kinematics
ik = InverseKinematics(twists=twists, target_pose=end_effector_pose)
solved_angles = ik.solve(initial_angles=torch.zeros(6))

# Plan and visualize trajectory
trajectory_planner = Trajectory(twists, torch.tensor([1.0, 1.0]), DualQuaternion.identity())
trajectory = trajectory_planner.plan_trajectory(DualQuaternion.identity(), DualQuaternion.from_translation(torch.tensor([1.0, 1.0, 1.0])), 10)
trajectory_planner.visualize_trajectory(trajectory)

# Calculate wrench
force = torch.tensor([0.0, 10.0, 0.0])  # 10 N along the y-axis
torque = torch.tensor([0.0, 0.0, 5.0])  # 5 Nm along the z-axis
wrench = Wrench(force, torque)

# Calculate joint loads
dynamics = Dynamics(mass=1.0, inertia=torch.eye(3))
jacobian = torch.rand(6, 6)  # Example Jacobian matrix
joint_loads = dynamics.compute_joint_loads(wrench, jacobian)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/torch-quaternion.git
   cd torch-quaternion
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### 7. **Conclusion**
With these extensions, you can integrate the calculation of **moments** and **forces** into your project, plan and visualize **trajectories**, and document the functionalities in a `README.md`. This makes your project clear and easy to understand for other developers. Good luck with the implementation! 🚀
