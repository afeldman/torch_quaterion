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
   git clone https://github.com/afeldman/torch-quaternion.git
   cd torch-quaternion
   ```

2. Install the dependencies:
   ```bash
   poetry install
   ```

## Usage

### 1. **Calculation of Moments and Forces**
We extend the `Wrench` class to calculate forces and moments acting on the end effector.

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

---

### 4. **Trajectory Planning**
You can use the `Trajectory` class to plan and visualize trajectories for the end effector.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
