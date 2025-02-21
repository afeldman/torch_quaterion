import torch
from typing import Tuple
import numpy as np
from loguru import logger
import paho.mqtt.client as mqtt
import json
import time

from torch_quaternion.quaternion import Quaternion  # Import your Quaternion class

class AckermannKinematics:
    """
    Simulates vehicle motion using the Ackermann steering model with quaternions.
    """

    def __init__(self, wheelbase: float):
        """
        Initializes the Ackermann kinematics model.

        Args:
            wheelbase (float): Wheelbase of the vehicle in meters.
        """
        self.wheelbase = wheelbase  # Wheelbase of the vehicle in meters

    def update(self, position: torch.Tensor, 
               quaternion: Quaternion, 
               velocity: float, 
               steering_angle: float, 
               dt: float) -> Tuple[torch.Tensor, Quaternion]:
        """
        Updates the vehicle position and orientation based on the Ackermann model.

        Args:
            position (torch.Tensor): Current position of the vehicle as a (3,)-tensor.
            quaternion (Quaternion): Current orientation of the vehicle as a quaternion.
            velocity (float): Speed of the vehicle.
            steering_angle (float): Steering angle in radians.
            dt (float): Time step for integration.

        Returns:
            Tuple[torch.Tensor, Quaternion]: New position and new quaternion.
        """
        # Convert steering angle to tensor if needed
        steering_angle = torch.tensor(steering_angle, dtype=torch.float32)

        # Calculate the angular velocity
        theta_dot = (velocity / self.wheelbase) * torch.tan(steering_angle)
        d_theta = theta_dot * dt  # Change in orientation

        # Quaternion for the small rotation
        delta_q = Quaternion.from_rodrigues(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32), d_theta)

        # New orientation through quaternion multiplication
        new_quaternion = quaternion * delta_q

        # Direction of the vehicle (X-axis of the vehicle coordinate system)
        forward_vector = quaternion.rotate_vector(
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32))

        # Calculate new position
        new_position = position + velocity * forward_vector * dt

        return new_position, new_quaternion

    def update_from_mqtt(self, mqtt_client: mqtt.Client, topic: str, dt: float):
        """
        Continuously updates the vehicle state based on MQTT messages.

        Args:
            mqtt_client (mqtt.Client): The MQTT client instance.
            topic (str): MQTT topic to subscribe to.
            dt (float): Time step for integration.
        """
        velocity = 0.
        steering_angle = 0.
        quaternion = Quaternion(size=1)
        position = torch.tensor([0.0, 0.0, 0.0])

        def on_message(client, userdata, message):
            """
            Callback function to handle incoming MQTT messages.
            """
            try:
                # Parse JSON payload
                data = json.loads(message.payload.decode())

                # Update vehicle state
                velocity = float(data.get("velocity", 0.0))
                steering_angle = float(data.get("steering_angle", 0.0))

            except Exception as e:
                print(f"Error parsing MQTT message: {e}")

        # Set MQTT message callback
        mqtt_client.on_message = on_message
        mqtt_client.subscribe(topic)
        mqtt_client.loop_start()

        print(f"Subscribed to MQTT topic: {topic}")

        # Continuous loop to update vehicle state
        try:
            while True:
                self.update(
                    position=position,
                    quaternion=quaternion,
                    velocity=velocity,
                    steering_angle=steering_angle,
                    dt=dt)
                logger.debug(f"Position: {self.position.numpy()}, Quaternion: {self.quaternion.quaternion.numpy()}")
                time.sleep(dt)
        except KeyboardInterrupt:
            logger.debug("MQTT Update Stopped.")
            mqtt_client.loop_stop()

if __name__ == "__main__":
    # Test the Ackermann kinematics
    wheelbase = 3.5  # Wheelbase of the vehicle in meters
    ackermann = AckermannKinematics(wheelbase)

    velocity = 5.  # Speed of the vehicle in m/s
    steering_angle = 0.1  # Steering angle in radians (~5.7 degrees)
    dt = 0.1  # Time step for integration

    # Initial position and orientation
    position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    quaternion = Quaternion(size=1)

    # Simulate the motion for 5 seconds
    positions = []
    for i in range(500):
        pos = position.numpy()
        if position.numpy().shape[0] == 1:
            pos = position.numpy()[0]
            
        positions.append(pos[:2])
        position, quaternion = ackermann.update(position, quaternion, velocity, steering_angle, dt)
        
    print("\nFinal State:")
    print("New position:", position.numpy())
    print("New orientation:", quaternion.quaternion.numpy())

    positions = np.array(positions)

    # Visualisierung
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(positions[:, 0], positions[:, 1], 'bo-', markersize=2, label="Fahrzeugtrajektorie")
    plt.xlabel("X-Position (m)")
    plt.ylabel("Y-Position (m)")
    plt.title("Fahrzeugbewegung mit Ackermann-Kinematik")
    plt.legend()
    plt.grid()
    plt.show()