import gymnasium as gym, spaces
import numpy as np
from stable_baselines3 import PPO

class AckermannEnv(gym.Env):
    """
    Gymnasium-Umgebung für Ackermann-Kinematik mit RL.
    Ziel: Fahrzeug soll eine Kurve fahren.
    """
    def __init__(self):
        super(AckermannEnv, self).__init__()

        # Positions- und Orientierungsraum
        self.observation_space = spaces.Box(low=-100, high=100, shape=(5,), dtype=np.float32)

        # Aktionen: Geschwindigkeit und Lenkwinkel
        self.action_space = spaces.Box(low=np.array([0, -0.5]), high=np.array([10, 0.5]), dtype=np.float32)

        # Initialisiere Fahrzeug
        self.wheelbase = 2.5
        self.position = torch.tensor([0.0, 0.0, 0.0])
        self.orientation = Quaternion(size=1)
        self.ackermann = AckermannKinematics(self.wheelbase)

    def step(self, action):
        velocity, steering_angle = action
        self.position, self.orientation = self.ackermann.update(self.position, self.orientation, velocity, steering_angle, dt=0.1)

        # Reward: Belohne, wenn das Fahrzeug sich entlang der x-Achse bewegt
        reward = self.position[0].item() - abs(self.position[1].item()) * 0.1  # Geradeausfahrt belohnen

        # Beobachtung: Position + Quaternion
        obs = np.concatenate([self.position.numpy()[:2], self.orientation.quaternion.numpy().flatten()])

        done = self.position[0].item() > 50  # Stoppe nach 50m
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = torch.tensor([0.0, 0.0, 0.0])
        self.orientation = Quaternion(size=1)
        obs = np.concatenate([self.position.numpy()[:2], self.orientation.quaternion.numpy().flatten()])
        return obs, {}
if __name__ == "__main__":
    # Teste die Ackermann-Umgebung
    env = AckermannEnv()
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(f"Position {obs[:2]}, Quaternion {obs[2:]}")
        if done:
            break

