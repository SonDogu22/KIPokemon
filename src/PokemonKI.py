from google.colab import files
import retro, shutil, os, json

print('📂 Lade deine PokemonEmerald.gba hoch...')
uploaded = files.upload()

rom_filename = list(uploaded.keys())[0]
game_dir = os.path.join(retro.data.path(), 'stable', 'PokemonEmerald-GbAdvance')
os.makedirs(game_dir, exist_ok=True)
shutil.copy(rom_filename, os.path.join(game_dir, 'rom.gba'))
with open(os.path.join(game_dir, 'metadata.json'), 'w') as f:
    json.dump({'default_player_value': 1, 'rom.gba': {'sha': ''}}, f)

print('✅ ROM eingebunden!')

import retro
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces

class PokemonEmeraldEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = retro.make(
            game='PokemonEmerald-GbAdvance',
            use_restricted_actions=retro.Actions.FILTERED
        )
        self.action_space = spaces.Discrete(8)
        self.action_map = [
            [0,0,0,0,0,0,0,0,0,0],  # Nichts
            [0,1,0,0,0,0,0,0,0,0],  # A
            [1,0,0,0,0,0,0,0,0,0],  # B
            [0,0,0,1,0,0,0,0,0,0],  # Start
            [0,0,0,0,1,0,0,0,0,0],  # Rechts
            [0,0,0,0,0,1,0,0,0,0],  # Links
            [0,0,0,0,0,0,1,0,0,0],  # Hoch
            [0,0,0,0,0,0,0,1,0,0],  # Runter
        ]
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.visited_positions = set()
        self.last_badge_count = 0
        self.last_level = 0
        self.steps = 0
        self.max_steps = 10000

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84))[:, :, np.newaxis]

    def _get_reward(self, info):
        reward = -0.001
        pos = (info.get('x', 0) // 16, info.get('y', 0) // 16)
        if pos not in self.visited_positions:
            self.visited_positions.add(pos)
            reward += 1.0
        level = info.get('party_level_1', 0)
        if level > self.last_level:
            reward += 5.0 * (level - self.last_level)
            self.last_level = level
        badges = bin(info.get('badges', 0)).count('1')
        if badges > self.last_badge_count:
            reward += 50.0 * (badges - self.last_badge_count)
            self.last_badge_count = badges
            print(f'🏅 Orden! Gesamt: {badges}')
        return reward

    def step(self, action):
        total_reward = 0
        for _ in range(4):
            obs, _, terminated, truncated, info = self.env.step(self.action_map[action])
            total_reward += self._get_reward(info)
        self.steps += 1
        return self._process_frame(obs), total_reward, terminated, self.steps >= self.max_steps, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset()
        self.visited_positions = set()
        self.last_badge_count = 0
        self.last_level = 0
        self.steps = 0
        return self._process_frame(obs), info

    def render(self): return self.env.render()
    def close(self): self.env.close()

# Test
env = None # Initialize env to None
try:
    env = PokemonEmeraldEnv()
    obs, _ = env.reset()
    print(f'✅ Funktioniert! Shape: {obs.shape}')
finally:
    if env:
        env.close()